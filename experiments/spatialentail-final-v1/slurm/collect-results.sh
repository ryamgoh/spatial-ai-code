#!/usr/bin/env bash
# Submit from the repository root:
# sbatch experiments/spatialentail-final-v1/slurm/collect-results.sh [evaluation-directory]
# No model loading or GPU allocation; running evaluations are left untouched.
#SBATCH --job-name=spatialentail-results
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=spatialentail-results-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch from the repository root}"
EVALUATION=${1:-experiments/spatialentail-final-v1/runs/sft-y63EhpI5/campaign/evaluation/final-v1-evaluation-8192-fewshot-v1}
command -v python3 >/dev/null
python3 - "$EVALUATION" "$SLURM_JOB_ID" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

evaluation = Path(sys.argv[1]).resolve()
if not evaluation.is_dir():
    raise SystemExit(f"Evaluation directory does not exist: {evaluation}")
policy = "final-v1-evaluation-8192-fewshot-v1"
if evaluation.name != policy or evaluation.parent.name != "evaluation":
    raise SystemExit(f"Expected campaign/evaluation/{policy}, got {evaluation}")
campaign = evaluation.parent.parent
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
destination = campaign / "result-reports" / f"results-{stamp}-job-{sys.argv[2]}"
destination.mkdir(parents=True, exist_ok=False)
report = {
    "collected_at_utc": stamp,
    "evaluation_directory": str(evaluation),
    "note": "Live snapshot; missing summaries are not final results or failed jobs.",
    "results": [],
    "training": {},
}
lines = [
    f"SpatialEntail final-v1 results ({stamp})",
    str(evaluation),
    "Live snapshot: pending results are NOT scored as zero.",
    "S1/S2 = micro accuracy; macro accuracy and all slices are in report.json.",
    "",
    "Model | Style | Benchmark | Status | Completed | S1 | S2 | S1 mean tokens | Length finishes",
]


def snapshot(path):
    """Copy small receipts once so the report and bundled evidence agree."""
    content = path.read_bytes()
    target = destination / "evidence" / path.relative_to(campaign)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return json.loads(content)


def percentage(scores, stage):
    metrics = scores[stage]
    return f"{100 * metrics['correct'] / metrics['total']:.2f}%"


models = [
    (f"{style}-seed-{seed}", (style,))
    for seed in (42, 43)
    for style in ("natural", "symbolic")
] + [(f"untuned-{shots}-shot", ("natural", "symbolic")) for shots in (0, 1, 3)]
for model, styles in models:
    for style in styles:
        for benchmark, total in (("sft-bench", 3600), ("external-test", 1500)):
            base = evaluation / model / f"{style}-{benchmark}"
            summary_path = base.with_suffix(".summary.json")
            result_path = base.with_suffix(".json")
            partial_path = base.with_suffix(".partial.jsonl")
            entry = {
                "model": model, "style": style, "benchmark": benchmark,
                "expected": total, "completed": 0, "status": "pending",
            }
            display = ["-", "-", "-", "-"]
            try:
                if summary_path.exists():
                    summary = snapshot(summary_path)
                    entry["summary"] = summary
                    entry["completed"] = summary["scores"]["stage1"]["total"]
                    # Raw responses can be large; validate their digest, but do not bundle them.
                    if not result_path.exists():
                        raise ValueError("Summary exists but raw result is missing")
                    sha = hashlib.sha256()
                    with result_path.open("rb") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            sha.update(chunk)
                    if sha.hexdigest() != summary["result_sha256"]:
                        raise ValueError("Raw result SHA256 does not match summary")
                    if entry["completed"] != total or summary["scores"]["stage2"]["total"] != total:
                        raise ValueError("Summary count differs from expected benchmark size")
                    if summary["style"] != style or summary["evaluation"]["policy"]["version"] != policy:
                        raise ValueError("Summary style or evaluation policy differs")
                    seed = None if model.startswith("untuned-") else int(model.rsplit("-", 1)[1])
                    shots = int(model.split("-")[1]) if seed is None else 0
                    if summary["training_seed"] != seed or summary["evaluation"]["shots"] != shots:
                        raise ValueError("Summary training seed or shot setting differs")
                    entry["status"] = "complete"
                    tokens = summary["token_profile"]["stage1"]
                    display = [
                        percentage(summary["scores"], "stage1"),
                        percentage(summary["scores"], "stage2"),
                        f"{tokens['mean']:.1f}", str(tokens["length_finish_count"]),
                    ]
                elif result_path.exists():
                    entry["status"] = "awaiting-summary"
                elif partial_path.exists():
                    entry["status"] = "partial"
                    # Bound the read to the starting size; ignore a write interrupted mid-line.
                    with partial_path.open("rb") as source:
                        remaining = os.fstat(source.fileno()).st_size
                        while remaining:
                            line = source.readline(remaining)
                            if not line:
                                break
                            remaining -= len(line)
                            if line.endswith(b"\n"):
                                json.loads(line)
                                entry["completed"] += 1
                    if entry["completed"] > total:
                        raise ValueError("Partial journal exceeds expected benchmark size")
                receipt = base.with_suffix(".resume.json")
                if receipt.exists():
                    entry["resume"] = snapshot(receipt)
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)
                display = ["-", "-", "-", "-"]
            report["results"].append(entry)
            lines.append(" | ".join([
                model, style, benchmark, entry["status"],
                f"{entry['completed']}/{total}", *display,
            ]))
            if "error" in entry:
                lines.append(f"  ERROR: {entry['error']}")

for model, _ in models[:4]:
    receipts = {}
    for name in ("selection.json", "dev-epoch-1.json", "dev-epoch-2.json"):
        path = campaign / model / name
        if path.exists():
            try:
                receipts[name] = snapshot(path)
            except (OSError, ValueError) as exc:
                receipts[name] = {"error": str(exc)}
    report["training"][model] = receipts
    selection = receipts.get("selection.json", {})
    lines.append(f"Selected epoch {model}: {selection.get('epoch', 'unavailable')}")

# Submission records identify evaluation jobs. Capture bounded log tails, not full logs.
job_ids = set()
for submission in sorted(campaign.glob("eval-submission-*")):
    if not submission.is_dir():
        continue
    for path in sorted(submission.iterdir()):
        if not path.is_file() or path.suffix not in (".txt", ".out", ".err"):
            continue
        target = destination / "evidence" / path.relative_to(campaign)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            source.seek(max(0, size - 32768))
            content = source.read(32768)
        target.write_bytes(content)
        if path.name == "jobs.txt":
            for line in content.decode(errors="replace").splitlines():
                value = line.partition("=")[2]
                if value.isdigit():
                    job_ids.add(value)

commands = [["squeue", "-u", os.environ.get("USER", ""), "-o", "%.18i %.40j %.10T %.12M %.6D %R"]]
if job_ids:
    commands.append([
        "sacct", "-j", ",".join(sorted(job_ids)), "--parsable2",
        "--format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS",
    ])
for command in commands:
    name = command[0]
    if not shutil.which(name):
        report[name] = "Command unavailable"
        continue
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=30)
        report[name] = {
            "returncode": process.returncode, "stdout": process.stdout,
            "stderr": process.stderr,
        }
        (destination / f"{name}.txt").write_text(process.stdout + process.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report[name] = str(exc)

complete = sum(entry["status"] == "complete" for entry in report["results"])
lines.extend(["", f"Completed summaries: {complete}/20",
              "Full subgroup, depth, format, token and provenance details: report.json"])
text = "\n".join(lines) + "\n"
(destination / "report.txt").write_text(text)
(destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
archive = destination.with_suffix(".tar.gz")
with tarfile.open(archive, "w:gz") as bundle:
    bundle.add(destination, arcname=destination.name)
print(text)
print(f"Send back this archive: {archive}")
print(f"Or paste this report: {destination / 'report.txt'}")
PY
