"""Prepare six SFT jobs from a verified release; never submit Slurm jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from spatial.v2.study.seal import seal_release

PACKAGES = {
    "answer-only": ("natural", "answer-only"),
    "natural": ("natural", "checked-trace"),
    "symbolic": ("symbolic", "checked-trace"),
}


def prepare_jobs(release: Path, output: Path, *, repo_root: Path | None = None) -> dict:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    release = release.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"job output already exists: {output}")
    if not release.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError(
            "release and jobs must live inside the transferable repository"
        )
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "spatialentail-study-release-v1"
        or manifest.get("status") != "sealed"
    ):
        raise ValueError("SFT jobs require a sealed study release")
    audit_path = release / "release-audit.json"
    if not audit_path.is_file() or hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest() != manifest.get("release_audit_sha256"):
        raise ValueError("release audit is missing or changed")
    index_path = release / "quality/index.json"
    if not index_path.is_file() or hashlib.sha256(
        index_path.read_bytes()
    ).hexdigest() != manifest.get("review_index_sha256"):
        raise ValueError("sealed quality index is missing or changed")
    verified = seal_release(release, index_path, verify_only=True)
    if verified != json.loads(audit_path.read_text()):
        raise ValueError("sealed audit does not match revalidated quality receipts")
    budget = manifest["context_budget"]
    if not budget.get("tokenizer_revision"):
        raise ValueError("tokenizer/model revision must be pinned")
    rows_by_role = {}
    for role in ("train", "dev"):
        path = release / (role + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            raise ValueError(f"{role} set must be nonempty")
        rows_by_role[role] = rows
    views = {}
    for role, rows in rows_by_role.items():
        groups = {}
        for row in rows:
            groups.setdefault(row["metadata"]["base_id"], []).append(row)
        for package, (fmt, arm) in PACKAGES.items():
            selected = [
                row
                for row in rows
                if row["metadata"]["trace_format"] == fmt
                and row["metadata"]["supervision_arm"] == arm
            ]
            ids = [row["metadata"]["base_id"] for row in selected]
            if len(ids) != len(set(ids)) or set(ids) != set(groups):
                raise ValueError(f"{role}/{package} lacks one paired target per base")
            views[role, package] = "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in selected
            )
    train_ids = {r["metadata"]["base_id"] for r in rows_by_role["train"]}
    dev_ids = {r["metadata"]["base_id"] for r in rows_by_role["dev"]}
    if train_ids & dev_ids:
        raise ValueError("train/dev base overlap")
    template = yaml.safe_load(
        (root / "experiments/15-v2-ablation/templates/train.yaml").read_text()
    )
    template.update(
        base_model=budget["tokenizer_name"],
        revision_of_model=budget["tokenizer_revision"],
        sequence_len=budget["train_max_tokens"],
        num_epochs=2,
        load_best_model_at_end=False,
        val_set_size=0,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        use_wandb=True,
        wandb_project="spatialentail-sft",
        wandb_mode="online",
    )
    template.pop("metric_for_best_model", None)
    template.pop("greater_is_better", None)
    files = {}
    for (role, package), text in views.items():
        files[f"data/{role}-{package}.jsonl"] = text
    jobs = []
    for package in PACKAGES:
        for seed in (42, 43):
            name = f"{package}-seed-{seed}"
            model_relative = (output / "models" / name).relative_to(root)
            config = dict(
                template,
                seed=seed,
                wandb_name=f"qwen3.5-4b-{name}",
                wandb_run_id=f"spatialentail-qwen3.5-4b-{name}",
                output_dir="../" + str(model_relative),
                datasets=[
                    {
                        "path": "../"
                        + str(
                            (output / "data" / f"train-{package}.jsonl").relative_to(
                                root
                            )
                        ),
                        "type": "chat_template",
                        "field_messages": "messages",
                    }
                ],
                test_datasets=[
                    {
                        "path": "../"
                        + str(
                            (output / "data" / f"dev-{package}.jsonl").relative_to(root)
                        ),
                        "type": "chat_template",
                        "field_messages": "messages",
                    }
                ],
            )
            config_relative = (output / "configs" / f"{name}.yaml").relative_to(root)
            files[f"configs/{name}.yaml"] = yaml.safe_dump(config, sort_keys=False)
            # Paths below are repository-relative; transfer keeps server location arbitrary.
            files[f"slurm/{name}.sh"] = f"""#!/bin/bash
#SBATCH --job-name=spatialentail-{name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output={output.relative_to(root)}/logs/%x-%j.out
#SBATCH --error={output.relative_to(root)}/logs/%x-%j.err
set -euo pipefail
: "${{SLURM_SUBMIT_DIR:?Submit this script from the repository root}}"
cd "$SLURM_SUBMIT_DIR"
export ARM={name}
export TRAIN_CONFIG="$SLURM_SUBMIT_DIR/{config_relative}"
export MODEL_DIR="$SLURM_SUBMIT_DIR/{model_relative}"
export WANDB_RUN_GROUP="${{WANDB_RUN_GROUP:-spatialentail-qwen3.5-4b}}"
export WANDB_JOB_TYPE={package}
export WANDB_RESUME=allow
python3 - "$SLURM_SUBMIT_DIR/{output.relative_to(root)}/jobs.json" "$SLURM_SUBMIT_DIR/{output.relative_to(root)}" "$SLURM_SUBMIT_DIR" <<'CHECK'
import hashlib, json, pathlib, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
base = pathlib.Path(sys.argv[2])
for name, expected in receipt["file_sha256"].items():
    path = base / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Prepared job input changed: " + name)
for name, expected in receipt["environment_sha256"].items():
    path = pathlib.Path(sys.argv[3]) / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Training environment input changed: " + name)
CHECK
bash experiments/15-v2-ablation/jobs/train.sh
"""
            jobs.append(
                {
                    "name": name,
                    "package": package,
                    "seed": seed,
                    "config": str(config_relative),
                    "script": str((output / "slurm" / f"{name}.sh").relative_to(root)),
                }
            )
    receipt = {
        "schema": "spatialentail-sft-jobs-v1",
        "release_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "jobs": jobs,
        "submitted": False,
        "context_budget": budget,
        "environment_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "pyproject.toml",
                "uv.lock",
                "finetune/finetune.py",
                "experiments/15-v2-ablation/jobs/train.sh",
                "experiments/15-v2-ablation/jobs/common.bash",
                "slurm/lib/pin-srun-cpus.sh",
            )
            if (root / name).is_file()
        },
        "file_sha256": {
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in files.items()
        },
    }
    output.mkdir(parents=True)
    for filename, content in files.items():
        path = output / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
    (output / "logs").mkdir()
    (output / "logs/.gitkeep").write_text("")
    (output / "jobs.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    (output / "README.md").write_text("""# Prepared SFT jobs

Six jobs compare answer-only, Natural, and Symbolic on the same base problems,
with seeds 42 and 43. These files have not been submitted. Transfer the whole
repository plus the release/job artifacts to the Slurm server. From its repo
root, submit the six scripts individually using `sbatch <script-path>`.
The default resource request is one H200 on gpu for three hours. Confirm
that request against the server's limits before submission. Authenticate once
with `wandb login` or export `WANDB_API_KEY` before submission. Training and
system metrics are logged to the `spatialentail-sft` project; stable run IDs and
`WANDB_RESUME=allow` keep resumed allocations in the same run. Model checkpoints
are not uploaded to W&B. Training uses the existing finetune environment and
fixed final-epoch checkpoints. Bench data is absent from the Axolotl
configurations. Slurm sends BEGIN, END, FAIL, TIME_LIMIT, and TIME_LIMIT_90
notifications to the same NUS address used by Experiments 13/14. Logs are
written under `logs/`. The data/config hashes are in jobs.json.
""")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_jobs(args.release, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
