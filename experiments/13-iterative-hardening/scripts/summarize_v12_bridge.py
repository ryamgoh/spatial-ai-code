"""Summarize the original V12 2K test under the stage-1 protocol."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
V12_SCRIPTS = REPO / "experiments" / "12-v6-mix-reweight" / "scripts"
sys.path.insert(0, str(V12_SCRIPTS))
from kinds import kind, user_text  # noqa: E402

TASK = "spatial_eval_v12_synth"
TEST = REPO / "data" / "spatial_sft_v12_2000_test.jsonl"
DEFAULT_TAGS = ("v12test-baseline-stage1", "v12test-v12-4b-1.5k-stage1", "v12test-v12-4b-6k-stage1")
BUCKETS = (
    "dir-1", "dir-2", "dir-undetermined", "dir-cycle",
    "dir-incomplete", "dir-omit", "which-1", "which-2",
    "which-3", "which-4", "which-0", "count-1", "count-omit",
)


def letters(raw: object) -> tuple[str, ...]:
    return tuple(sorted({part for part in re.split(r"[,;|\s]+", str(raw or "").strip().upper()) if part in "ABCDE"}))


def prediction(sample: dict) -> tuple[str, ...]:
    filtered = sample.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        value = filtered[0]
        if isinstance(value, list):
            value = value[0] if value else ""
        return letters(value)
    raw = sample.get("resps")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    found = re.findall(r"Answer:\s*([A-E](?:\s*,\s*[A-E])*)", str(raw or ""), re.I)
    return letters(found[-1]) if found else ()


def gold(sample: dict) -> tuple[str, ...]:
    doc = sample.get("doc") or {}
    return letters(doc.get("oracle_option") or sample.get("target"))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_samples(results_dir: Path, tag: str) -> list[dict]:
    path = results_dir / tag / f"responses_{TASK}.jsonl"
    return load_rows(path) if path.is_file() else []


def summarize(results_dir: Path, tags: tuple[str, ...], test_path: Path = TEST) -> str:
    index = {user_text(row): kind(row) for row in load_rows(test_path)}
    lines = [
        "# V12 stage-1 bridge",
        "",
        "Original matched V12 2K test, strict answer-letter-set accuracy, one-pass protocol.",
        "",
        "| model | overall | invalid output |",
        "|---|---:|---:|",
    ]
    all_buckets: dict[str, dict[str, list[int]]] = {}
    for tag in tags:
        samples = load_samples(results_dir, tag)
        buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        correct = invalid = 0
        for sample in samples:
            doc = sample.get("doc") or {}
            bucket = index.get(str(doc.get("text") or ""), "unknown")
            pred, expected = prediction(sample), gold(sample)
            buckets[bucket][1] += 1
            if expected and pred == expected:
                correct += 1
                buckets[bucket][0] += 1
            if not pred:
                invalid += 1
        all_buckets[tag] = buckets
        total = len(samples)
        fmt = lambda value: "—" if not total else f"{100 * value / total:.1f}% ({value}/{total})"
        lines.append(f"| `{tag}` | {fmt(correct)} | {fmt(invalid)} |")
    lines += ["", "## V12 subtype breakdown", ""]
    lines.append("| subtype |" + "".join(f" {tag} |" for tag in tags))
    lines.append("|---|" + "---:|" * len(tags))
    for bucket in BUCKETS:
        values = []
        for tag in tags:
            ok, total = all_buckets[tag].get(bucket, (0, 0))
            values.append("—" if not total else f"{100 * ok / total:.1f}% ({ok}/{total})")
        lines.append(f"| `{bucket}` | " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    parser.add_argument("--test-path", type=Path, default=TEST)
    args = parser.parse_args()
    tags = tuple(tag.strip() for tag in args.tags.split(",") if tag.strip())
    report = summarize(args.results_dir, tags, args.test_path)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "V12-STAGE1-BRIDGE.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
