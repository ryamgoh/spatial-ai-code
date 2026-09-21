"""Summarize V14 GRPO feasibility against the delta-state SFT policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V13_SUMMARY = ROOT.parent / "13-iterative-hardening" / "scripts" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("v14_summary_helpers", V13_SUMMARY)
assert SPEC and SPEC.loader
V13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V13)

TASKS = {
    "holdout": "spatial_eval_v14_grpo_holdout",
    "v13": "spatial_eval_v13_diagnostic",
    "breakpoint": "spatial_eval_v13_breakpoint",
}
SCALE_RE = re.compile(r"(?:-d|-l)(10|12|14)(?:-|$)")


def load(results_dir: Path, tag: str, task: str) -> list[dict]:
    path = results_dir / tag / f"responses_{task}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell(row: dict) -> str:
    return str((row.get("doc") or {}).get("generation_cell") or "")


def family(row: dict) -> str:
    name = cell(row)
    if "-loop-" in name and name.endswith("-closed"):
        return "closed loop"
    if "-loop-" in name and name.endswith("-open-control"):
        return "open chain"
    if "-large-" in name:
        return "large which/count"
    if "-unequal-" in name:
        return "unequal interference"
    if "query-branch" in name:
        return "query-branch interference"
    return "disconnected interference"


def scale(row: dict) -> str | None:
    match = SCALE_RE.search(cell(row))
    return match.group(1) if match else None


def accuracy(rows: list[dict], predicate=lambda _row: True) -> tuple[int, int]:
    selected = [row for row in rows if predicate(row)]
    return sum(V13.strict_correct(row) for row in selected), len(selected)


def paired(
    before: list[dict], after: list[dict], predicate: Callable[[dict], bool]
) -> tuple[int, int, int]:
    left = {str((row.get("doc") or {}).get("text") or ""): row for row in before}
    right = {str((row.get("doc") or {}).get("text") or ""): row for row in after}
    if set(left) != set(right):
        raise ValueError("SFT and GRPO files do not contain identical prompts")
    shared = [key for key in left if predicate(left[key])]
    fixes = sum(
        not V13.strict_correct(left[key]) and V13.strict_correct(right[key])
        for key in shared
    )
    regressions = sum(
        V13.strict_correct(left[key]) and not V13.strict_correct(right[key])
        for key in shared
    )
    return len(shared), fixes, regressions


def table(
    before: list[dict], after: list[dict], predicates: tuple[tuple[str, Callable], ...]
) -> list[str]:
    lines = [
        "| slice | delta SFT | GRPO probe | fixed by GRPO | regressed at GRPO | net fixes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, predicate in predicates:
        _n, fixes, regressions = paired(before, after, predicate)
        lines.append(
            f"| {name} | {V13.pct(*accuracy(before, predicate))} | "
            f"{V13.pct(*accuracy(after, predicate))} | {fixes} | {regressions} | "
            f"{fixes - regressions:+d} |"
        )
    return lines


def summarize(results_dir: Path, calibration: Path) -> str:
    sft_holdout = load(results_dir, "sft-holdout-stage1", TASKS["holdout"])
    grpo_holdout = load(results_dir, "grpo-holdout-stage1", TASKS["holdout"])
    v13_results = ROOT.parent / "13-iterative-hardening" / "results"
    sft_v13 = load(v13_results, "trace-delta-v13-stage1", TASKS["v13"])
    grpo_v13 = load(results_dir, "grpo-v13-stage1", TASKS["v13"])
    sft_break = load(v13_results, "trace-delta-breakpoint-stage1", TASKS["breakpoint"])
    grpo_break = load(results_dir, "grpo-breakpoint-stage1", TASKS["breakpoint"])
    calibration_data = (
        json.loads(calibration.read_text()) if calibration.is_file() else {}
    )

    holdout_predicates = (
        ("overall", lambda _row: True),
        (
            "disconnected interference",
            lambda row: family(row) == "disconnected interference",
        ),
        (
            "query-branch interference",
            lambda row: family(row) == "query-branch interference",
        ),
        ("unequal interference", lambda row: family(row) == "unequal interference"),
        ("closed loop", lambda row: family(row) == "closed loop"),
        ("open chain", lambda row: family(row) == "open chain"),
        ("large which/count", lambda row: family(row) == "large which/count"),
        ("scale 14", lambda row: scale(row) == "14"),
    )
    lines = [
        "# V14 V13-GRPO feasibility probe",
        "",
        "## Pre-training rollout calibration",
        "",
        f"- Pass rate: {100 * calibration_data.get('pass_rate', 0):.1f}%",
        f"- Mixed-reward groups: {calibration_data.get('mixed_groups', 0)}/{calibration_data.get('num_prompts', 0)}",
        f"- Calibration gate: {'PASS' if calibration_data.get('go') else 'FAIL'}",
        "",
        "## Frozen hard holdout",
        "",
    ]
    if sft_holdout and grpo_holdout:
        lines += table(sft_holdout, grpo_holdout, holdout_predicates)
    else:
        lines.append("SFT or GRPO holdout responses are missing.")
    lines += ["", "## Original V13 retention", ""]
    if sft_v13 and grpo_v13:
        lines += table(sft_v13, grpo_v13, (("overall", lambda _row: True),))
    else:
        lines.append("Retention evaluation not requested or unavailable.")
    lines += ["", "## V13.1 retention", ""]
    if sft_break and grpo_break:
        lines += table(sft_break, grpo_break, (("overall", lambda _row: True),))
    else:
        lines.append("Retention evaluation not requested or unavailable.")
    lines += ["", "## Feasibility verdict", ""]
    if not calibration_data.get("go"):
        lines.append("- FAIL — pre-training reward variance gate did not pass.")
    elif not sft_holdout or not grpo_holdout:
        lines.append(
            "- INCOMPLETE — training ran but the frozen holdout comparison is missing."
        )
    else:
        before = accuracy(sft_holdout)[0] / len(sft_holdout)
        after = accuracy(grpo_holdout)[0] / len(grpo_holdout)
        lines.append(
            f"- {'PASS' if after > before else 'FAIL'} — GRPO improves the frozen hard holdout ({100 * before:.1f}% → {100 * after:.1f}%)."
        )
        if sft_v13 and grpo_v13:
            old = accuracy(sft_v13)[0] / len(sft_v13)
            new = accuracy(grpo_v13)[0] / len(grpo_v13)
            lines.append(
                f"- {'PASS' if new >= old - 0.02 else 'FAIL'} — original V13 drop <= 2 pp ({100 * old:.1f}% → {100 * new:.1f}%)."
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--calibration", type=Path, default=ROOT / "results" / "CALIBRATION.json"
    )
    args = parser.parse_args()
    report = summarize(args.results_dir, args.calibration)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "SUMMARY.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
