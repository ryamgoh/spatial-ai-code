"""Summarize the native V13 400-row SFT safety probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = Path(__file__).with_name("summarize.py")
SPEC = importlib.util.spec_from_file_location("v13_summary_for_probe", SUMMARY_PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
V12_TASK = "spatial_eval_v12_synth"


def accuracy(rows: list[dict], predicate=lambda _row: True) -> tuple[int, int]:
    selected = [row for row in rows if predicate(row)]
    return sum(SUMMARY.strict_correct(row) for row in selected), len(selected)


def fmt(value: tuple[int, int]) -> str:
    correct, total = value
    return "—" if not total else f"{100 * correct / total:.1f}% ({correct}/{total})"


def delta(after: tuple[int, int], before: tuple[int, int]) -> float | None:
    if not after[1] or not before[1]:
        return None
    return (after[0] / after[1] - before[0] / before[1]) * 100


def fmt_delta(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:+.1f} pp"


def load_task(results_dir: Path, tag: str, task: str) -> list[dict]:
    path = results_dir / tag / f"responses_{task}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(results_dir: Path) -> str:
    cells = {
        "base-stage1": SUMMARY.load_samples(results_dir, "baseline-stage1"),
        "probe-stage1": SUMMARY.load_samples(results_dir, "probe-v13-stage1"),
        "base-stage2": SUMMARY.load_samples(results_dir, "baseline"),
        "probe-stage2": SUMMARY.load_samples(results_dir, "probe-v13-stage2"),
    }
    lines = [
        "# Native V13 400-row probe",
        "",
        "| model/protocol | V13 overall | dir-2 | consistent which | consistent count | depth 5 | closed loop | open chain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    prompt_sets = {
        tag: {str((row.get("doc") or {}).get("text") or "") for row in rows}
        for tag, rows in cells.items()
        if rows
    }
    if prompt_sets and not all(
        prompts == next(iter(prompt_sets.values())) for prompts in prompt_sets.values()
    ):
        raise ValueError("V13 baseline/probe result files do not contain identical prompts")
    predicates = (
        lambda _row: True,
        lambda row: SUMMARY._semantic_subtype(row) == "dir-2" and SUMMARY._consistency(row) == "consistent",
        lambda row: SUMMARY._question_family(row) == "which" and SUMMARY._consistency(row) == "consistent",
        lambda row: SUMMARY._question_family(row) == "count" and SUMMARY._consistency(row) == "consistent",
        lambda row: SUMMARY._depth_group(row) == "5",
        lambda row: SUMMARY._cycle_group(row) == "closed-loop condition",
        lambda row: SUMMARY._cycle_group(row) == "open-chain control",
    )
    for tag, rows in cells.items():
        lines.append(
            f"| `{tag}` | "
            + " | ".join(fmt(accuracy(rows, predicate)) for predicate in predicates)
            + " |"
        )

    v12_rows = load_task(results_dir, "probe-v12-stage1", V12_TASK)
    if v12_rows and len(v12_rows) != 2000:
        raise ValueError(f"expected 2000 V12 retention samples, found {len(v12_rows)}")
    lines += [
        "",
        "## V12 retention",
        "",
        f"Probe adapter on the original V12 2K test, stage 1: **{fmt(accuracy(v12_rows))}**.",
        "",
        "## Safety gates",
        "",
    ]
    base2, probe2 = cells["base-stage2"], cells["probe-stage2"]
    if not base2 or not probe2:
        lines.append("Baseline or probe stage-2 samples are missing; no go/no-go verdict.")
    else:
        base_dir2 = accuracy(base2, predicates[1])
        probe_dir2 = accuracy(probe2, predicates[1])
        base_depth = accuracy(base2, predicates[4])
        probe_depth = accuracy(probe2, predicates[4])
        base_weak = accuracy(base2, lambda row: predicates[2](row) or predicates[3](row))
        probe_weak = accuracy(probe2, lambda row: predicates[2](row) or predicates[3](row))
        dir2_delta = delta(probe_dir2, base_dir2)
        depth_delta = delta(probe_depth, base_depth)
        weak_delta = delta(probe_weak, base_weak)
        retention = accuracy(v12_rows)
        checks = {
            "dir-2 improves": dir2_delta is not None and dir2_delta > 0,
            "weak which/count improves": weak_delta is not None and weak_delta > 0,
            "depth-5 drop <= 5 pp": depth_delta is not None and depth_delta >= -5,
            "V12 stage-1 retention >= 90%": bool(retention[1]) and retention[0] / retention[1] >= 0.90,
            "closed-loop and open-chain conditions each >= 50%": all(
                accuracy(probe2, predicate)[1]
                and accuracy(probe2, predicate)[0] / accuracy(probe2, predicate)[1] >= 0.50
                for predicate in predicates[5:7]
            ),
        }
        lines.append(
            f"Dir-2 delta: **{fmt_delta(dir2_delta)}**; weak-family delta: "
            f"**{fmt_delta(weak_delta)}**; depth-5 delta: **{fmt_delta(depth_delta)}**."
        )
        lines.append("")
        for name, passed in checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
        lines += ["", "**GO** only if all five gates pass; otherwise revise the probe recipe before scaling."]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report = summarize(args.results_dir)
    (args.results_dir / "PROBE-SUMMARY.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
