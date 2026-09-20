"""Compare Probe v2 with the untuned base and Probe v1."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_PATH = Path(__file__).with_name("summarize_probe.py")
SPEC = importlib.util.spec_from_file_location("probe_v1_summary_helpers", V1_PATH)
assert SPEC and SPEC.loader
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)
SUMMARY = V1.SUMMARY


def load_v12(results_dir: Path, tag: str) -> list[dict]:
    path = results_dir / tag / f"responses_{V1.V12_TASK}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(rows: list[dict], predicate) -> tuple[int, int]:
    return V1.accuracy(rows, predicate)


def rate(value: tuple[int, int]) -> float | None:
    return value[0] / value[1] if value[1] else None


def summarize(results_dir: Path) -> str:
    cells = {
        "base-stage1": SUMMARY.load_samples(results_dir, "baseline-stage1"),
        "probe-v1-stage1": SUMMARY.load_samples(results_dir, "probe-v13-stage1"),
        "probe-v2-stage1": SUMMARY.load_samples(results_dir, "probe-v2-v13-stage1"),
        "base-stage2": SUMMARY.load_samples(results_dir, "baseline"),
        "probe-v1-stage2": SUMMARY.load_samples(results_dir, "probe-v13-stage2"),
        "probe-v2-stage2": SUMMARY.load_samples(results_dir, "probe-v2-v13-stage2"),
    }
    prompt_sets = [
        {str((row.get("doc") or {}).get("text") or "") for row in rows}
        for rows in cells.values()
        if rows
    ]
    if prompt_sets and not all(prompts == prompt_sets[0] for prompts in prompt_sets[1:]):
        raise ValueError("Probe comparison result files do not contain identical prompts")

    predicates = (
        ("overall", lambda _row: True),
        ("dir-2", lambda row: SUMMARY._semantic_subtype(row) == "dir-2" and SUMMARY._consistency(row) == "consistent"),
        ("which", lambda row: SUMMARY._question_family(row) == "which" and SUMMARY._consistency(row) == "consistent"),
        ("count", lambda row: SUMMARY._question_family(row) == "count" and SUMMARY._consistency(row) == "consistent"),
        ("depth-5", lambda row: SUMMARY._depth_group(row) == "5"),
        ("closed loop", lambda row: SUMMARY._cycle_group(row) == "closed-loop condition"),
        ("open chain", lambda row: SUMMARY._cycle_group(row) == "open-chain control"),
    )
    lines = [
        "# Native V13 Probe v2",
        "",
        "| model/protocol | " + " | ".join(name for name, _ in predicates) + " |",
        "|---|" + "---:|" * len(predicates),
    ]
    for tag, rows in cells.items():
        lines.append(
            f"| `{tag}` | "
            + " | ".join(V1.fmt(pct(rows, predicate)) for _name, predicate in predicates)
            + " |"
        )

    v1_v12 = load_v12(results_dir, "probe-v12-stage1")
    v2_v12 = load_v12(results_dir, "probe-v2-v12-stage1")
    if v2_v12 and len(v2_v12) != 2000:
        raise ValueError(f"expected 2000 Probe v2 V12 samples, found {len(v2_v12)}")
    lines += [
        "",
        "## Informational V12 compatibility",
        "",
        f"- Probe v1 stage 1: {V1.fmt(V1.accuracy(v1_v12))}",
        f"- Probe v2 stage 1: {V1.fmt(V1.accuracy(v2_v12))}",
        "",
        "V12 compatibility is reported but is not a Probe v2 gate because the native V13 model starts from the untouched base and targets different semantics.",
        "",
        "## Probe v2 gates",
        "",
    ]
    v1_rows, v2_rows = cells["probe-v1-stage1"], cells["probe-v2-stage1"]
    if not v1_rows or not v2_rows:
        lines.append("Probe v1 or Probe v2 stage-1 samples are missing; no verdict.")
    else:
        values = {name: (pct(v1_rows, pred), pct(v2_rows, pred)) for name, pred in predicates}
        checks = {
            "V13 stage-1 overall >= Probe v1 - 3 pp": (rate(values["overall"][1]) or 0) >= (rate(values["overall"][0]) or 0) - 0.03,
            "dir-2 >= 50%": (rate(values["dir-2"][1]) or 0) >= 0.50,
            "consistent which >= 40%": (rate(values["which"][1]) or 0) >= 0.40,
            "consistent count >= 55%": (rate(values["count"][1]) or 0) >= 0.55,
            "depth-5 stage 1 >= 67%": (rate(values["depth-5"][1]) or 0) >= 0.67,
            "closed-loop and open-chain conditions each >= 50%": all(
                (rate(values[name][1]) or 0) >= 0.50
                for name in ("closed loop", "open chain")
            ),
        }
        for name, passed in checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
        lines += ["", "**GO** only if all six V13 gates pass."]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report = summarize(args.results_dir)
    (args.results_dir / "PROBE-V2-SUMMARY.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
