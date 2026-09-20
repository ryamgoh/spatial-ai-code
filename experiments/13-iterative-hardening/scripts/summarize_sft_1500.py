"""Compare the native V13 1.5K model with Probe v2."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = Path(__file__).with_name("summarize_probe_v2.py")
SPEC = importlib.util.spec_from_file_location("v13_1500_summary_helpers", HELPERS_PATH)
assert SPEC and SPEC.loader
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
SUMMARY = HELPERS.SUMMARY


def load_v12(results_dir: Path, tag: str) -> list[dict]:
    path = results_dir / tag / f"responses_{HELPERS.V1.V12_TASK}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def rate(value: tuple[int, int]) -> float | None:
    return value[0] / value[1] if value[1] else None


def summarize(results_dir: Path) -> str:
    cells = {
        "base-stage1": SUMMARY.load_samples(results_dir, "baseline-stage1"),
        "probe-v2-stage1": SUMMARY.load_samples(results_dir, "probe-v2-v13-stage1"),
        "sft-1.5k-stage1": SUMMARY.load_samples(results_dir, "sft-1500-v13-stage1"),
        "base-stage2": SUMMARY.load_samples(results_dir, "baseline"),
        "probe-v2-stage2": SUMMARY.load_samples(results_dir, "probe-v2-v13-stage2"),
        "sft-1.5k-stage2": SUMMARY.load_samples(results_dir, "sft-1500-v13-stage2"),
    }
    prompt_sets = [
        {str((row.get("doc") or {}).get("text") or "") for row in rows}
        for rows in cells.values() if rows
    ]
    if prompt_sets and not all(prompts == prompt_sets[0] for prompts in prompt_sets[1:]):
        raise ValueError("1.5K comparison files do not contain identical prompts")
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
        "# Native V13 SFT 1.5K",
        "",
        "| model/protocol | " + " | ".join(name for name, _ in predicates) + " |",
        "|---|" + "---:|" * len(predicates),
    ]
    for tag, rows in cells.items():
        lines.append(
            f"| `{tag}` | "
            + " | ".join(HELPERS.V1.fmt(HELPERS.V1.accuracy(rows, pred)) for _name, pred in predicates)
            + " |"
        )

    v12 = load_v12(results_dir, "sft-1500-v12-stage1")
    if v12 and len(v12) != 2000:
        raise ValueError(f"expected 2000 V12 samples, found {len(v12)}")
    lines += [
        "",
        "## Informational V12 compatibility",
        "",
        f"Native V13 1.5K, stage 1: {HELPERS.V1.fmt(HELPERS.V1.accuracy(v12))}.",
        "",
        "## 1.5K scaling gates",
        "",
    ]
    probe, scaled = cells["probe-v2-stage1"], cells["sft-1.5k-stage1"]
    if not probe or not scaled:
        lines.append("Probe v2 or 1.5K stage-1 samples are missing; no verdict.")
    else:
        values = {name: (HELPERS.V1.accuracy(probe, pred), HELPERS.V1.accuracy(scaled, pred)) for name, pred in predicates}
        checks = {
            "V13 overall >= Probe v2": (rate(values["overall"][1]) or 0) >= (rate(values["overall"][0]) or 0),
            "dir-2 >= 55%": (rate(values["dir-2"][1]) or 0) >= 0.55,
            "consistent which >= 50%": (rate(values["which"][1]) or 0) >= 0.50,
            "consistent count >= 60%": (rate(values["count"][1]) or 0) >= 0.60,
            "depth-5 stage 1 >= 67%": (rate(values["depth-5"][1]) or 0) >= 0.67,
            "closed-loop and open-chain conditions each >= 60%": all(
                (rate(values[name][1]) or 0) >= 0.60 for name in ("closed loop", "open chain")
            ),
        }
        for name, passed in checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
        lines += ["", "**Train 6K only if all six gates pass.**"]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report = summarize(args.results_dir)
    (args.results_dir / "SFT-1500-SUMMARY.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
