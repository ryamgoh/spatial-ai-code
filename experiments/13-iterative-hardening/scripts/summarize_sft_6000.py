"""Compare the nested native V13 6K model with the 1.5K model."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = Path(__file__).with_name("summarize_sft_1500.py")
SPEC = importlib.util.spec_from_file_location("v13_6000_summary_helpers", HELPERS_PATH)
assert SPEC and SPEC.loader
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
SUMMARY = HELPERS.SUMMARY
V1 = HELPERS.HELPERS.V1


def rate(value: tuple[int, int]) -> float | None:
    return value[0] / value[1] if value[1] else None


def indexed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        prompt = str((row.get("doc") or {}).get("text") or "")
        if not prompt or prompt in result:
            raise ValueError("result file has missing or duplicate prompts")
        result[prompt] = row
    return result


def paired(
    before: list[dict], after: list[dict], predicate: Callable[[dict], bool]
) -> tuple[int, int, int, int, int]:
    left, right = indexed(before), indexed(after)
    if set(left) != set(right):
        raise ValueError("1.5K and 6K result files do not contain identical prompts")
    both_correct = fixes = regressions = both_wrong = total = 0
    for prompt in sorted(left):
        if not predicate(left[prompt]):
            continue
        total += 1
        a = SUMMARY.strict_correct(left[prompt])
        b = SUMMARY.strict_correct(right[prompt])
        if a and b:
            both_correct += 1
        elif not a and b:
            fixes += 1
        elif a and not b:
            regressions += 1
        else:
            both_wrong += 1
    return total, both_correct, fixes, regressions, both_wrong


def paired_examples(before: list[dict], after: list[dict]) -> list[dict]:
    left, right = indexed(before), indexed(after)
    if set(left) != set(right):
        raise ValueError("1.5K and 6K result files do not contain identical prompts")
    examples = []
    for prompt in sorted(left):
        before_correct = SUMMARY.strict_correct(left[prompt])
        after_correct = SUMMARY.strict_correct(right[prompt])
        if before_correct == after_correct:
            continue
        doc = left[prompt].get("doc") or {}
        examples.append(
            {
                "transition": "fixed-by-6k" if after_correct else "regressed-at-6k",
                "prompt": prompt,
                "generation_cell": doc.get("generation_cell"),
                "difficulty": doc.get("difficulty"),
                "gold": list(SUMMARY.gold(left[prompt])),
                "prediction_1500": list(SUMMARY.prediction(left[prompt])),
                "prediction_6000": list(SUMMARY.prediction(right[prompt])),
                "response_1500": left[prompt].get("resps"),
                "response_6000": right[prompt].get("resps"),
            }
        )
    return examples


def summarize(results_dir: Path) -> tuple[str, list[dict]]:
    cells = {
        "base-stage1": SUMMARY.load_samples(results_dir, "baseline-stage1"),
        "sft-1.5k-stage1": SUMMARY.load_samples(results_dir, "sft-1500-v13-stage1"),
        "sft-6k-stage1": SUMMARY.load_samples(results_dir, "sft-6000-v13-stage1"),
        "sft-1.5k-stage2": SUMMARY.load_samples(results_dir, "sft-1500-v13-stage2"),
        "sft-6k-stage2": SUMMARY.load_samples(results_dir, "sft-6000-v13-stage2"),
    }
    nonempty = [indexed(rows) for rows in cells.values() if rows]
    if nonempty and not all(set(rows) == set(nonempty[0]) for rows in nonempty[1:]):
        raise ValueError("6K comparison files do not contain identical prompts")

    predicates = (
        ("overall", lambda _row: True),
        (
            "dir-2",
            lambda row: (
                SUMMARY._semantic_subtype(row) == "dir-2"
                and SUMMARY._consistency(row) == "consistent"
            ),
        ),
        (
            "which",
            lambda row: (
                SUMMARY._question_family(row) == "which"
                and SUMMARY._consistency(row) == "consistent"
            ),
        ),
        (
            "count",
            lambda row: (
                SUMMARY._question_family(row) == "count"
                and SUMMARY._consistency(row) == "consistent"
            ),
        ),
        ("depth-5", lambda row: SUMMARY._depth_group(row) == "5"),
        (
            "closed loop",
            lambda row: SUMMARY._cycle_group(row) == "closed-loop condition",
        ),
        ("open chain", lambda row: SUMMARY._cycle_group(row) == "open-chain control"),
    )
    lines = [
        "# Native V13 nested SFT 6K",
        "",
        "The 6K train set contains the exact frozen 1.5K rows plus 4,500 new rows; both models start independently from untouched Qwen3.5-4B.",
        "",
        "| model/protocol | " + " | ".join(name for name, _ in predicates) + " |",
        "|---|" + "---:|" * len(predicates),
    ]
    for tag, rows in cells.items():
        lines.append(
            f"| `{tag}` | "
            + " | ".join(V1.fmt(V1.accuracy(rows, pred)) for _name, pred in predicates)
            + " |"
        )

    before, after = cells["sft-1.5k-stage1"], cells["sft-6k-stage1"]
    examples: list[dict] = []
    lines += ["", "## Paired stage-1 changes", ""]
    if not before or not after:
        lines.append(
            "1.5K or 6K stage-1 samples are missing; paired analysis unavailable."
        )
    else:
        lines += [
            "| slice | rows | both correct | fixed by 6K | regressed at 6K | both wrong | net fixes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, pred in predicates:
            total, both_correct, fixes, regressions, both_wrong = paired(
                before, after, pred
            )
            lines.append(
                f"| {name} | {total} | {both_correct} | {fixes} | {regressions} | {both_wrong} | {fixes - regressions:+d} |"
            )
        examples = paired_examples(before, after)

    lines += ["", "## 6K gates", ""]
    if not before or not after:
        lines.append("1.5K or 6K stage-1 samples are missing; no verdict.")
    else:
        values = {name: V1.accuracy(after, pred) for name, pred in predicates}
        before_overall = rate(V1.accuracy(before)) or 0
        checks = {
            "overall does not fall more than 2 pp versus 1.5K": (
                rate(values["overall"]) or 0
            )
            >= before_overall - 0.02,
            "dir-2 >= 55%": (rate(values["dir-2"]) or 0) >= 0.55,
            "consistent which >= 50%": (rate(values["which"]) or 0) >= 0.50,
            "consistent count >= 60%": (rate(values["count"]) or 0) >= 0.60,
            "depth-5 stage 1 >= 67%": (rate(values["depth-5"]) or 0) >= 0.67,
            "closed-loop and open-chain conditions each >= 60%": all(
                (rate(values[name]) or 0) >= 0.60
                for name in ("closed loop", "open chain")
            ),
        }
        for name, passed in checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
        lines += [
            "",
            "Inspect paired regressions before deciding whether 6K replaces 1.5K.",
        ]

    v12 = HELPERS.load_v12(results_dir, "sft-6000-v12-stage1")
    lines += [
        "",
        "## Informational V12 compatibility",
        "",
        f"Native V13 6K, stage 1: {V1.fmt(V1.accuracy(v12))}.",
        "",
    ]
    return "\n".join(lines), examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, examples = summarize(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "SFT-6000-SUMMARY.md").write_text(report + "\n")
    examples_path = args.results_dir / "SFT-6000-PAIRED-CHANGES.jsonl"
    examples_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples)
    )
    print(report)


if __name__ == "__main__":
    main()
