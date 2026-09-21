"""Report native V13.1 8K gains and original-V13 retention versus 6K."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SFT6 = load_module(
    "v13_sft_8000_summary_sft6", Path(__file__).with_name("summarize_sft_6000.py")
)
BREAK = load_module(
    "v13_sft_8000_summary_break", Path(__file__).with_name("summarize_breakpoint.py")
)
SUMMARY = SFT6.SUMMARY
V1 = SFT6.V1


def accuracy(rows: list[dict], predicate=lambda _row: True) -> tuple[int, int]:
    selected = [row for row in rows if predicate(row)]
    return sum(SUMMARY.strict_correct(row) for row in selected), len(selected)


def rate(value: tuple[int, int]) -> float | None:
    return value[0] / value[1] if value[1] else None


def paired(
    before: list[dict], after: list[dict], predicate: Callable[[dict], bool]
) -> tuple[int, int, int, int, int]:
    return SFT6.paired(before, after, predicate)


def paired_examples(suite: str, before: list[dict], after: list[dict]) -> list[dict]:
    return [{"suite": suite, **row} for row in SFT6.paired_examples(before, after)]


def paired_table(
    before: list[dict],
    after: list[dict],
    predicates: tuple[tuple[str, Callable[[dict], bool]], ...],
) -> list[str]:
    lines = [
        "| slice | rows | both correct | fixed by 8K | regressed at 8K | both wrong | net fixes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, predicate in predicates:
        total, both_correct, fixes, regressions, both_wrong = paired(
            before, after, predicate
        )
        lines.append(
            f"| {name} | {total} | {both_correct} | {fixes} | "
            f"{regressions} | {both_wrong} | {fixes - regressions:+d} |"
        )
    return lines


def challenge_scale(row: dict) -> str | None:
    family = BREAK.challenge(row)
    scale = BREAK.scale(row)
    return f"{family} / {scale}" if family and scale else None


def summarize(results_dir: Path) -> tuple[str, list[dict]]:
    v13_6 = SUMMARY.load_samples(results_dir, "sft-6000-v13-stage1")
    v13_8 = SUMMARY.load_samples(results_dir, "sft-8000-v13-stage1")
    break_6 = BREAK.load(results_dir, "breakpoint-sft-6k-stage1")
    break_8 = BREAK.load(results_dir, "breakpoint-sft-8k-stage1")

    v13_predicates = (
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
    break_predicates = (
        ("overall", lambda _row: True),
        ("long disconnected", lambda row: BREAK.challenge(row) == "long-disconnected"),
        ("long query branch", lambda row: BREAK.challenge(row) == "long-query-branch"),
        (
            "unequal query branch",
            lambda row: BREAK.challenge(row) == "unequal-query-branch",
        ),
        ("length/depth 10", lambda row: BREAK.scale(row) == "10"),
        ("long closed loop", lambda row: BREAK.challenge(row) == "long closed loop"),
        (
            "matched open chain",
            lambda row: BREAK.challenge(row) == "matched open chain",
        ),
        ("which / open", lambda row: BREAK.loop_family(row) == "which / open"),
    )
    cross_predicates = tuple(
        (
            f"{family} / {scale}",
            lambda row, expected=f"{family} / {scale}": (
                challenge_scale(row) == expected
            ),
        )
        for family in (
            "long-disconnected",
            "long-query-branch",
            "unequal-query-branch",
            "long closed loop",
            "matched open chain",
        )
        for scale in ("6", "8", "10")
    )

    lines = [
        "# Native V13.1 nested SFT 8K",
        "",
        "The 8K train set contains the exact V13 6K rows plus 2,000 failure-targeted rows. Both models start independently from untouched Qwen3.5-4B.",
        "",
        "## Original V13 retention",
        "",
        "| model | " + " | ".join(name for name, _ in v13_predicates) + " |",
        "|---|" + "---:|" * len(v13_predicates),
    ]
    for name, rows in (("sft-6k", v13_6), ("sft-8k", v13_8)):
        lines.append(
            f"| `{name}` | "
            + " | ".join(V1.fmt(accuracy(rows, pred)) for _, pred in v13_predicates)
            + " |"
        )
    lines += ["", "### Paired original-V13 changes", ""]
    if v13_6 and v13_8:
        lines += paired_table(v13_6, v13_8, v13_predicates)
    else:
        lines.append("6K or 8K original-V13 results are missing.")

    lines += [
        "",
        "## V13.1 breakpoint generalization",
        "",
        "| model | " + " | ".join(name for name, _ in break_predicates) + " |",
        "|---|" + "---:|" * len(break_predicates),
    ]
    for name, rows in (("sft-6k", break_6), ("sft-8k", break_8)):
        lines.append(
            f"| `{name}` | "
            + " | ".join(V1.fmt(accuracy(rows, pred)) for _, pred in break_predicates)
            + " |"
        )
    lines += ["", "### Paired breakpoint changes", ""]
    if break_6 and break_8:
        lines += paired_table(break_6, break_8, break_predicates)
    else:
        lines.append("6K or 8K breakpoint results are missing.")

    lines += ["", "### Challenge by structural scale", ""]
    lines += [
        "| slice | sft-6k | sft-8k | delta |",
        "|---|---:|---:|---:|",
    ]
    for name, predicate in cross_predicates:
        before = accuracy(break_6, predicate)
        after = accuracy(break_8, predicate)
        before_rate = rate(before)
        after_rate = rate(after)
        delta = (
            "—"
            if before_rate is None or after_rate is None
            else f"{100 * (after_rate - before_rate):+.1f} pp"
        )
        lines.append(f"| {name} | {V1.fmt(before)} | {V1.fmt(after)} | {delta} |")

    lines += ["", "## 8K gates", ""]
    if not all((v13_6, v13_8, break_6, break_8)):
        lines.append("Required 6K or 8K result files are missing; no verdict.")
    else:
        original_6 = rate(accuracy(v13_6)) or 0
        values = {name: accuracy(break_8, pred) for name, pred in break_predicates}
        checks = {
            "original V13 overall >= 93%": (rate(accuracy(v13_8)) or 0) >= 0.93,
            "original V13 drop versus 6K <= 2 pp": (rate(accuracy(v13_8)) or 0)
            >= original_6 - 0.02,
            "breakpoint overall improves over 6K": (rate(accuracy(break_8)) or 0)
            > (rate(accuracy(break_6)) or 0),
            "each interference family >= 75%": all(
                (rate(values[name]) or 0) >= 0.75
                for name in (
                    "long disconnected",
                    "long query branch",
                    "unequal query branch",
                )
            ),
            "held-out depth/length 10 >= 70%": (rate(values["length/depth 10"]) or 0)
            >= 0.70,
            "which/open >= 80%": (rate(values["which / open"]) or 0) >= 0.80,
            "closed and open each >= 80%": all(
                (rate(values[name]) or 0) >= 0.80
                for name in ("long closed loop", "matched open chain")
            ),
        }
        for name, passed in checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")

    examples = []
    if v13_6 and v13_8:
        examples += paired_examples("original-v13", v13_6, v13_8)
    if break_6 and break_8:
        examples += paired_examples("v13.1-breakpoint", break_6, break_8)
    lines.append("")
    return "\n".join(lines), examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, examples = summarize(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "SFT-8000-SUMMARY.md").write_text(report + "\n")
    (args.results_dir / "SFT-8000-PAIRED-CHANGES.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples)
    )
    print(report)


if __name__ == "__main__":
    main()
