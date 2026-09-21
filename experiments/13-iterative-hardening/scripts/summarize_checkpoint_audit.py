"""Compare Axolotl's exported best 8K adapter with the final checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = Path(__file__).with_name("summarize_sft_8000.py")
SPEC = importlib.util.spec_from_file_location("v13_checkpoint_audit_helpers", MODULE)
assert SPEC and SPEC.loader
SFT8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SFT8)
SUMMARY = SFT8.SUMMARY
BREAK = SFT8.BREAK
V1 = SFT8.V1


def summarize(results_dir: Path) -> tuple[str, list[dict]]:
    suites = (
        (
            "original V13",
            SUMMARY.load_samples(results_dir, "sft-8000-v13-stage1"),
            SUMMARY.load_samples(results_dir, "sft-8000-final-v13-stage1"),
            (("overall", lambda _row: True),),
        ),
        (
            "V13.1 breakpoint",
            BREAK.load(results_dir, "breakpoint-sft-8k-stage1"),
            BREAK.load(results_dir, "breakpoint-sft-8k-final-stage1"),
            (
                ("overall", lambda _row: True),
                (
                    "long disconnected",
                    lambda row: BREAK.challenge(row) == "long-disconnected",
                ),
                (
                    "long query branch",
                    lambda row: BREAK.challenge(row) == "long-query-branch",
                ),
                (
                    "unequal query branch",
                    lambda row: BREAK.challenge(row) == "unequal-query-branch",
                ),
                ("depth/length 10", lambda row: BREAK.scale(row) == "10"),
                (
                    "long closed loop",
                    lambda row: BREAK.challenge(row) == "long closed loop",
                ),
                (
                    "matched open chain",
                    lambda row: BREAK.challenge(row) == "matched open chain",
                ),
                ("which / open", lambda row: BREAK.loop_family(row) == "which / open"),
            ),
        ),
    )
    lines = [
        "# V13.1 8K checkpoint-selection audit",
        "",
        "`exported-best` is the adapter selected by old-V13 validation loss. `final-checkpoint` is the numerically latest retained training checkpoint.",
        "",
    ]
    examples = []
    for title, best, final, predicates in suites:
        lines += [f"## {title}", ""]
        if not best or not final:
            lines += ["Exported-best or final-checkpoint responses are missing.", ""]
            continue
        lines += [
            "| slice | exported-best | final-checkpoint | fixed by final | regressed at final | net fixes |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, predicate in predicates:
            before = SFT8.accuracy(best, predicate)
            after = SFT8.accuracy(final, predicate)
            _total, _both, fixes, regressions, _wrong = SFT8.paired(
                best, final, predicate
            )
            lines.append(
                f"| {name} | {V1.fmt(before)} | {V1.fmt(after)} | "
                f"{fixes} | {regressions} | {fixes - regressions:+d} |"
            )
        examples += SFT8.paired_examples(title, best, final)
        lines.append("")
    lines += [
        "If the final checkpoint materially improves V13.1 without reducing original V13 below 93%, repair validation selection before changing the curriculum. Otherwise, the remaining limit is curricular/structural rather than checkpoint selection.",
        "",
    ]
    return "\n".join(lines), examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, examples = summarize(args.results_dir)
    (args.results_dir / "SFT-8000-CHECKPOINT-AUDIT.md").write_text(report + "\n")
    (args.results_dir / "SFT-8000-CHECKPOINT-CHANGES.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples)
    )
    print(report)


if __name__ == "__main__":
    main()
