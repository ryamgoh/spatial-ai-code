"""Compare matched full-state and delta-state V13.1 8K training runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = Path(__file__).with_name("summarize_sft_8000.py")
SPEC = importlib.util.spec_from_file_location("trace_ablation_helpers", MODULE)
assert SPEC and SPEC.loader
SFT8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SFT8)
SUMMARY = SFT8.SUMMARY
BREAK = SFT8.BREAK
V1 = SFT8.V1


def report_table(
    title: str,
    full: list[dict],
    delta: list[dict],
    predicates: tuple,
) -> list[str]:
    lines = [
        f"## {title}", "",
        "| slice | full-state 8K | delta-state 8K | fixed by delta | regressed at delta | net fixes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not full or not delta:
        return lines + ["| unavailable | — | — | — | — | — |", ""]
    for name, predicate in predicates:
        before = SFT8.accuracy(full, predicate)
        after = SFT8.accuracy(delta, predicate)
        _total, _both, fixes, regressions, _wrong = SFT8.paired(
            full, delta, predicate
        )
        lines.append(
            f"| {name} | {V1.fmt(before)} | {V1.fmt(after)} | "
            f"{fixes} | {regressions} | {fixes - regressions:+d} |"
        )
    lines.append("")
    return lines


def summarize(results_dir: Path, manifest_path: Path) -> tuple[str, list[dict]]:
    full_v13 = SUMMARY.load_samples(results_dir, "sft-8000-v13-stage1")
    delta_v13 = SUMMARY.load_samples(results_dir, "trace-delta-v13-stage1")
    full_break = BREAK.load(results_dir, "breakpoint-sft-8k-stage1")
    delta_break = BREAK.load(results_dir, "trace-delta-breakpoint-stage1")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    v13_predicates = (
        ("overall", lambda _row: True),
        ("dir-2", lambda row: SUMMARY._semantic_subtype(row) == "dir-2" and SUMMARY._consistency(row) == "consistent"),
        ("which", lambda row: SUMMARY._question_family(row) == "which" and SUMMARY._consistency(row) == "consistent"),
        ("count", lambda row: SUMMARY._question_family(row) == "count" and SUMMARY._consistency(row) == "consistent"),
        ("closed loop", lambda row: SUMMARY._cycle_group(row) == "closed-loop condition"),
        ("open chain", lambda row: SUMMARY._cycle_group(row) == "open-chain control"),
    )
    break_predicates = (
        ("overall", lambda _row: True),
        ("long clean", lambda row: BREAK.challenge(row) == "long-clean"),
        ("long disconnected", lambda row: BREAK.challenge(row) == "long-disconnected"),
        ("long query branch", lambda row: BREAK.challenge(row) == "long-query-branch"),
        ("unequal query branch", lambda row: BREAK.challenge(row) == "unequal-query-branch"),
        ("depth/length 10", lambda row: BREAK.scale(row) == "10"),
        ("matched open chain", lambda row: BREAK.challenge(row) == "matched open chain"),
        ("which / open", lambda row: BREAK.loop_family(row) == "which / open"),
    )
    cross = tuple(
        (
            f"{family} / {scale}",
            lambda row, expected=f"{family} / {scale}": SFT8.challenge_scale(row) == expected,
        )
        for family in ("long-disconnected", "long-query-branch", "unequal-query-branch")
        for scale in ("6", "8", "10")
    )
    ratio = manifest.get("delta_to_full_char_ratio")
    ratio_text = "—" if ratio is None else f"{100 * ratio:.1f}%"
    lines = [
        "# V13.1 matched trace-format ablation", "",
        "Both arms use identical 8,000 prompts, gold answers, row order, validation worlds, and optimization. Only the assistant reasoning target differs.",
        "",
        f"Delta-state assistant characters are {ratio_text} of full-state characters.",
        "",
    ]
    lines += report_table("Original V13 retention", full_v13, delta_v13, v13_predicates)
    lines += report_table("V13.1 breakpoint", full_break, delta_break, break_predicates)
    lines += report_table("Interference by scale", full_break, delta_break, cross)
    lines += ["## Decision rule", ""]
    if full_v13 and delta_v13 and full_break and delta_break:
        full_old = SFT8.rate(SFT8.accuracy(full_v13)) or 0
        delta_old = SFT8.rate(SFT8.accuracy(delta_v13)) or 0
        full_new = SFT8.rate(SFT8.accuracy(full_break)) or 0
        delta_new = SFT8.rate(SFT8.accuracy(delta_break)) or 0
        checks = {
            "original V13 drop <= 2 pp": delta_old >= full_old - 0.02,
            "V13.1 overall improves": delta_new > full_new,
            "query-branch depth 10 improves": (
                SFT8.rate(
                    SFT8.accuracy(
                        delta_break,
                        lambda row: SFT8.challenge_scale(row) == "long-query-branch / 10",
                    )
                ) or 0
            ) > (
                SFT8.rate(
                    SFT8.accuracy(
                        full_break,
                        lambda row: SFT8.challenge_scale(row) == "long-query-branch / 10",
                    )
                ) or 0
            ),
        }
        for name, passed in checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    else:
        lines.append("Full-state or delta-state result files are missing.")
    lines += ["", "Adopt delta-state traces only if they improve structural generalization without materially harming original V13 retention.", ""]

    examples = []
    if full_v13 and delta_v13:
        examples += SFT8.paired_examples("original-v13", full_v13, delta_v13)
    if full_break and delta_break:
        examples += SFT8.paired_examples("v13.1-breakpoint", full_break, delta_break)
    return "\n".join(lines), examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT.parents[1] / "data" / "spatial_v13_trace_ablation_manifest.json",
    )
    args = parser.parse_args()
    report, examples = summarize(args.results_dir, args.manifest)
    (args.results_dir / "TRACE-ABLATION-SUMMARY.md").write_text(report + "\n")
    (args.results_dir / "TRACE-ABLATION-PAIRED-CHANGES.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples)
    )
    print(report)


if __name__ == "__main__":
    main()
