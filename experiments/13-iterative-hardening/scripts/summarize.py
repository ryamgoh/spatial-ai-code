"""Summarize V12-to-V13 transfer at structural and semantic levels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
TASK = "spatial_eval_v13_diagnostic"
DEFAULT_TAGS = ("baseline", "v12-4b-1.5k", "v12-4b-6k")
CYCLE_CELL_RE = re.compile(
    r"^cycle-(dir-1|which-2|count-1)-"
    r"(diagonal|cardinal|mixed)-(x|y|both)-(direct|indirect)-"
    r"(query-connected|disconnected)-l(\d+)(-control)?$"
)
DEPTH_CELL_RE = re.compile(
    r"^depth-d(\d+)-(cardinal|mixed)-(none|disconnected|query-branch)$"
)


def letters(raw: object) -> tuple[str, ...]:
    parts = re.split(r"[,;|\s]+", str(raw or "").strip().upper())
    return tuple(sorted(set(parts)))


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
    match = re.findall(
        r"Answer:\s*([A-E](?:\s*,\s*[A-E])*)", str(raw or ""), re.I
    )
    return letters(match[-1]) if match else ()


def gold(sample: dict) -> tuple[str, ...]:
    doc = sample.get("doc") or {}
    return letters(doc.get("oracle_option") or sample.get("target"))


def strict_correct(sample: dict) -> bool:
    expected = gold(sample)
    return bool(expected) and prediction(sample) == expected


def load_samples(results_dir: Path, tag: str) -> list[dict]:
    path = results_dir / tag / f"responses_{TASK}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _difficulty(sample: dict) -> dict:
    return dict((sample.get("doc") or {}).get("difficulty") or {})


def _cell(sample: dict) -> str:
    return str((sample.get("doc") or {}).get("generation_cell") or "unknown")


def _question_family(sample: dict) -> str:
    return {0: "direction", 1: "which", 2: "count"}.get(
        _difficulty(sample).get("question_type"), "unknown"
    )


def _depth_group(sample: dict) -> str | None:
    match = DEPTH_CELL_RE.match(_cell(sample))
    return match.group(1) if match else None


def _distractor_group(sample: dict) -> str | None:
    match = DEPTH_CELL_RE.match(_cell(sample))
    return match.group(3) if match else None


def _cycle_group(sample: dict) -> str | None:
    match = CYCLE_CELL_RE.match(_cell(sample))
    if not match:
        return None
    return "open control" if match.group(7) else "closed cycle"


def _cycle_dimension(sample: dict, index: int) -> str | None:
    match = CYCLE_CELL_RE.match(_cell(sample))
    return match.group(index) if match else None


def tally(
    samples: Iterable[dict],
    classifier: Callable[[dict], str | None],
) -> dict[str, tuple[int, int, int]]:
    values: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for sample in samples:
        bucket = classifier(sample)
        if bucket is None:
            continue
        values[bucket][1] += 1
        if strict_correct(sample):
            values[bucket][0] += 1
        if not prediction(sample):
            values[bucket][2] += 1
    return {key: tuple(value) for key, value in values.items()}


def pct(ok: int, total: int) -> str:
    return "—" if total == 0 else f"{100.0 * ok / total:.1f}% ({ok}/{total})"


def cell_macro(samples: Iterable[dict]) -> str:
    grouped = tally(samples, _cell)
    values = [ok / total for ok, total, _invalid in grouped.values() if total]
    return "—" if not values else f"{100.0 * sum(values) / len(values):.1f}%"


def _table(
    title: str,
    tags: tuple[str, ...],
    samples: dict[str, list[dict]],
    classifier: Callable[[dict], str | None],
    order: Iterable[str] | None = None,
) -> list[str]:
    grouped = {tag: tally(samples[tag], classifier) for tag in tags}
    buckets = list(order or sorted({key for rows in grouped.values() for key in rows}))
    lines = [
        f"## {title}",
        "",
        "| bucket |" + "".join(f" {tag} |" for tag in tags),
        "|---|" + "---:|" * len(tags),
    ]
    for bucket in buckets:
        cells = [pct(*grouped[tag].get(bucket, (0, 0, 0))[:2]) for tag in tags]
        lines.append(f"| `{bucket}` | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _write_cell_csv(
    path: Path, tags: tuple[str, ...], samples: dict[str, list[dict]]
) -> None:
    grouped = {tag: tally(samples[tag], _cell) for tag in tags}
    cells = sorted({cell for rows in grouped.values() for cell in rows})
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation_cell", *tags])
        for cell in cells:
            writer.writerow(
                [cell, *[pct(*grouped[tag].get(cell, (0, 0, 0))[:2]) for tag in tags]]
            )


def _paired_error_lines(samples: dict[str, list[dict]]) -> list[str]:
    left, right = "v12-4b-1.5k", "v12-4b-6k"
    if not samples.get(left) or not samples.get(right):
        return []

    def index(rows: list[dict]) -> dict[str, bool]:
        return {
            str((row.get("doc") or {}).get("text") or ""): strict_correct(row)
            for row in rows
        }

    a, b = index(samples[left]), index(samples[right])
    shared = sorted(set(a) & set(b))
    both_wrong = sum(not a[key] and not b[key] for key in shared)
    fixed_by_6k = sum(not a[key] and b[key] for key in shared)
    regressed_at_6k = sum(a[key] and not b[key] for key in shared)
    return [
        "## Paired 1.5k versus 6k errors",
        "",
        f"Compared on {len(shared)} identical prompts: {both_wrong} both wrong, "
        f"{fixed_by_6k} fixed by 6k, {regressed_at_6k} regressed at 6k.",
        "",
    ]


def summarize(results_dir: Path, tags: tuple[str, ...]) -> str:
    samples = {tag: load_samples(results_dir, tag) for tag in tags}
    lines = [
        "# Exp 13 diagnostic summary",
        "",
        "Frozen Qwen3.5-4B baselines evaluated on the same 2,256-row V13 suite. "
        "All accuracies are strict answer-letter-set accuracy.",
        "",
        "| model | overall | cell macro | invalid output |",
        "|---|---:|---:|---:|",
    ]
    for tag in tags:
        total = len(samples[tag])
        correct = sum(strict_correct(row) for row in samples[tag])
        invalid = sum(not prediction(row) for row in samples[tag])
        lines.append(
            f"| `{tag}` | {pct(correct, total)} | {cell_macro(samples[tag])} | "
            f"{pct(invalid, total)} |"
        )
    lines.append("")

    lines += _table(
        "Question family", tags, samples, _question_family,
        ("direction", "which", "count"),
    )
    lines += _table(
        "Relation mode", tags, samples,
        lambda row: str(_difficulty(row).get("relation_mix") or "unknown"),
        ("diagonal-only", "cardinal-only", "mixed"),
    )
    lines += _table(
        "Semantic subtype", tags, samples,
        lambda row: str(_difficulty(row).get("semantic_subtype") or "unknown"),
    )
    lines += _table("Controlled proof depth", tags, samples, _depth_group, map(str, range(1, 6)))
    lines += _table(
        "Controlled distractor condition", tags, samples, _distractor_group,
        ("none", "disconnected", "query-branch"),
    )
    lines += _table(
        "Cycle versus open-chain control", tags, samples, _cycle_group,
        ("closed cycle", "open control"),
    )
    lines += _table(
        "Cycle/control by question family", tags, samples,
        lambda row: (
            f"{_cycle_dimension(row, 1)} / {_cycle_group(row)}"
            if _cycle_group(row) else None
        ),
    )
    lines += _table(
        "Cycle/control by placement", tags, samples,
        lambda row: (
            f"{_cycle_dimension(row, 5)} / {_cycle_group(row)}"
            if _cycle_group(row) else None
        ),
    )
    lines += _paired_error_lines(samples)
    lines += [
        "The complete 233-cell table is in `BUCKETS.csv`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    args = parser.parse_args()
    tags = tuple(tag.strip() for tag in args.tags.split(",") if tag.strip())
    args.results_dir.mkdir(parents=True, exist_ok=True)
    samples = {tag: load_samples(args.results_dir, tag) for tag in tags}
    report = summarize(args.results_dir, tags)
    (args.results_dir / "SUMMARY.md").write_text(report + "\n")
    _write_cell_csv(args.results_dir / "BUCKETS.csv", tags, samples)
    print(report)


if __name__ == "__main__":
    main()
