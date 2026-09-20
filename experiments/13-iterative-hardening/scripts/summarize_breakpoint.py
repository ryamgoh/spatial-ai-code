"""Summarize where native V13 models break on the V13.1 challenge."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = Path(__file__).with_name("summarize.py")
SPEC = importlib.util.spec_from_file_location(
    "v13_breakpoint_summary_helpers", SUMMARY_PATH
)
assert SPEC and SPEC.loader
V13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V13)

TASK = "spatial_eval_v13_breakpoint"
DEFAULT_TAGS = (
    "breakpoint-base-stage1",
    "breakpoint-sft-1.5k-stage1",
    "breakpoint-sft-6k-stage1",
)
DEPTH_RE = re.compile(
    r"^break-(long-clean|long-disconnected|long-query-branch|unequal-clean|unequal-query-branch)-"
    r"(cardinal|mixed)-x(\d+)-y(\d+)-(none|disconnected|query-branch)$"
)
LOOP_RE = re.compile(
    r"^break-long-loop-(direction|which|count)-(diagonal|cardinal|mixed)-"
    r"(x|y|both)-(query-connected|disconnected)-l(6|8|10)-"
    r"(closed|open-control)$"
)


def load(results_dir: Path, tag: str) -> list[dict]:
    path = results_dir / tag / f"responses_{TASK}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell(row: dict) -> str:
    return str((row.get("doc") or {}).get("generation_cell") or "")


def challenge(row: dict) -> str | None:
    name = cell(row)
    depth = DEPTH_RE.match(name)
    if depth:
        return depth.group(1)
    loop = LOOP_RE.match(name)
    if loop:
        return "long closed loop" if loop.group(6) == "closed" else "matched open chain"
    return None


def scale(row: dict) -> str | None:
    depth = DEPTH_RE.match(cell(row))
    if depth:
        return str(max(int(depth.group(3)), int(depth.group(4))))
    loop = LOOP_RE.match(cell(row))
    return loop.group(5) if loop else None


def relation_mode(row: dict) -> str | None:
    depth = DEPTH_RE.match(cell(row))
    if depth:
        return depth.group(2)
    loop = LOOP_RE.match(cell(row))
    return loop.group(2) if loop else None


def loop_family(row: dict) -> str | None:
    loop = LOOP_RE.match(cell(row))
    return (
        f"{loop.group(1)} / {'closed' if loop.group(6) == 'closed' else 'open'}"
        if loop
        else None
    )


def tally(
    rows: list[dict], classifier: Callable[[dict], str | None]
) -> dict[str, tuple[int, int]]:
    result: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = classifier(row)
        if bucket is None:
            continue
        result[bucket][1] += 1
        result[bucket][0] += int(V13.strict_correct(row))
    return {key: tuple(value) for key, value in result.items()}


def table(
    title: str,
    tags: tuple[str, ...],
    samples: dict[str, list[dict]],
    classifier: Callable[[dict], str | None],
    order: tuple[str, ...],
) -> list[str]:
    grouped = {tag: tally(samples[tag], classifier) for tag in tags}
    lines = [
        f"## {title}",
        "",
        "| bucket | " + " | ".join(tags) + " |",
        "|---|" + "---:|" * len(tags),
    ]
    for bucket in order:
        lines.append(
            f"| `{bucket}` | "
            + " | ".join(V13.pct(*grouped[tag].get(bucket, (0, 0))) for tag in tags)
            + " |"
        )
    lines.append("")
    return lines


def summarize(results_dir: Path, tags: tuple[str, ...]) -> str:
    samples = {tag: load(results_dir, tag) for tag in tags}
    prompt_sets = [
        {str((row.get("doc") or {}).get("text") or "") for row in rows}
        for rows in samples.values()
        if rows
    ]
    if prompt_sets and not all(values == prompt_sets[0] for values in prompt_sets[1:]):
        raise ValueError("breakpoint results do not contain identical prompts")

    lines = [
        "# V13.1 structural breakpoint",
        "",
        "Frozen 1,224-row evaluation-only suite. Stage-1 strict answer-letter-set accuracy.",
        "",
        "| model | overall | invalid output |",
        "|---|---:|---:|",
    ]
    for tag, rows in samples.items():
        correct = sum(V13.strict_correct(row) for row in rows)
        invalid = sum(not V13.prediction(row) for row in rows)
        lines.append(
            f"| `{tag}` | {V13.pct(correct, len(rows))} | {V13.pct(invalid, len(rows))} |"
        )
    lines.append("")
    lines += table(
        "Challenge family",
        tags,
        samples,
        challenge,
        (
            "long-clean",
            "long-disconnected",
            "long-query-branch",
            "unequal-clean",
            "unequal-query-branch",
            "long closed loop",
            "matched open chain",
        ),
    )
    lines += table("Structural scale", tags, samples, scale, ("6", "8", "10"))
    lines += table(
        "Relation mode", tags, samples, relation_mode, ("cardinal", "mixed", "diagonal")
    )
    lines += table(
        "Long loop/control by question family",
        tags,
        samples,
        loop_family,
        (
            "direction / closed",
            "direction / open",
            "which / closed",
            "which / open",
            "count / closed",
            "count / open",
        ),
    )
    lines += [
        "## Interpretation bands",
        "",
        "- Above 90%: this suite is already close to saturation.",
        "- 50–80%: useful controlled headroom for the next SFT curriculum.",
        "- Below 30%: use an intermediate curriculum before training on the hardest bucket.",
        "",
        "Choose the next training intervention from the first scale/bucket with a material drop; do not train on this frozen suite itself.",
        "",
    ]
    return "\n".join(lines)


def write_cells(
    path: Path, tags: tuple[str, ...], samples: dict[str, list[dict]]
) -> None:
    grouped = {tag: tally(samples[tag], cell) for tag in tags}
    names = sorted({name for values in grouped.values() for name in values})
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation_cell", *tags])
        for name in names:
            writer.writerow(
                [name, *[V13.pct(*grouped[tag].get(name, (0, 0))) for tag in tags]]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    args = parser.parse_args()
    tags = tuple(tag.strip() for tag in args.tags.split(",") if tag.strip())
    report = summarize(args.results_dir, tags)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "BREAKPOINT-SUMMARY.md").write_text(report + "\n")
    write_cells(
        args.results_dir / "BREAKPOINT-CELLS.csv",
        tags,
        {tag: load(args.results_dir, tag) for tag in tags},
    )
    print(report)


if __name__ == "__main__":
    main()
