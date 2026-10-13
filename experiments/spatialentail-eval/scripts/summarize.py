"""Summarize section, cell, query-family, and depth diagnostics."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

GROUPS = (
    ("bench-core", "CORE"),
    ("bench-composition", "COMP"),
    ("bench-shift", "SHIFT"),
)
QUERY_KINDS = (
    ("direction", "Direction"),
    ("which", "Selection"),
    ("count", "Count"),
)


def _metric(result: dict[str, Any], name: str) -> float:
    for key, value in result.items():
        if key == name or key.startswith(name + ","):
            return float(value)
    raise ValueError(f"task result lacks {name}")


def _optional_metric(result: dict[str, Any], name: str) -> float | None:
    try:
        return _metric(result, name)
    except ValueError:
        return None


def _depth(cell: str) -> int | None:
    match = re.fullmatch(r"(?:direction|which|count)-depth-(\d+)", cell)
    return int(match.group(1)) if match else None


def _display_metric(modules: dict[str, float], role: str) -> str:
    return f"{modules[role]:.4f}" if role in modules else "—"


def _display_value(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def summarize(prepared_dir: Path) -> dict[str, Any]:
    prepared_dir = prepared_dir.resolve()
    receipt = json.loads((prepared_dir / "jobs.json").read_text())
    tasks = {item["task"]: item for item in receipt["tasks"]}
    models: dict[str, Any] = {}
    for job in receipt["jobs"]:
        name = job["name"]
        result_dir = prepared_dir / "results" / name
        result_path = result_dir / "results.json"
        if not (result_dir / "COMPLETED").is_file() or not result_path.is_file():
            models[name] = {"complete": False}
            continue
        payload = json.loads(result_path.read_text())
        raw_results = payload.get("results") or payload
        cells = {}
        group_correct: dict[str, float] = defaultdict(float)
        group_total: dict[str, int] = defaultdict(int)
        group_query_correct: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        group_query_total: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        depth_values: dict[str, dict[int, float]] = defaultdict(dict)
        stage1_correct = 0.0
        stage1_format_valid = 0.0
        stage1_total = 0
        weighted_correct = 0.0
        weighted_total = 0
        for task_name, metadata in tasks.items():
            if task_name not in raw_results:
                raise ValueError(f"{name} results omit task {task_name}")
            strict = _metric(raw_results[task_name], "strict_acc")
            loose = _metric(raw_results[task_name], "loose_acc")
            stage1 = _optional_metric(raw_results[task_name], "stage1_acc")
            stage1_format = _optional_metric(
                raw_results[task_name], "stage1_format_valid"
            )
            count = int(metadata["count"])
            cells[task_name] = {
                **metadata,
                "strict_acc": strict,
                "loose_acc": loose,
                "stage1_acc": stage1,
                "stage1_format_valid": stage1_format,
            }
            if stage1 is not None and stage1_format is not None:
                stage1_correct += stage1 * count
                stage1_format_valid += stage1_format * count
                stage1_total += count
            role = str(metadata["role"])
            group_correct[role] += strict * count
            group_total[role] += count
            query_kind = str(metadata["query_kind"])
            group_query_correct[role][query_kind] += strict * count
            group_query_total[role][query_kind] += count
            weighted_correct += strict * count
            weighted_total += count
            depth = _depth(metadata["cell"])
            if depth is not None:
                depth_values[str(metadata["query_kind"])][depth] = strict
        models[name] = {
            "complete": True,
            "overall_micro_accuracy": weighted_correct / weighted_total,
            "stage1_overall_micro_accuracy": (
                stage1_correct / stage1_total
                if stage1_total == weighted_total
                else None
            ),
            "stage1_format_validity": (
                stage1_format_valid / stage1_total
                if stage1_total == weighted_total
                else None
            ),
            "group_accuracy": {
                role: group_correct[role] / count
                for role, count in sorted(group_total.items())
            },
            "group_query_accuracy": {
                role: {
                    query_kind: group_query_correct[role][query_kind] / count
                    for query_kind, count in sorted(query_totals.items())
                }
                for role, query_totals in sorted(group_query_total.items())
            },
            "depth_accuracy": {
                family: {str(depth): value for depth, value in sorted(values.items())}
                for family, values in sorted(depth_values.items())
            },
            "cells": cells,
        }
    summary = {
        "schema": "spatialentail-frozen-eval-summary-v1",
        "total_bases": receipt["total_bases"],
        "models": models,
    }
    (prepared_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = ["# SpatialEntail Frozen Benchmark", ""]
    lines.append("Stage 2 constrained-choice accuracy is the primary metric.")
    lines.append("")
    lines.append("| Model | Overall | CORE | COMP | SHIFT |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, result in models.items():
        if not result["complete"]:
            lines.append(f"| {name} | incomplete | — | — | — |")
            continue
        groups = result["group_accuracy"]
        lines.append(
            f"| {name} | {result['overall_micro_accuracy']:.4f} | "
            f"{_display_metric(groups, 'bench-core')} | "
            f"{_display_metric(groups, 'bench-composition')} | "
            f"{_display_metric(groups, 'bench-shift')} |"
        )
    lines.extend(
        [
            "",
            "All within-group diagnostics are in `summary.json`: section cell "
            "results, query-family summaries, and optional depth-curve details.",
        ]
    )
    lines.extend(
        [
            "",
            "## Stage comparison",
            "",
            "| Model | Stage 1 native | Stage 1 canonical format | Stage 2 constrained |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in models.items():
        if not result["complete"]:
            lines.append(f"| {name} | incomplete | — | — |")
            continue
        stage1 = result["stage1_overall_micro_accuracy"]
        stage1_format = result["stage1_format_validity"]
        lines.append(
            f"| {name} | "
            f"{stage1:.4f} | " if stage1 is not None else f"| {name} | — | "
        )
        suffix = (
            f"{stage1_format:.4f} | " if stage1_format is not None else "— | "
        )
        lines[-1] += suffix + f"{result['overall_micro_accuracy']:.4f} |"

    lines.extend(
        [
            "",
            "## Cell-name legend",
            "",
            "| Code | Meaning |",
            "|---|---|",
            "| `DIR` | Direction Identification query |",
            "| `SEL` | Entity Selection query |",
            "| `CNT` | Cardinality Determination query |",
            "| `ARC` | Atomic Relation Comprehension |",
            "| `TRC` | Transitive Relation Composition |",
            "| `AXC` | Axis Composition |",
            "| `MP` | Modus Ponens |",
            "| `IFF` | Equivalence Elimination |",
            "| `NEG` | Negation |",
            "| `DJE` | Disjunctive Elimination |",
            "| `CSR` | Case-Split Reasoning |",
            "| `JCR` | Joint Count Reasoning obligation |",
            "| `UNI` | Unique or invariant answer |",
            "| `ALT` | Alternative answers across valid worlds |",
            "| `MUL` | Multiple certain Selection matches |",
            "| `NOM` | No possible listed match |",
            "| `NOISE` | Solver-verified removable distractors |",
        ]
    )

    for role, label in GROUPS:
        section_tasks = sorted(
            (item for item in tasks.values() if item["role"] == role),
            key=lambda item: (str(item["component"]), str(item["cell"])),
        )
        section_total = sum(int(item["count"]) for item in section_tasks)
        lines.extend(
            [
                "",
                f"## {label} breakdown ({section_total:,} examples)",
                "",
                "| Model | Section cumulative | Direction | Selection | Count |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, result in models.items():
            if not result["complete"]:
                lines.append(f"| {name} | incomplete | — | — | — |")
                continue
            query_values = result["group_query_accuracy"].get(role, {})
            lines.append(
                f"| {name} | {_display_metric(result['group_accuracy'], role)} | "
                + " | ".join(
                    _display_value(query_values.get(query_kind))
                    for query_kind, _ in QUERY_KINDS
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "| Cell | Slice | N | " + " | ".join(models) + " |",
                "|---|---|---:|" + "---:|" * len(models),
            ]
        )
        for item in section_tasks:
            values = []
            for result in models.values():
                value = (
                    result["cells"][item["task"]]["strict_acc"]
                    if result["complete"]
                    else None
                )
                values.append(_display_value(value))
            lines.append(
                f"| `{item['cell']}` | {item['component']} | {item['count']} | "
                + " | ".join(values)
                + " |"
            )
    (prepared_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
