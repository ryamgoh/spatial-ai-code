"""Summarize cell, module, and depth accuracy for the frozen benchmark."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _metric(result: dict[str, Any], name: str) -> float:
    for key, value in result.items():
        if key == name or key.startswith(name + ","):
            return float(value)
    raise ValueError(f"task result lacks {name}")


def _depth(cell: str) -> int | None:
    match = re.fullmatch(r"(?:direction|which|count)-depth-(\d+)", cell)
    return int(match.group(1)) if match else None


def _display_metric(modules: dict[str, float], role: str) -> str:
    return f"{modules[role]:.4f}" if role in modules else "—"


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
        component_correct: dict[str, float] = defaultdict(float)
        component_total: dict[str, int] = defaultdict(int)
        depth_values: dict[str, dict[int, float]] = defaultdict(dict)
        weighted_correct = 0.0
        weighted_total = 0
        for task_name, metadata in tasks.items():
            if task_name not in raw_results:
                raise ValueError(f"{name} results omit task {task_name}")
            strict = _metric(raw_results[task_name], "strict_acc")
            loose = _metric(raw_results[task_name], "loose_acc")
            count = int(metadata["count"])
            cells[task_name] = {**metadata, "strict_acc": strict, "loose_acc": loose}
            role = str(metadata["role"])
            component = str(metadata["component"])
            group_correct[role] += strict * count
            group_total[role] += count
            component_key = f"{role}/{component}"
            component_correct[component_key] += strict * count
            component_total[component_key] += count
            weighted_correct += strict * count
            weighted_total += count
            depth = _depth(metadata["cell"])
            if depth is not None:
                depth_values[str(metadata["query_kind"])][depth] = strict
        models[name] = {
            "complete": True,
            "overall_micro_accuracy": weighted_correct / weighted_total,
            "group_accuracy": {
                role: group_correct[role] / count
                for role, count in sorted(group_total.items())
            },
            "component_accuracy": {
                component: component_correct[component] / count
                for component, count in sorted(component_total.items())
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
    lines.append(
        "| Model | Overall | CORE | COMP | SHIFT | SHIFT composition | SHIFT depth |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, result in models.items():
        if not result["complete"]:
            lines.append(f"| {name} | incomplete | — | — | — | — | — |")
            continue
        groups = result["group_accuracy"]
        components = result["component_accuracy"]
        lines.append(
            f"| {name} | {result['overall_micro_accuracy']:.4f} | "
            f"{_display_metric(groups, 'bench-core')} | "
            f"{_display_metric(groups, 'bench-composition')} | "
            f"{_display_metric(groups, 'bench-shift')} | "
            f"{_display_metric(components, 'bench-shift/heldout-composition')} | "
            f"{_display_metric(components, 'bench-shift/depth')} |"
        )
    lines.extend(["", "Cell-level and depth-curve values are in `summary.json`."])
    (prepared_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
