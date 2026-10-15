"""Summarize an engineering SpatialEntail-ALL-POSSIBLE pilot bundle."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def _metric(result: dict[str, Any], name: str) -> float:
    for key, value in result.items():
        if key == name or key.startswith(name + ","):
            return float(value)
    raise ValueError(f"task result lacks {name}")


def _tag(cell: str, prefix: str) -> str | None:
    match = re.search(rf"(?:^|-){re.escape(prefix)}(\d+)(?:$|-)", cell)
    return match.group(1) if match else None


def _weighted(
    cells: dict[str, dict[str, Any]],
    field: str,
    predicate: Callable[[dict[str, Any]], bool] = lambda _: True,
) -> tuple[float, int]:
    selected = [cell for cell in cells.values() if predicate(cell)]
    total = sum(int(cell["count"]) for cell in selected)
    if not total:
        raise ValueError(f"no cells selected for {field}")
    return (
        sum(float(cell[field]) * int(cell["count"]) for cell in selected) / total,
        total,
    )


def _breakdown(
    cells: dict[str, dict[str, Any]],
    key: Callable[[dict[str, Any]], str | None],
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells.values():
        label = key(cell)
        if label is not None:
            groups[label].append(cell)
    result = {}
    for label, members in sorted(groups.items()):
        count = sum(int(cell["count"]) for cell in members)
        result[label] = {
            "count": count,
            "exact_set_accuracy": sum(
                float(cell["strict_acc"]) * int(cell["count"]) for cell in members
            )
            / count,
        }
    return result


def _model_summary(
    payload: dict[str, Any], tasks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    raw_results = payload.get("results") or payload
    cells = {}
    for task_name, task in tasks.items():
        if task_name not in raw_results:
            raise ValueError(f"results omit task {task_name}")
        result = raw_results[task_name]
        strict = _metric(result, "strict_acc")
        loose = _metric(result, "loose_acc")
        if loose + 1e-12 < strict:
            raise ValueError(f"task {task_name}: loose accuracy is below strict")
        cell_name = str(task["cell"])
        parts = cell_name.split("-")
        reasoning = "+".join(
            part
            for part in parts
            if part in {"ARC", "TRC", "AXC", "MP", "IFF", "NEG", "DJE", "CSR"}
        )
        cells[task_name] = {
            **task,
            "strict_acc": strict,
            "loose_acc": loose,
            "stage1_acc": _metric(result, "stage1_acc"),
            "stage1_format_valid": _metric(result, "stage1_format_valid"),
            "section": str(task.get("section") or "matched"),
            "reasoning_family": reasoning,
            "gold_size": _tag(cell_name, "G"),
            "depth": _tag(cell_name, "D"),
            "diagnostic": (
                "JCR"
                if "JCR" in parts
                else "NOISE"
                if "NOISE" in parts
                else None
            ),
        }

    strict, total = _weighted(cells, "strict_acc")
    loose, _ = _weighted(cells, "loose_acc")
    stage1, _ = _weighted(cells, "stage1_acc")
    stage1_format, _ = _weighted(cells, "stage1_format_valid")
    return {
        "complete": True,
        "examples": total,
        "exact_set_accuracy": strict,
        "gold_covered_accuracy": loose,
        "over_inclusive_rate": loose - strict,
        "missing_gold_rate": 1.0 - loose,
        "stage1_exact_set_accuracy": stage1,
        "stage1_format_validity": stage1_format,
        "query_accuracy": _breakdown(cells, lambda cell: str(cell["query_kind"])),
        "section_accuracy": _breakdown(cells, lambda cell: str(cell["section"])),
        "reasoning_accuracy": _breakdown(
            cells, lambda cell: str(cell["reasoning_family"])
        ),
        "gold_size_accuracy": _breakdown(
            cells,
            lambda cell: (
                str(cell["gold_size"]) if cell["gold_size"] is not None else None
            ),
        ),
        "depth_accuracy": _breakdown(
            cells,
            lambda cell: str(cell["depth"]) if cell["depth"] is not None else None,
        ),
        "diagnostic_accuracy": _breakdown(
            cells,
            lambda cell: (
                str(cell["diagnostic"])
                if cell["diagnostic"] is not None
                else None
            ),
        ),
        "cells": cells,
    }


def _display(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _breakdown_lines(
    title: str,
    field: str,
    models: dict[str, dict[str, Any]],
) -> list[str]:
    labels = sorted(
        {
            label
            for model in models.values()
            if model.get("complete")
            for label in model[field]
        }
    )
    names = list(models)
    lines = [f"## {title}", "", "| Slice | N | " + " | ".join(names) + " |"]
    lines.append("|---|---:|" + "---:|" * len(names))
    for label in labels:
        count = next(
            model[field][label]["count"]
            for model in models.values()
            if model.get("complete") and label in model[field]
        )
        values = []
        for model in models.values():
            item = model.get(field, {}).get(label)
            values.append(_display(item["exact_set_accuracy"] if item else None))
        lines.append(f"| {label} | {count} | " + " | ".join(values) + " |")
    return lines + [""]


def _render_markdown(summary: dict[str, Any]) -> str:
    models = summary["models"]
    lines = [
        "# SpatialEntail-ALL-POSSIBLE calibration pilot",
        "",
        "Engineering-only zero-shot transfer result; this draft pilot is not a "
        "frozen benchmark.",
        "Stage-2 constrained exact-set accuracy is the primary metric.",
        "",
        "| Model | Exact set | Stage 1 exact set | Stage 1 format | Gold covered "
        "| Over-inclusive | Missing gold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in models.items():
        if not model["complete"]:
            lines.append(f"| {name} | incomplete | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {_display(model['exact_set_accuracy'])} | "
            f"{_display(model['stage1_exact_set_accuracy'])} | "
            f"{_display(model['stage1_format_validity'])} | "
            f"{_display(model['gold_covered_accuracy'])} | "
            f"{_display(model['over_inclusive_rate'])} | "
            f"{_display(model['missing_gold_rate'])} |"
        )
    lines.append("")
    lines.extend(_breakdown_lines("Section", "section_accuracy", models))
    lines.extend(_breakdown_lines("Query kind", "query_accuracy", models))
    lines.extend(
        _breakdown_lines("Reasoning family", "reasoning_accuracy", models)
    )
    lines.extend(_breakdown_lines("Gold set size", "gold_size_accuracy", models))
    lines.extend(_breakdown_lines("Depth", "depth_accuracy", models))
    lines.extend(
        _breakdown_lines("Structural diagnostics", "diagnostic_accuracy", models)
    )

    complete_names = [name for name, model in models.items() if model["complete"]]
    tasks = summary["tasks"]
    lines.extend(
        [
            "## Cell breakdown",
            "",
            "| Cell | Query | N | " + " | ".join(complete_names) + " |",
            "|---|---|---:|" + "---:|" * len(complete_names),
        ]
    )
    for task_name, task in tasks.items():
        values = [
            _display(models[name]["cells"][task_name]["strict_acc"])
            for name in complete_names
        ]
        lines.append(
            f"| `{task['cell']}` | {task['query_kind']} | {task['count']} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines) + "\n"


def summarize(prepared_dir: Path) -> dict[str, Any]:
    prepared_dir = prepared_dir.resolve()
    receipt = json.loads((prepared_dir / "jobs.json").read_text())
    if receipt.get("schema") != "spatialentail-all-possible-pilot-v1":
        raise ValueError("not a SpatialEntail-ALL-POSSIBLE pilot receipt")
    tasks = {str(item["task"]): item for item in receipt["tasks"]}
    models = {}
    for job in receipt["jobs"]:
        name = str(job["name"])
        result_dir = prepared_dir / "results" / name
        result_path = result_dir / "results.json"
        if not (result_dir / "COMPLETED").is_file() or not result_path.is_file():
            models[name] = {"complete": False}
            continue
        models[name] = _model_summary(json.loads(result_path.read_text()), tasks)
    summary = {
        "schema": "spatialentail-all-possible-pilot-summary-v1",
        "status": "engineering-only",
        "total_rows": int(receipt["total_rows"]),
        "samples_per_cell": int(receipt["samples_per_cell"]),
        "tasks": tasks,
        "models": models,
    }
    (prepared_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (prepared_dir / "SUMMARY.md").write_text(_render_markdown(summary))
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} PREPARED_DIR")
    summarize(Path(sys.argv[1]))
