"""Validate and summarize downloaded randomized-v2 evaluations without GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("strict_acc", "stage1_acc", "stage1_format_valid")
SCHEMAS = {
    "spatialentail-randomized-v2-eval-jobs-v1",
    "spatialentail-general-instruction-eval-v1",
    "spatialentail-reasoning-eval-v1",
}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def metric(result: dict, key: str) -> float:
    value = float(result[key] if key in result else result[f"{key},none"])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid {key}: {value}")
    return value


def tasks_for(receipt: dict, name: str, general: Path | None) -> list[dict]:
    if receipt["schema"] != "spatialentail-reasoning-eval-v1":
        return receipt["tasks"]
    if general is None:
        raise ValueError("Reasoning summaries require --general-prepared")
    path = general / "jobs.json"
    if digest(path) != receipt["upstream_general_receipt_sha256"]:
        raise ValueError("Reasoning upstream general receipt hash mismatch")
    upstream = json.loads(path.read_text())
    if upstream["benchmark_sha256"] != receipt["benchmark_sha256"]:
        raise ValueError("Reasoning benchmark hash mismatch")
    slug = lambda value: re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    tasks = [
        {
            **task,
            "task": (
                f"spatialentail_reasoning_{slug(name)}_"
                f"{slug(task['query_kind'])}_g{task['gold_size']}_"
                f"{slug(task['partition'])}"
            ),
        }
        for task in upstream["tasks"]
    ]
    if len(tasks) != receipt["tasks_per_model"]:
        raise ValueError("Reasoning task count mismatch")
    return tasks


def scores(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        **{key: mean(row[key] for row in rows) for key in METRICS},
    }


def grouped(rows: list[dict], dimension: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        values = (
            row[dimension] if dimension == "heldout_transition" else [row[dimension]]
        )
        for value in values:
            groups[str(value)].append(row)
    return {key: scores(values) for key, values in sorted(groups.items())}


def describe(rows: list[dict]) -> dict:
    partitions = grouped(rows, "partition")
    return {
        **scores(rows),
        "partition_macro_acc": mean(
            value["strict_acc"] for value in partitions.values()
        ),
        "by_partition": partitions,
        **{
            f"by_{dimension}": grouped(rows, dimension)
            for dimension in ("query", "G", "L", "X", "heldout_transition")
        },
    }


def sample_row(sample: dict, task: dict) -> dict:
    doc = sample["doc"]
    metadata = doc["metadata"]
    difficulty = metadata["difficulty"]
    for field, expected in (
        ("query_kind", task["query_kind"]),
        ("gold_size", task["gold_size"]),
        ("benchmark_partition", task["partition"]),
    ):
        if metadata[field] != expected:
            raise ValueError(f"Sample metadata disagrees with task: {field}")
    contract = metadata["answer_mode"]
    expected_contract = "single" if task["gold_size"] == 1 else "all-possible"
    if contract != expected_contract:
        raise ValueError(f"Unexpected G/contract combination: {contract}")
    values = {key: metric(sample, key) for key in METRICS}
    if any(value not in (0, 1) for value in values.values()):
        raise ValueError("Per-example metrics must be binary")
    return {
        "base_id": metadata["base_id"],
        "target": doc["oracle_option"],
        "contract": contract,
        "partition": task["partition"],
        "query": task["query_kind"],
        "G": task["gold_size"],
        "L": difficulty["proof_complexity"]["semantic_length"],
        "X": difficulty["num_distractor_premises"],
        "heldout_transition": metadata["heldout_transitions"],
        **values,
    }


def summarize_model(prepared: Path, job: dict, tasks: list[dict]) -> tuple[dict, str]:
    folder = prepared / "results" / job["name"]
    path = folder / "results.json"
    if not path.is_file() or not (folder / "COMPLETED").is_file():
        return {"complete": False, "reason": "Missing results.json or COMPLETED"}, ""
    payload = json.loads(path.read_text())
    if payload["config"].get("limit") is not None:
        raise ValueError(f"{job['name']}: limited/smoke run is not a full evaluation")
    raw = payload["results"]
    if set(raw) != {task["task"] for task in tasks}:
        raise ValueError(f"{job['name']}: result task coverage mismatch")
    rows, cells, hashes = [], {}, {"results.json": digest(path)}
    for task in tasks:
        name = task["task"]
        count = task["count"]
        reported = raw[name]
        effective = payload.get("n-samples", {}).get(name, {}).get("effective")
        if reported.get("sample_len") != count or effective not in (None, count):
            raise ValueError(f"{job['name']}/{name}: evaluated count mismatch")
        response_path = folder / f"responses_{name}.jsonl"
        task_rows = []
        with response_path.open() as source:
            for line in source:
                task_rows.append(sample_row(json.loads(line), task))
        if len(task_rows) != count:
            raise ValueError(f"{job['name']}/{name}: response count mismatch")
        observed = scores(task_rows)
        for key in METRICS:
            if not math.isclose(observed[key], metric(reported, key), abs_tol=1e-10):
                raise ValueError(f"{job['name']}/{name}: sample/result {key} mismatch")
        hashes[response_path.name] = digest(response_path)
        cells[name] = {**task, **observed}
        rows.extend(task_rows)
    ids = [row["base_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{job['name']}: duplicate base IDs")
    # The fingerprint excludes predictions but binds IDs, labels and slice metadata.
    canonical = [
        {key: value for key, value in row.items() if key not in METRICS}
        for row in sorted(rows, key=lambda row: row["base_id"])
    ]
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()
    ).hexdigest()
    failures = Counter(
        f"stage1_{'correct' if row['stage1_acc'] else 'wrong'}"
        f"_stage2_{'correct' if row['strict_acc'] else 'wrong'}"
        for row in rows
    )
    return {
        "complete": True,
        "job": job,
        "runtime_config": payload["config"],
        "input_sha256": hashes,
        "sample_fingerprint": fingerprint,
        "overall": describe(rows),
        "by_contract": {
            contract: describe([row for row in rows if row["contract"] == contract])
            for contract in sorted({row["contract"] for row in rows})
        },
        "stage_outcomes": dict(sorted(failures.items())),
        "stage1_invalid_format_count": sum(
            not row["stage1_format_valid"] for row in rows
        ),
        "cells": cells,
    }, fingerprint


def seed_summary(models: dict) -> dict:
    output = {}
    for arm in ("single-natural", "single-answer-only", "multi-natural"):
        names = [f"{arm}-seed-{seed}" for seed in (42, 43)]
        if not all(models.get(name, {}).get("complete") for name in names):
            continue
        output[arm] = {}
        for subset in ("overall", "single", "all-possible"):
            values = [
                (
                    models[name]["overall"]
                    if subset == "overall"
                    else models[name]["by_contract"][subset]
                )["strict_acc"]
                for name in names
            ]
            output[arm][subset] = {
                "seeds": [42, 43],
                "values": values,
                "mean": mean(values),
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
            }
    return output


def summarize(prepared: Path, general: Path | None = None) -> dict:
    receipt_path = prepared / "jobs.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["schema"] not in SCHEMAS:
        raise ValueError(f"Unsupported receipt schema: {receipt['schema']}")
    models, fingerprints = {}, set()
    for job in receipt["jobs"]:
        tasks = tasks_for(receipt, job["name"], general)
        if len({task["task"] for task in tasks}) != len(tasks):
            raise ValueError("Duplicate tasks in receipt")
        if sum(task["count"] for task in tasks) != receipt["total_bases"]:
            raise ValueError("Receipt task counts disagree with total_bases")
        result, fingerprint = summarize_model(prepared, job, tasks)
        models[job["name"]] = result
        if fingerprint:
            fingerprints.add(fingerprint)
    if len(fingerprints) > 1:
        raise ValueError("Models did not evaluate the same labeled examples")
    return {
        "schema": "spatialentail-randomized-v2-summary-v1",
        "source_package": prepared.name,
        "jobs_sha256": digest(receipt_path),
        "benchmark_sha256": receipt["benchmark_sha256"],
        "total_bases": receipt["total_bases"],
        "models": models,
        "seed_summary": seed_summary(models),
    }


def table(lines: list[str], header: list[str], rows: list[list[str]]) -> None:
    lines.extend(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)])
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def render(batches: dict) -> str:
    lines = [
        "# Randomized-v2 Evaluation Summary",
        "",
        (
            "Accuracy values are percentages. Stage 2 exact-set accuracy is primary; "
            "stage 1 native extraction and canonical-format validity are diagnostics."
        ),
        "",
        (
            "Overall is example-weighted. Partition macro is the unweighted mean of "
            "the four partition accuracies. SINGLE is G1 (900 rows); ALL-POSSIBLE is "
            "G2-G4 (2,700 rows). L is measured semantic proof length; X is measured "
            "distractor count. Transition slices can overlap."
        ),
        "",
        (
            "Validation checks completion markers, full task/sample counts, unique base "
            "IDs, common labeled examples, and agreement of sample metrics with results.json. "
            "Receipt hashes bind the claimed benchmark; source data, training receipts, "
            "adapter hashes and the cluster environment were not independently audited."
        ),
        "",
        (
            "Raw responses remain in each package's results directory. The JSON report "
            "retains per-task metrics, input hashes, runtime settings, per-contract slices "
            "and diagnostic failure counts. Those categories describe stage outcomes, "
            "not causal reasoning-error diagnoses."
        ),
        "",
        (
            "Compare response regimes cautiously: the P0 untuned baseline and fresh "
            "reasoning runs use different evaluation configurations. Seed ranges below "
            "are descriptive, not confidence intervals."
        ),
        "",
    ]
    for label, batch in batches.items():
        lines.extend([f"## {label}", ""])
        complete = {
            name: model for name, model in batch["models"].items() if model["complete"]
        }
        for name, model in batch["models"].items():
            if not model["complete"]:
                lines.extend([f"- {name}: incomplete ({model['reason']}).", ""])
        table(
            lines,
            [
                "Model",
                "N",
                "Stage 1",
                "Format",
                "Stage 2",
                "Partition macro",
                "SINGLE",
                "ALL-POSSIBLE",
            ],
            [
                [
                    name,
                    str(model["overall"]["n"]),
                    *[
                        pct(model["overall"][key])
                        for key in (
                            "stage1_acc",
                            "stage1_format_valid",
                            "strict_acc",
                            "partition_macro_acc",
                        )
                    ],
                    *[
                        pct(model["by_contract"][contract]["strict_acc"])
                        for contract in ("single", "all-possible")
                    ],
                ]
                for name, model in complete.items()
            ],
        )
        for dimension in ("partition", "query", "G", "L", "X", "heldout_transition"):
            lines.extend([f"### {dimension} breakdown", ""])
            values = sorted(
                {
                    key
                    for model in complete.values()
                    for key in model["overall"][f"by_{dimension}"]
                }
            )
            table(
                lines,
                [dimension, "N per model", *complete],
                [
                    [
                        value,
                        str(
                            next(
                                model["overall"][f"by_{dimension}"][value]["n"]
                                for model in complete.values()
                                if value in model["overall"][f"by_{dimension}"]
                            )
                        ),
                        *[
                            pct(
                                model["overall"][f"by_{dimension}"][value]["strict_acc"]
                            )
                            if value in model["overall"][f"by_{dimension}"]
                            else "N/A"
                            for model in complete.values()
                        ],
                    ]
                    for value in values
                ],
            )
        if batch["seed_summary"]:
            lines.extend(["### SFT Seed Summary", ""])
            table(
                lines,
                [
                    "Arm",
                    "Subset",
                    "Seed 42",
                    "Seed 43",
                    "Mean",
                    "Min",
                    "Max",
                    "Range (pp)",
                ],
                [
                    [
                        arm,
                        subset,
                        *[pct(value) for value in stats["values"]],
                        *[pct(stats[key]) for key in ("mean", "min", "max", "range")],
                    ]
                    for arm, subsets in batch["seed_summary"].items()
                    for subset, stats in subsets.items()
                ],
            )
            lines.extend(
                [
                    "### Primary Comparisons",
                    "",
                    "Deltas are first minus second, in percentage points.",
                    "",
                ]
            )
            comparisons = [
                (
                    "Natural minus answer-only",
                    "single-natural",
                    "single-answer-only",
                    "single",
                ),
                ("MULTI minus untuned", "multi-natural", "untuned", "all-possible"),
                (
                    "MULTI minus SINGLE Natural",
                    "multi-natural",
                    "single-natural",
                    "all-possible",
                ),
                (
                    "SINGLE retention: MULTI minus SINGLE Natural",
                    "multi-natural",
                    "single-natural",
                    "single",
                ),
            ]
            comparison_rows = []
            for title, first, second, subset in comparisons:
                deltas = []
                for seed in (42, 43):
                    a = complete.get(f"{first}-seed-{seed}")
                    b = complete.get(
                        second if second == "untuned" else f"{second}-seed-{seed}"
                    )
                    if a and b:
                        deltas.append(
                            a["by_contract"][subset]["strict_acc"]
                            - b["by_contract"][subset]["strict_acc"]
                        )
                if len(deltas) == 2:
                    comparison_rows.append(
                        [
                            title,
                            subset,
                            *[pct(delta) for delta in deltas],
                            pct(mean(deltas)),
                        ]
                    )
            table(
                lines,
                [
                    "Comparison",
                    "Subset",
                    "Seed 42 delta",
                    "Seed 43 delta",
                    "Mean delta",
                ],
                comparison_rows,
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path, nargs="+")
    parser.add_argument("--general-prepared", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len({path.name for path in args.prepared}) != len(args.prepared):
        parser.error("Package directory names must be unique")
    batches = {
        path.name: summarize(path, args.general_prepared) for path in args.prepared
    }
    hashes = {batch["benchmark_sha256"] for batch in batches.values()}
    fingerprints = {
        model["sample_fingerprint"]
        for batch in batches.values()
        for model in batch["models"].values()
        if model["complete"]
    }
    if len(hashes) != 1 or len(fingerprints) > 1:
        raise ValueError("Batches do not share the same benchmark and labeled examples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(batches, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "SUMMARY.md").write_text(render(batches))
    for label, batch in batches.items():
        completed = sum(model["complete"] for model in batch["models"].values())
        print(f"{label}: {completed}/{len(batch['models'])} validated full runs")
    print(f"Reports: {args.output_dir}")


if __name__ == "__main__":
    main()
