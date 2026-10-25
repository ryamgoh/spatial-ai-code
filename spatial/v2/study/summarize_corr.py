"""Validate a Corr result archive and summarize external SINGLE transfer offline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from statistics import mean

import yaml

from spatial.v2.study.analyze_randomized import (
    WRAPPER,
    answer_set,
    archive_digest,
    normalized_footer,
    source_rows,
)
from spatial.v2.study.metrics import paired_macro_difference
from spatial.v2.study.summarize_randomized import METRICS, digest, metric, pct, table

SCHEMA = "spatialentail-randomized-v2-corr-transfer-v1"


def read_member(archive, name):
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f"Expected regular archive file: {name}")
    with stream:
        return stream.read()


def inspect_sample(sample, source, scorer, generation):
    if sample["doc"] != source or sample["target"] != source["oracle_option"]:
        raise ValueError("Saved document/target differs from prepared source")
    if sample["arguments"] != [[source["evaluation_prompt"], generation]]:
        raise ValueError("Saved prompt/generation differs from prepared task")
    if len(sample["filtered_resps"]) != 1:
        raise ValueError("Expected exactly one filtered response")
    response = sample["filtered_resps"][0]
    if sample["resps"] != [[response]]:
        raise ValueError("Raw and filtered responses differ")
    match = WRAPPER.fullmatch(response)
    if match is None:
        raise ValueError("Missing stage-1 response wrapper")
    predicted = answer_set(response)
    if len(predicted) != 1 or not predicted <= set("ABCDE"):
        raise ValueError("Stage 2 violates SINGLE A-E contract")
    replay = scorer(source, [response])
    if any(value != sample[key] for key, value in replay.items()):
        raise ValueError("Scoring replay differs from saved metrics")
    native = answer_set(match[1], bare=True)
    normalized = normalized_footer(match[1])
    return {
        "id": source["id"],
        "gold": source["oracle_option"],
        "query": source["query_kind"],
        "status": "underdetermined"
        if source["oracle_option"] == "E"
        else "determinate",
        "stage1_pred": ",".join(sorted(native)),
        "stage2_pred": next(iter(predicted)),
        **{key: replay[key] for key in METRICS},
        "normalized_stage1_acc": int(normalized == {source["oracle_option"]}),
        "stage1_closed_think": int("</think>" in match[1]),
        "stage1_chars": len(match[1]),
    }


def describe(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty slice")
    determinate = [r for r in rows if r["status"] == "determinate"]
    ambiguous = [r for r in rows if r["status"] == "underdetermined"]
    return {
        "n": len(rows),
        **{key: mean(r[key] for r in rows) for key in METRICS},
        "normalized_stage1_acc": mean(r["normalized_stage1_acc"] for r in rows),
        "stage1_closed_think_rate": mean(r["stage1_closed_think"] for r in rows),
        "e_recall": mean(r["stage2_pred"] == "E" for r in ambiguous)
        if ambiguous
        else None,
        "false_abstention": mean(r["stage2_pred"] == "E" for r in determinate)
        if determinate
        else None,
        "stage2_prediction_counts": dict(
            sorted(Counter(r["stage2_pred"] for r in rows).items())
        ),
        "stage2_rescues": sum(not r["stage1_acc"] and r["strict_acc"] for r in rows),
        "stage2_regressions": sum(
            r["stage1_acc"] and not r["strict_acc"] for r in rows
        ),
    }


def contrasts(models, repetitions=2000):
    pairs = []
    for seed in (42, 43):
        for arm in ("single-natural", "single-answer-only", "multi-natural"):
            pairs.append(("untuned", f"{arm}-seed-{seed}"))
        pairs.extend(
            [
                (f"single-natural-seed-{seed}", f"multi-natural-seed-{seed}"),
                (f"single-answer-only-seed-{seed}", f"single-natural-seed-{seed}"),
            ]
        )
    output = []
    for first, second in pairs:
        a = {r["id"]: r for r in models[first]}
        b = {r["id"]: r for r in models[second]}
        if a.keys() != b.keys():
            raise ValueError("Paired model IDs differ")
        for subset in ("overall", "determinate", "underdetermined"):
            ids = sorted(
                i for i in a if subset == "overall" or a[i]["status"] == subset
            )
            left = {i: bool(a[i]["strict_acc"]) for i in ids}
            right = {i: bool(b[i]["strict_acc"]) for i in ids}
            # One cell yields the example-weighted paired accuracy difference.
            stats = paired_macro_difference(
                {subset: ids}, left, right, repetitions=repetitions
            )
            output.append(
                {
                    "first": first,
                    "second": second,
                    "subset": subset,
                    "n": len(ids),
                    "wins": sum(right[i] and not left[i] for i in ids),
                    "losses": sum(left[i] and not right[i] for i in ids),
                    "difference_pp": stats["difference"] * 100,
                    "interval95_pp": [v * 100 for v in stats["interval95"]],
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": 42,
                }
            )
    return output


def summarize(archive_path, prepared, root, repetitions=2000):
    receipt = json.loads((prepared / "jobs.json").read_text())
    if receipt["schema"] != SCHEMA:
        raise ValueError("Unsupported Corr receipt schema")
    for base, hashes in (
        (prepared, receipt["file_sha256"]),
        (root, receipt["environment_sha256"]),
    ):
        for path, expected in hashes.items():
            if digest(base / path) != expected:
                raise ValueError(f"Frozen file hash mismatch: {path}")
    upstream = prepared.parent / "prepared-eval/jobs.json"
    if digest(upstream) != receipt["source_sft_evaluation_receipt_sha256"]:
        raise ValueError("SFT preparation receipt hash mismatch")
    correction = json.loads((prepared / "correction-summary.json").read_text())
    if correction["outputs"]["corrected"]["sha256"] != receipt["corrected_sha256"]:
        raise ValueError("Correction hash mismatch")
    sources = {}
    for line in (prepared / "data/corr.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] in sources:
            raise ValueError("Duplicate prepared ID")
        expected_status = (
            "underdetermined-oracle-possible"
            if row["oracle_option"] == "E"
            else "exact-match"
        )
        if row["answer_mode"] != "single" or row["audit_status"] != expected_status:
            raise ValueError("Prepared answer contract/status mismatch")
        sources[row["id"]] = row
    (task,) = receipt["tasks"]
    if len(sources) != receipt["total_bases"] or task["count"] != len(sources):
        raise ValueError("Prepared count mismatch")
    if Counter(r["oracle_option"] for r in sources.values()) != receipt["gold_counts"]:
        raise ValueError("Prepared gold counts mismatch")
    # Ignore !function tags as data. Never execute code supplied by an archive.
    task_config = yaml.load(
        (prepared / f"tasks/{task['task']}.yaml").read_text(), Loader=yaml.BaseLoader
    )
    generation = task_config["generation_kwargs"]
    generation = {
        **generation,
        "max_gen_toks": int(generation["max_gen_toks"]),
        "temperature": float(generation["temperature"]),
        "repetition_penalty": float(generation["repetition_penalty"]),
    }
    spec = importlib.util.spec_from_file_location(
        "corr_task_utils", root / "experiments/tasks/utils.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned scoring implementation")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    models, all_rows, hashes, logs = {}, {}, {}, {}
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
        if archive_digest(archive, "jobs.json") != digest(prepared / "jobs.json"):
            raise ValueError("Archive receipt differs from prepared receipt")
        expected_models = {job["name"] for job in receipt["jobs"]}
        present_models = {
            PurePosixPath(m.name).parts[1]
            for m in members
            if m.isfile() and m.name.startswith("results/")
        }
        if present_models != expected_models:
            raise ValueError("Archive model coverage mismatch")
        for job in receipt["jobs"]:
            name = job["name"]
            prefix = f"results/{name}/"
            marker = prefix + "COMPLETED"
            if marker not in names or not archive.getmember(marker).isfile():
                raise ValueError(f"{name}: missing COMPLETED")
            result_path = prefix + "results.json"
            payload = json.loads(read_member(archive, result_path))
            config = yaml.safe_load((root / job["config"]).read_text())
            runtime = payload["config"]
            if runtime.get("limit") is not None:
                raise ValueError(f"{name}: limited run")
            if runtime["model"] != config["model"]:
                raise ValueError(f"{name}: wrong model backend")
            for key, expected in config["model_args"].items():
                if runtime["model_args"].get(key) != expected:
                    raise ValueError(f"{name}: runtime model argument differs: {key}")
            if set(payload["results"]) != {task["task"]}:
                raise ValueError(f"{name}: task coverage mismatch")
            reported = payload["results"][task["task"]]
            if reported["sample_len"] != len(sources):
                raise ValueError(f"{name}: reported count mismatch")
            effective = (
                payload.get("n-samples", {}).get(task["task"], {}).get("effective")
            )
            if effective not in (None, len(sources)):
                raise ValueError(f"{name}: effective count mismatch")
            response_path = prefix + f"responses_{task['task']}.jsonl"
            rows, seen = [], set()
            for sample in source_rows(archive, response_path):
                identity = sample["doc"]["id"]
                if identity in seen or identity not in sources:
                    raise ValueError(f"{name}: duplicate/unknown source ID")
                seen.add(identity)
                rows.append(
                    inspect_sample(
                        sample,
                        sources[identity],
                        utils.process_results_v2_answer,
                        generation,
                    )
                )
            if seen != sources.keys():
                raise ValueError(f"{name}: incomplete source ID coverage")
            overall = describe(rows)
            for key in METRICS:
                if not math.isclose(overall[key], metric(reported, key), abs_tol=1e-10):
                    raise ValueError(f"{name}: aggregate metric mismatch: {key}")
            all_rows[name] = rows
            models[name] = {
                "complete": True,
                "runtime_config": runtime,
                "overall": overall,
                **{
                    f"by_{field}": {
                        value: describe([r for r in rows if r[field] == value])
                        for value in sorted({r[field] for r in rows})
                    }
                    for field in ("query", "status")
                },
                "by_query_status": {
                    f"{query}/{status}": describe(
                        [
                            r
                            for r in rows
                            if r["query"] == query and r["status"] == status
                        ]
                    )
                    for query, status in sorted(
                        {(r["query"], r["status"]) for r in rows}
                    )
                },
            }
        for member in members:
            if member.isfile():
                hashes[member.name] = archive_digest(archive, member.name)
                if member.name.startswith("logs/"):
                    text = read_member(archive, member.name).decode(errors="replace")
                    logs[member.name] = {
                        "bytes": member.size,
                        "traceback_count": text.count(
                            "Traceback (most recent call last)"
                        ),
                        "results_saved": "Results saved to:" in text,
                    }
    seed_stats = {}
    for arm in ("single-natural", "single-answer-only", "multi-natural"):
        values = [models[f"{arm}-seed-{s}"]["overall"]["strict_acc"] for s in (42, 43)]
        seed_stats[arm] = {
            "seeds": [42, 43],
            "values": values,
            "mean": mean(values),
            "min": min(values),
            "max": max(values),
        }
    return {
        "schema": "spatialentail-randomized-v2-corr-summary-v1",
        "archive": str(archive_path),
        "archive_sha256": digest(archive_path),
        "jobs_sha256": digest(prepared / "jobs.json"),
        "corrected_sha256": receipt["corrected_sha256"],
        "total_bases": len(sources),
        "evaluated_responses": sum(len(r) for r in all_rows.values()),
        "always_e_accuracy": receipt["gold_counts"]["E"] / len(sources),
        "member_sha256": hashes,
        "logs": logs,
        "models": models,
        "seed_summary": seed_stats,
        "paired_comparisons": contrasts(all_rows, repetitions),
    }, all_rows


def render(report):
    lines = [
        "# Randomized-v2 Corr Transfer Summary",
        "",
        (
            "External SINGLE-contract evaluation. Accuracy values are percentages. "
            "Stage 2 constrained A-E accuracy remains primary; native extraction is diagnostic."
        ),
        "",
        (
            f"Validated {len(report['models'])} complete runs, {report['total_bases']:,} unique source IDs "
            f"per run, and {report['evaluated_responses']:,} responses. "
            f"Always choosing E scores {pct(report['always_e_accuracy'])}% overall."
        ),
        "",
        (
            "Archive receipt, prepared inputs and local environment source hashes match. "
            "Every saved document, target, prompt and generation setting matches the prepared data; "
            "runtime model arguments match configs. Original scorer replay agrees on every sample, "
            "and aggregated metrics agree with results.json. "
            "This does not independently verify cluster training consumption, adapter weight hashes, "
            "or all installed runtime packages."
        ),
        "",
        "## Scores",
        "",
    ]
    table(
        lines,
        ["Model", "Stage 1", "Stage 2", "Determinate", "E recall", "False abstention"],
        [
            [
                name,
                pct(m["overall"]["stage1_acc"]),
                pct(m["overall"]["strict_acc"]),
                pct(m["by_status"]["determinate"]["strict_acc"]),
                pct(m["overall"]["e_recall"]),
                pct(m["overall"]["false_abstention"]),
            ]
            for name, m in report["models"].items()
        ],
    )
    lines += [
        (
            "E recall is accuracy on gold-E rows; false abstention is the rate of predicting E "
            "on determinate rows. These have different denominators."
        ),
        "",
        "## Seed Summary",
        "",
    ]
    table(
        lines,
        ["Arm", "Seed 42", "Seed 43", "Mean", "Min", "Max"],
        [
            [
                arm,
                *[pct(v) for v in stats["values"]],
                *[pct(stats[k]) for k in ("mean", "min", "max")],
            ]
            for arm, stats in report["seed_summary"].items()
        ],
    )
    for field in ("query", "status", "query_status"):
        lines += [f"## By {field.replace('_', ' ').title()}", ""]
        models = report["models"]
        first = next(iter(models.values()))
        table(
            lines,
            ["Slice", "N", *models],
            [
                [
                    key,
                    str(value["n"]),
                    *[
                        pct(m[f"by_{field}"][key]["strict_acc"])
                        for m in models.values()
                    ],
                ]
                for key, value in first[f"by_{field}"].items()
            ],
        )
    lines += ["## Stage Diagnostics", ""]
    table(
        lines,
        [
            "Model",
            "Stage 1 format",
            "Normalized stage 1",
            "Closed think",
            "Rescues",
            "Regressions",
        ],
        [
            [
                name,
                *[
                    pct(m["overall"][k])
                    for k in (
                        "stage1_format_valid",
                        "normalized_stage1_acc",
                        "stage1_closed_think_rate",
                    )
                ],
                str(m["overall"]["stage2_rescues"]),
                str(m["overall"]["stage2_regressions"]),
            ]
            for name, m in report["models"].items()
        ],
    )
    lines += [
        (
            "Normalized extraction strips markdown emphasis/backticks and a trailing </think> only. "
            "It is a sensitivity diagnostic, not a replacement score. A missing </think> is not "
            "proof of token-limit truncation; generation finish reasons are not present here."
        ),
        "",
        "## Paired Checkpoint Comparisons",
        "",
        (
            "Differences are second minus first. Wins/losses pair the same source IDs. "
            "95% percentile bootstrap intervals resample IDs within each reported subset, with "
            "2,000 repetitions and seed 42. They condition on these checkpoints, treat IDs as "
            "sampling units, and do not capture training-seed or unseen-map uncertainty. "
            "Intervals are descriptive and not multiplicity-adjusted."
        ),
        "",
    ]
    table(
        lines,
        [
            "First",
            "Second",
            "Subset",
            "Wins",
            "Losses",
            "Delta (pp)",
            "95% interval (pp)",
        ],
        [
            [
                r["first"],
                r["second"],
                r["subset"],
                str(r["wins"]),
                str(r["losses"]),
                f"{r['difference_pp']:.2f}",
                " to ".join(f"{v:.2f}" for v in r["interval95_pp"]),
            ]
            for r in report["paired_comparisons"]
        ],
    )
    lines += [
        "## Scope",
        "",
        (
            "Do not merge these scores with internal SpatialEntail scores or legacy 1,329-row Corr. "
            "The two SFT seeds are reported separately; their range is not a seed-level confidence interval. "
            "Gold labels inherit the current qualitative-coordinate correction assumptions; manual "
            "assumption sensitivity remains pending. Raw responses and logs stay in the original archive."
        ),
        "",
        f"Archive SHA-256: `{report['archive_sha256']}`.",
        "",
        f"Receipt SHA-256: `{report['jobs_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace prior analysis: {args.output_dir}")
    report, rows = summarize(
        args.archive, args.prepared, Path(__file__).resolve().parents[3]
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "SUMMARY.md").write_text(render(report))
    with (args.output_dir / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["model", *next(iter(rows.values()))[0]]
        )
        writer.writeheader()
        for name, samples in rows.items():
            writer.writerows({"model": name, **r} for r in samples)
    print(
        f"Validated {len(rows)} models and {report['evaluated_responses']} responses: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
