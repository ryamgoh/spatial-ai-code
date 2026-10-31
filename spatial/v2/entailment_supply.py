"""Optimistic training-reserve bounds for entailment contract differences."""

import argparse
import json
import multiprocessing
import time
from collections import Counter
from pathlib import Path

from spatial.v2 import composition_production as sources
from spatial.v2.composition_execution import atomic_json
from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import digest
from spatial.v2.entailment_contracts import VERSION, Contract, analyze, encode
from spatial.v2.entailment_dataset import menus_for
from spatial.v2.entailment_support_audit import audit_schedule

_worker = None


def worker_init(timeout, omission_percent):
    global _worker
    _worker = QuestionSolver(timeout_ms=timeout), omission_percent


def inspect(entry):
    solver, percent = _worker
    solver.clear()
    problem = entry.story.problems()[1]
    semantics = analyze(problem, solver=solver)
    base_id = entry.story.story_id + "-which"
    menus = menus_for(semantics, stable_int(base_id), omission_percent=percent)
    differences = {
        name: any(
            encode(semantics, menu, a).letters != encode(semantics, menu, b).letters
            for menu in menus
        )
        for name, a, b in (
            ("unique_listed_differ", Contract.UNIQUE, Contract.LISTED),
            ("unique_exhaustive_differ", Contract.UNIQUE, Contract.EXHAUSTIVE),
            ("listed_exhaustive_differ", Contract.LISTED, Contract.EXHAUSTIVE),
        )
    }
    return {
        "base_id": base_id,
        "family_id": entry.story.source_family,
        "source": sources.source_for(entry),
        "semantic_case": semantics.case,
        "entailed_count": len(semantics.entailed),
        **differences,
    }


def bounds(records, policy):
    common = {**policy, "schema": sources.SCHEMA}
    caps = sources.source_quotas(common)
    result = {}
    for flag in (
        "unique_listed_differ",
        "unique_exhaustive_differ",
        "listed_exhaustive_differ",
    ):
        by_source = Counter(row["source"] for row in records if row[flag])
        maximum = sum(
            min(by_source[source], caps[f"train/which/{source}"])
            for source in ("random-world", "random-atoms", "targeted")
        )
        matching = [
            rule
            for rule in policy["coverage_minima"]
            if rule["match"].get("split") == "train"
            and rule["match"].get("query_kind") == "which"
            and rule["match"].get(flag)
        ]
        required = max((rule["minimum_bases"] for rule in matching), default=0)
        result[flag] = {
            "raw_supply": sum(by_source.values()),
            "raw_supply_by_source": dict(by_source),
            "optimistic_retained_upper_bound": maximum,
            "required_retained_bases": required,
            "provably_insufficient": maximum < required,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh supply report path")
    if not 1 <= args.workers <= 128:
        parser.error("workers must be in 1-128")
    policy = json.loads(args.config.read_text())
    if policy.get("interface") != VERSION:
        parser.error("provide the entailment-only draft")
    for batch in policy["source_batches"]:
        if (
            batch["split"] == "train"
            and batch.get("depth") is not None
            and "which" in batch["query_kinds"]
            and (batch["generator"] != "sampled-forest" or batch.get("matches") != 1)
        ):
            parser.error("the bound requires singleton controlled Selection batches")
    paths = [args.config, *sorted(Path(__file__).parent.glob("*.py"))]
    hashes = {str(path): digest(path) for path in paths}
    maximum_prefix = max(batch["count"] for batch in policy["source_batches"])
    entries = [
        entry
        for entry in audit_schedule(
            policy, semantic_scenes=maximum_prefix, targeted_scenes=maximum_prefix
        )
        if entry.depth is None and "which" in entry.query_kinds
    ]
    print(
        json.dumps({"training_semantic_selection_questions": len(entries)}), flush=True
    )
    started = time.perf_counter()
    records = []
    with multiprocessing.get_context("spawn").Pool(
        args.workers,
        initializer=worker_init,
        initargs=(policy["solver_timeout_ms"], policy["paired_omission_percent"]),
    ) as pool:
        for row in pool.imap(inspect, entries):
            records.append(row)
            if len(records) % 100 == 0:
                print(
                    json.dumps({"completed": len(records), "total": len(entries)}),
                    flush=True,
                )
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("source changed during supply inspection")
    report = {
        "interface": VERSION,
        "status": "training-reserve-supply-bound",
        "seconds": time.perf_counter() - started,
        "training_questions_checked": len(records),
        "held_out_questions_inspected": 0,
        "bounds": bounds(records, policy),
        "case_counts": dict(
            Counter(f"{r['source']}/{r['semantic_case']}" for r in records)
        ),
        "boundary": (
            "Upper bounds ignore deduplication, proof/token failures and other quota "
            "constraints. A bound below a required minimum proves infeasibility; "
            "a sufficient bound does not prove that retained coverage is attainable."
        ),
        "sources_sha256": hashes,
        "records": records,
        "training_authorized": False,
        "sealed": False,
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("records", "sources_sha256")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
