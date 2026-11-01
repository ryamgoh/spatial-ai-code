"""Reconstruct the unchanged fixed internal benchmark from tracked fingerprints."""

import argparse
import json
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import run_all_controlled_prototype as tokenizer_loader
from audit_entailment_candidate import audit_record

from spatial.v2 import composition_production as historical
from spatial.v2 import entailment_production as production
from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_scene_protocol import SHORT_VERSION, all_scene_problem
from spatial.v2.entailment_scene_recipes import map_fixed_slots
from spatial.v2.entailment_thinking_training import (
    EXPERIMENT,
    digest,
    require,
    write_json,
)
from spatial.v2.entailment_unique_study import training_allocation_policy

POLICY = None


def fixed_jobs():
    policy = training_allocation_policy(
        json.loads((EXPERIMENT / "entailment-config-frozen-v1.json").read_text())
    )
    inventory = json.loads((EXPERIMENT / "generation-exclusions-v1.json").read_text())
    ids, keys = set(inventory["base_ids"]), set(inventory["query_keys"])
    entries = [
        e
        for e in historical.schedule(production.common_policy(policy))
        if any(f"{e.story.story_id}-{k}" in ids for k in e.query_kinds)
    ]
    overrides = {
        e.story.story_id: 2 + stable_int(f"{e.story.story_id}/cardinality") % 3
        for e in entries
        if e.depth is not None
        and "which" in e.query_kinds
        and stable_int(f"{e.story.story_id}/multi-supply") % 2 == 0
    }
    mapped, _ = map_fixed_slots(entries, policy, selection_matches=overrides)
    jobs = [
        (e, k)
        for e in mapped
        for k in e.query_kinds
        if f"{e.story.story_id}-{k}" in ids
    ]
    actual = {
        query_profile(all_scene_problem(e.story.problems()[historical.KINDS.index(k)]))[
            "key"
        ]
        for e, k in jobs
    }
    require(
        len(jobs) == 4800 and actual == keys,
        "reconstructed benchmark differs from fixed inventory",
    )
    policy["prompt_version"] = SHORT_VERSION
    return policy, jobs


def initialize(snapshot, policy):
    global POLICY
    POLICY = policy
    tokenizer_loader.initialize(snapshot)


def admit(job):
    entry, kind = job
    problem = entry.story.problems()[historical.KINDS.index(kind)]
    record = production.admit(
        entry, problem, POLICY, tokenizer_loader.TOKENIZER, QuestionSolver()
    )
    require(
        record["status"] == "accepted",
        f"evaluation admission failed: {record.get('reason')}",
    )
    audit_record(record)
    return record


def positive_workers(value):
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be at least 1")
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENT / "evaluation-internal"
    )
    parser.add_argument("--workers", type=positive_workers, default=4)
    args = parser.parse_args()
    require(not args.output.exists(), "evaluation output already exists")
    policy, jobs = fixed_jobs()
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        "Qwen/Qwen3.5-4B",
        revision=tokenizer_loader.REVISION,
        allow_patterns=["tokenizer*", "chat_template.jinja", "config.json"],
    )
    args.output.mkdir()
    print(f"Auditing 4800 internal evaluation bases with {args.workers} workers", flush=True)
    counts = Counter()
    with (
        get_context("spawn").Pool(
            args.workers, initializer=initialize, initargs=(snapshot, policy)
        ) as pool,
        (args.output / "references.jsonl").open("x") as references,
        (args.output / "evidence.jsonl").open("x") as evidence,
    ):
        for index, record in enumerate(pool.imap(admit, jobs), 1):
            counts[record["split"]] += 1
            evidence.write(json.dumps(record) + "\n")
            for row in record["rows"]:
                references.write(json.dumps(row) + "\n")
            if index % 100 == 0:
                print(f"Audited internal evaluation bases: {index}/4800", flush=True)
    require(
        counts == {"dev": 1200, "in-range-test": 2400, "extrapolation-test": 1200},
        "split counts differ",
    )
    write_json(
        args.output / "manifest.json",
        {
            "status": "fixed-internal-evaluation-materialized",
            "base_counts": dict(counts),
            "inventory_sha256": digest(EXPERIMENT / "generation-exclusions-v1.json"),
            "payload_sha256": {
                n: digest(args.output / n)
                for n in ("references.jsonl", "evidence.jsonl")
            },
        },
    )


if __name__ == "__main__":
    main()
