"""Bounded training-source support audit; no held-out tuning, freeze or production run."""

import argparse
import json
import multiprocessing
import os
import time
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from spatial.v2 import composition_production as sources
from spatial.v2.composition_execution import atomic_json
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_contracts import VERSION
from spatial.v2.entailment_dataset import build_attempt

_worker = None


def audit_schedule(policy, *, semantic_scenes=12, targeted_scenes=2):
    """Select source prefixes and controlled cells before solving any question."""
    if semantic_scenes < 1 or targeted_scenes < 1:
        raise ValueError("positive audit prefix sizes are required")
    selected, controlled = [], set()
    for batch_index, original in enumerate(policy["source_batches"]):
        if original["split"] != "train":
            continue
        available = original.get("indices", list(range(original["count"])))
        if not available:
            continue
        depth = original.get("depth")
        kinds = tuple(original["query_kinds"])
        if depth is not None:
            kinds = tuple(
                kind
                for kind in kinds
                if (
                    kind,
                    depth,
                    original["proof_pattern"],
                    original["matches"] if kind == "count" else None,
                )
                not in controlled
            )
            if not kinds:
                continue
            indices = available[:1]
            controlled.update(
                (
                    kind,
                    depth,
                    original["proof_pattern"],
                    original["matches"] if kind == "count" else None,
                )
                for kind in kinds
            )
        else:
            cap = (
                targeted_scenes
                if original["generator"] == "sampled-targeted"
                else semantic_scenes
            )
            indices = available[:cap]
        batch = deepcopy(original)
        # Preserve the original prefix RNG sequence and raw story indices.
        batch["count"] = max(indices) + 1
        batch["indices"] = indices
        if "seeds" in batch:
            batch["seeds"] = batch["seeds"][: batch["count"]]
        batch["query_kinds"] = list(kinds)
        temporary = {**policy, "source_batches": [batch]}
        for entry in sources.schedule(temporary):
            raw_index = entry.story.story_id.removeprefix("production-b0-s")
            story = replace(
                entry.story, story_id=f"production-b{batch_index}-s{raw_index}"
            )
            selected.append(replace(entry, story=story))
    return tuple(selected)


def worker_init(tokenizer, timeout, omission_percent):
    global _worker
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _worker = (
        AutoTokenizer.from_pretrained(tokenizer, local_files_only=True),
        QuestionSolver(timeout_ms=timeout),
        omission_percent,
    )


def worker_admit(task):
    entry, problem = task
    tokenizer, solver, omission_percent = _worker
    solver.clear()
    started = time.perf_counter()
    result = build_attempt(
        entry.story,
        problem,
        entry.depth,
        tokenizer,
        solver=solver,
        omission_percent=omission_percent,
    )
    result["source"] = sources.source_for(entry)
    result["audit_seconds"] = time.perf_counter() - started
    solver.clear()
    return result


def summary(records):
    accepted = [r for r in records if r["status"] == "accepted"]
    return {
        "interface": VERSION,
        "status": "training-source-support-audit-not-production",
        "attempted_questions": len(records),
        "accepted_questions": len(accepted),
        "statuses": dict(Counter(r["status"] for r in records)),
        "source_statuses": dict(
            Counter(f"{r['source']}/{r['status']}" for r in records)
        ),
        "semantic_presence": dict(
            Counter(
                f"{r['source']}/{r['query_kind']}/{r.get('semantics', {}).get('case')}"
                for r in accepted
            )
        ),
        "failures": [
            {
                key: r.get(key)
                for key in (
                    "base_id",
                    "recipe",
                    "source",
                    "query_kind",
                    "requested_depth",
                    "stage",
                    "status",
                    "reason",
                )
            }
            for r in records
            if r["status"] != "accepted"
        ],
        "negative_category_presence": dict(
            Counter(
                f"{r['source']}/{r['query_kind']}/{category}"
                for r in accepted
                for category in set(r["semantics"]["negative_categories"].values())
            )
        ),
        "training_source_only": True,
        "held_out_questions_inspected": 0,
        "production_generation": False,
        "sealed": False,
        "training_authorized": False,
        "boundary": "Fixed software-support audit, not retained coverage or acceptance of the 10,800-question dataset.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--semantic-scenes", type=int, default=12)
    parser.add_argument("--targeted-scenes", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh report path under an existing parent")
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("provide the pinned local tokenizer")
    if not 1 <= args.workers <= 128:
        parser.error("workers must be in 1-128")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and args.workers > int(allocated):
        parser.error("workers exceed the Slurm allocation")
    policy = json.loads(args.config.read_text())
    if policy.get("interface") != VERSION:
        parser.error("the entailment-only draft configuration is required")
    paths = [args.config, *sorted(Path(__file__).parent.glob("*.py"))]
    hashes = {str(p): digest(p) for p in paths}
    entries = audit_schedule(
        policy,
        semantic_scenes=args.semantic_scenes,
        targeted_scenes=args.targeted_scenes,
    )
    jobs = [
        (entry, problem)
        for entry in entries
        for kind, problem in zip(sources.KINDS, entry.story.problems())
        if kind in entry.query_kinds
    ]
    print(
        json.dumps({"audit_questions": len(jobs), "training_sources_only": True}),
        flush=True,
    )
    records = []
    started = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(
        args.workers,
        initializer=worker_init,
        initargs=(
            str(args.tokenizer),
            policy["solver_timeout_ms"],
            policy["paired_omission_percent"],
        ),
    ) as pool:
        for record in pool.imap(worker_admit, jobs):
            records.append(record)
            if len(records) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "completed": len(records),
                            "total": len(jobs),
                            "statuses": dict(Counter(r["status"] for r in records)),
                        }
                    ),
                    flush=True,
                )
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("sources/configuration changed during the support audit")
    report = {
        **summary(records),
        "seconds": time.perf_counter() - started,
        "sources_sha256": hashes,
        "tokenizer_sha256": {
            p.name: digest(p) for p in args.tokenizer.iterdir() if p.is_file()
        },
        "schedule": [
            {
                "story_id": entry.story.story_id,
                "family_id": entry.story.source_family,
                "requested_depth": entry.depth,
                "queries": list(entry.query_kinds),
            }
            for entry in entries
        ],
        "records": records,
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "records",
                    "sources_sha256",
                    "tokenizer_sha256",
                    "schedule",
                    "failures",
                )
            },
            indent=2,
        )
    )
    if report["statuses"].get("verification-failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
