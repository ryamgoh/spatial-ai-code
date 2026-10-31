"""Reconstruct existing questions and audit selected explanations before selection."""

import argparse
import json
import tempfile
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import run_unique_training_repair as first
from audit_selected_explanation_depth import replay

from spatial.v2.entailment_training_selection import select
from spatial.v2.entailment_unique_study import project_record


def reconstruct(job):
    entry, old = job
    problem = entry.story.problems()[first.historical.KINDS.index(old["query_kind"])]
    if first.query_profile(first.all_scene_problem(problem))["key"] != old["query_key"]:
        raise ValueError("preserved canonical question changed")
    result = first.production.admit(
        entry, problem, first.POLICY, first.TOKENIZER, first.QuestionSolver()
    )
    if result["status"] != "accepted":
        raise ValueError(f"preserved question failed readmission: {old['base_id']}")
    for field in ("base_id", "family_id", "split", "semantics", "measurement"):
        if result[field] != old[field]:
            raise ValueError(f"preserved {field} changed: {old['base_id']}")
    previous = {r["id"]: r for r in old["rows"] if r["contract"] == "unique-entailed"}
    if set(previous) != {r["id"] for r in result["rows"]}:
        raise ValueError("preserved menu inventory changed")
    for row in result["rows"]:
        for field in (
            "menu",
            "answer",
            "depth",
            "semantic_case",
            "escape_reason",
            "substantive_depth_support",
            "decision_depth_support",
        ):
            if row[field] != previous[row["id"]][field]:
                raise ValueError(f"preserved menu {field} changed")
    for field in ("query_key", "stratum", "source_stratum", "controlled_count_value"):
        if field in old:
            result[field] = old[field]
    result = replay(result)
    return result


def replay_saved(record):
    return replay(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; do not restart a running replay")
    if (
        args.tokenizer.name != first.REVISION
        or not args.tokenizer.is_dir()
        or not 1 <= args.workers <= 8
    ):
        parser.error("pinned tokenizer and 1-8 workers required")
    manifest = json.loads(args.manifest.read_text())
    receipt = json.loads(args.receipt.read_text())
    if (
        receipt["status"] != "bounded-admission-complete-not-selected"
        or receipt["manifest_sha256"] != first.digest(args.manifest)
        or receipt["combined_supply"]["conflicts"]
        or receipt["statuses"].get("verification-failed", 0)
    ):
        raise ValueError("completed second batch with passing supply bounds required")
    paths = [Path(p) for p in manifest["candidate_paths"]]
    for path, expected in manifest["input_sha256"].items():
        if first.digest(Path(path)) != expected:
            raise ValueError(f"fixed input changed: {path}")
    policy = manifest["policy"]
    old_records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    selected_ids = set(
        json.loads(
            (
                args.manifest.parent / "unique-only-selection-v1/selection.json"
            ).read_text()
        )["selected_base_ids"]
    )
    evaluation = [
        project_record(r)
        for r in old_records
        if r["split"] != "train" and r["base_id"] in selected_ids
    ]
    if len(evaluation) != 4800:
        raise ValueError("held-out selection inventory incomplete")
    original = first.historical.schedule(first.production.common_policy(policy))
    overrides = {
        e.story.story_id: 2 + first.stable_int(f"{e.story.story_id}/cardinality") % 3
        for e in original
        if e.depth is not None
        and "which" in e.query_kinds
        and first.stable_int(f"{e.story.story_id}/multi-supply") % 2 == 0
    }
    mapped, _ = first.map_fixed_slots(original, policy, selection_matches=overrides)
    entries = {e.story.story_id: e for e in mapped}
    jobs = [(entries[r["story_id"]], r) for r in old_records if r["split"] == "train"]
    prior = json.loads(
        (args.manifest.parent / "UNIQUE-TRAINING-REPAIR-ADMISSION-v1.json").read_text()
    )
    new_full = []
    for info in (prior, receipt):
        path = Path(info["metadata_directory"]) / "full-records.jsonl"
        if first.digest(path) != info["full_records_sha256"]:
            raise ValueError("supplementary full evidence hash mismatch")
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["status"] == "accepted":
                    new_full.append(record)
    spool = Path(tempfile.mkdtemp(prefix="spatial-unique-pool-replay-"))
    with args.output.open("x") as stream:
        json.dump({"status": "running", "metadata_directory": str(spool)}, stream)
    total = len(jobs) + len(new_full)
    print(
        json.dumps(
            {
                "phase": "evidence-replay",
                "existing_questions": total,
                "spool": str(spool),
                "new_seed_attempts": 0,
            }
        ),
        flush=True,
    )
    candidates = []
    with (
        (spool / "full-records.jsonl").open("w") as full,
        (spool / "candidates.jsonl").open("w") as compact,
        get_context("spawn").Pool(
            args.workers,
            initializer=first.initialize,
            initargs=(str(args.tokenizer), policy),
            maxtasksperchild=100,
        ) as pool,
    ):
        for phase, function, items in (
            ("original", reconstruct, jobs),
            ("supplementary", replay_saved, new_full),
        ):
            for record in pool.imap(function, items, chunksize=1):
                full.write(json.dumps(record) + "\n")
                small = first.historical.summary_record(record)
                small.pop("graph", None)
                compact.write(json.dumps(small) + "\n")
                candidates.append(small)
                if len(candidates) % 200 == 0:
                    full.flush()
                    compact.flush()
                    print(
                        json.dumps(
                            {
                                "phase": phase,
                                "replayed": len(candidates),
                                "total": total,
                            }
                        ),
                        flush=True,
                    )
    if len(candidates) != total:
        raise ValueError("replay inventory incomplete")
    print(json.dumps({"phase": "one-menu-selection", "candidates": total}), flush=True)
    selected = select(candidates, evaluation, policy)
    result = {
        "status": "selection-validated-not-sealed"
        if selected
        else "joint-selection-infeasible",
        "replayed_questions": total,
        "metadata_directory": str(spool),
        "candidates_sha256": first.digest(spool / "candidates.jsonl"),
        "full_records_sha256": first.digest(spool / "full-records.jsonl"),
        "source_manifest_sha256": first.digest(args.manifest),
        "source_receipt_sha256": first.digest(args.receipt),
        "evaluation_base_ids": sorted(r["base_id"] for r in evaluation),
        "selected_menus": []
        if selected is None
        else [
            {"base_id": r["base_id"], "row_id": r["rows"][0]["id"]} for r in selected
        ],
        "answer_counts": {}
        if selected is None
        else dict(
            Counter(f"{r['query_kind']}/{r['rows'][0]['answer']}" for r in selected)
        ),
        "sealed": False,
        "training_started": False,
        "evaluation_changed": False,
    }
    if selected:
        ids = {r["rows"][0]["id"] for r in selected}
        with (spool / "selected-metadata.jsonl").open("w") as stream:
            for record in selected:
                stream.write(json.dumps(record) + "\n")
        if len(ids) != 6000:
            raise ValueError("selected row count differs from target")
    for path, expected in manifest["input_sha256"].items():
        if first.digest(Path(path)) != expected:
            raise ValueError(f"input changed during replay: {path}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("selected_menus", "evaluation_base_ids")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
