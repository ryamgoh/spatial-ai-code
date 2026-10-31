"""Second explicitly approved fixed batch; first-batch artifacts stay immutable."""

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import run_unique_training_repair as first
from check_unique_training_allocation import analyze

from spatial.v2.entailment_unique_study import project_record

CAP = 1600
SLOTS = (
    ("direction", "random-world", 64),
    ("which", "random-world", 1024),
    ("which", "random-atoms", 512),
)
EXP = Path(__file__).resolve().parent


def batches():
    return [
        {
            "generator": source,
            "seed": 2031110101 + i * 10000,
            "count": count,
            "split": "train",
            "query_kinds": [kind],
        }
        for i, (kind, source, count) in enumerate(SLOTS)
    ]


def entries_for(policy):
    return tuple(
        replace(
            e,
            story=replace(
                e.story,
                story_id=e.story.story_id.replace("production-", "unique-repair-v2-"),
            ),
        )
        for e in first.historical.schedule({**policy, "source_batches": batches()})
    )


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if first.digest(Path(path)) != expected:
            raise ValueError(f"fixed input changed: {path}")


def prepare(path):
    if path.exists():
        raise ValueError("second manifest exists; never overwrite it")
    first_manifest_path = EXP / "UNIQUE-TRAINING-REPAIR-MANIFEST-v1.json"
    first_receipt_path = EXP / "UNIQUE-TRAINING-REPAIR-ADMISSION-v1.json"
    previous = json.loads(first_manifest_path.read_text())
    receipt = json.loads(first_receipt_path.read_text())
    verify_hashes(previous["input_sha256"])
    if (
        receipt["manifest_sha256"] != first.digest(first_manifest_path)
        or receipt["attempted_slots"] != 2048
        or receipt["status"] != "bounded-admission-complete-not-selected"
    ):
        raise ValueError("first batch provenance is incomplete")
    prior_metadata = Path(receipt["metadata_directory"]) / "records.jsonl"
    if first.digest(prior_metadata) != receipt["records_sha256"]:
        raise ValueError("first batch metadata changed")
    policy = previous["policy"]
    existing_seeds = set()
    for b in [*policy["source_batches"], *previous["batches"]]:
        existing_seeds.update(b.get("seeds", range(b["seed"], b["seed"] + b["count"])))
    for batch in batches():
        seeds = set(range(batch["seed"], batch["seed"] + batch["count"]))
        if seeds & existing_seeds:
            raise ValueError("second batch seed overlap")
        existing_seeds.update(seeds)

    print(json.dumps({"phase": "original-inventory-overlap-check"}), flush=True)
    original = first.historical.schedule(first.production.common_policy(policy))
    overrides = {
        e.story.story_id: 2 + first.stable_int(f"{e.story.story_id}/cardinality") % 3
        for e in original
        if e.depth is not None
        and "which" in e.query_kinds
        and first.stable_int(f"{e.story.story_id}/multi-supply") % 2 == 0
    }
    mapped, _ = first.map_fixed_slots(original, policy, selection_matches=overrides)
    families, scenes = defaultdict(set), defaultdict(set)
    for e in (*original, *mapped):
        families[first.historical.source_lineage(e.story.source_family)].add(e.split)
        scenes[first.historical.story_key(e.story.objects, e.story.premise)].add(
            e.split
        )
    for slot in previous["slots"]:
        families[first.historical.source_lineage(slot["family_id"])].add("train")
        scenes[slot["scene_key"]].add(
            "held-out" if slot["preflight_rejection"] else "train"
        )
    with (EXP / "spatialmap-unique-v1/evidence.jsonl").open() as stream:
        for line in stream:
            problem = json.loads(line)["semantics"]["problem"]
            atoms = tuple(
                first.direction_constraint(
                    a["subject"], a["reference"], first.Direction(a["allowed"][0])
                )
                for a in problem["premise"]["operands"]
            )
            scenes[
                first.historical.story_key(tuple(problem["objects"]), first.And(atoms))
            ].add("external")
    print(json.dumps({"phase": "second-batch-overlap-check", "slots": CAP}), flush=True)
    slots = first.inventory(entries_for(policy))
    if len(slots) != CAP or len({s["story_id"] for s in slots}) != CAP:
        raise ValueError("incorrect second batch inventory")
    for slot in slots:
        slot["preflight_rejection"] = (
            "held-out-family-overlap"
            if families[first.historical.source_lineage(slot["family_id"])] - {"train"}
            else "held-out-scene-overlap"
            if scenes[slot["scene_key"]] - {"train"}
            else None
        )
    hashes = {
        **previous["input_sha256"],
        **{
            str(p): first.digest(p)
            for p in (
                first_manifest_path,
                first_receipt_path,
                prior_metadata,
                Path(__file__),
                EXP / "check_unique_training_allocation.py",
            )
        },
    }
    verify_hashes(hashes)
    manifest = {
        "version": "unique-training-repair-v2",
        "approved_query_cap": CAP,
        "approval": "user approved second fixed batch; no automatic refill",
        "batches": batches(),
        "slots": slots,
        "policy": policy,
        "input_sha256": hashes,
        "candidate_paths": [previous["old_candidate_path"], str(prior_metadata)],
        "first_manifest_sha256": first.digest(first_manifest_path),
        "evaluation_changed": False,
        "sealed": False,
        "training_started": False,
    }
    with path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "manifest": str(path),
                "sha256": first.digest(path),
                "slots": CAP,
                "preflight_rejections": dict(
                    Counter(s["preflight_rejection"] for s in slots)
                ),
            }
        ),
        flush=True,
    )


def run(manifest_path, output, tokenizer, workers):
    if output.exists():
        raise ValueError("second receipt exists; no rerun or refill")
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = first.digest(manifest_path)
    verify_hashes(manifest["input_sha256"])
    if manifest["batches"] != batches() or manifest["approved_query_cap"] != CAP:
        raise ValueError("second batch inventory changed")
    if (
        tokenizer.name != first.REVISION
        or not tokenizer.is_dir()
        or not 1 <= workers <= 8
    ):
        raise ValueError("pinned tokenizer and 1-8 workers required")
    entries = entries_for(manifest["policy"])
    if first.inventory(entries) != [
        {k: v for k, v in s.items() if k != "preflight_rejection"}
        for s in manifest["slots"]
    ]:
        raise ValueError("second manifest replay mismatch")
    spool = Path(tempfile.mkdtemp(prefix="spatial-unique-second-batch-"))
    with output.open("x") as marker:
        json.dump(
            {
                "status": "running",
                "manifest_sha256": manifest_hash,
                "metadata_directory": str(spool),
            },
            marker,
        )
    seen, candidates, records = set(), [], []
    for path in manifest["candidate_paths"]:
        with Path(path).open() as stream:
            for line in stream:
                r = json.loads(line)
                if r["status"] == "accepted":
                    if r["query_key"] in seen:
                        raise ValueError("duplicate original candidate key")
                    seen.add(r["query_key"])
                    if r["split"] == "train":
                        candidates.append(project_record(r))
    exclusions = {
        s["story_id"]: s["preflight_rejection"]
        for s in manifest["slots"]
        if s["preflight_rejection"]
    }
    for entry in entries:
        if entry.story.story_id in exclusions:
            records.append(
                {
                    "base_id": f"{entry.story.story_id}-{entry.query_kinds[0]}",
                    "status": "preflight-rejected",
                    "reason": exclusions[entry.story.story_id],
                }
            )
    jobs = [e for e in entries if e.story.story_id not in exclusions]
    print(
        json.dumps({"phase": "admission", "jobs": len(jobs), "spool": str(spool)}),
        flush=True,
    )
    with (
        (spool / "full-records.jsonl").open("w") as full,
        (spool / "records.jsonl").open("w") as compact,
        get_context("spawn").Pool(
            workers,
            initializer=first.initialize,
            initargs=(str(tokenizer), manifest["policy"]),
            maxtasksperchild=100,
        ) as pool,
    ):
        for record in records:
            compact.write(json.dumps(record) + "\n")
        for record in pool.imap(first.admit, jobs, chunksize=1):
            if record["status"] == "accepted":
                if record["query_key"] in seen:
                    record["status"] = "duplicate-query-rejected"
                else:
                    seen.add(record["query_key"])
            full.write(json.dumps(record) + "\n")
            small = first.historical.summary_record(record)
            small.pop("graph", None)
            compact.write(json.dumps(small) + "\n")
            records.append(small)
            if record["status"] == "accepted":
                candidates.append(project_record(small))
            if len(records) % 100 == 0:
                full.flush()
                compact.flush()
                print(
                    json.dumps(
                        {
                            "checked": len(records),
                            "total": CAP,
                            "statuses": dict(Counter(r["status"] for r in records)),
                        }
                    ),
                    flush=True,
                )
    if len(records) != CAP or first.digest(manifest_path) != manifest_hash:
        raise ValueError("second batch incomplete or changed")
    verify_hashes(manifest["input_sha256"])
    supply = analyze(candidates, {**manifest["policy"], "coverage_minima": []})
    supply.pop("coverage_checks")
    report = {
        "status": "bounded-admission-complete-not-selected",
        "attempted_slots": CAP,
        "manifest_sha256": manifest_hash,
        "metadata_directory": str(spool),
        "records_sha256": first.digest(spool / "records.jsonl"),
        "full_records_sha256": first.digest(spool / "full-records.jsonl"),
        "statuses": dict(Counter(r["status"] for r in records)),
        "combined_supply": supply,
        "supply_scope": "necessary semantic/source bounds; no proof-depth or family feasibility claim",
        "new_semantic_supply": [
            {"query": q, "source": s, "case": case, "count": count}
            for (q, s, case), count in sorted(
                Counter(
                    (r["query_kind"], r["source"], r["semantics"]["case"])
                    for r in records
                    if r["status"] == "accepted"
                ).items()
            )
        ],
        "automatic_refill_authorized": False,
        "evaluation_changed": False,
        "sealed": False,
        "training_started": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "statuses": report["statuses"],
                "supply_status": supply["status"],
                "conflicts": supply["conflicts"],
                "new_semantic_supply": report["new_semantic_supply"],
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.manifest)
    else:
        if args.output is None or args.tokenizer is None:
            parser.error("run requires output and tokenizer")
        run(args.manifest, args.output, args.tokenizer, args.workers)


if __name__ == "__main__":
    main()
