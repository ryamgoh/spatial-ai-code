"""One approved fixed supplementary batch; no retry, refill, seal or training."""

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

from audit_entailment_candidate import audit_record

from spatial.v2 import composition_production as historical
from spatial.v2 import entailment_production as production
from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_scene_protocol import all_scene_problem
from spatial.v2.entailment_scene_recipes import map_fixed_slots
from spatial.v2.entailment_unique_study import training_allocation_policy
from spatial.v2.solver import And, Direction, direction_constraint

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SLOTS = (
    ("direction", "random-world", 768),
    ("direction", "random-atoms", 192),
    ("direction", "sampled-targeted", 64),
    ("count", "random-world", 192),
    ("count", "random-atoms", 192),
    ("which", "random-world", 384),
    ("which", "random-atoms", 256),
)
TOKENIZER = POLICY = None


def batches():
    return [
        {
            "generator": source,
            "seed": 2030110101 + i * 10000,
            "count": count,
            "split": "train",
            "query_kinds": [kind],
            **(
                {"pattern": "contingent", "span": 2}
                if source == "sampled-targeted"
                else {}
            ),
        }
        for i, (kind, source, count) in enumerate(SLOTS)
    ]


def entries_for(policy):
    return tuple(
        replace(
            entry,
            story=replace(
                entry.story,
                story_id=entry.story.story_id.replace(
                    "production-", "unique-repair-v1-"
                ),
            ),
        )
        for entry in historical.schedule({**policy, "source_batches": batches()})
    )


def inventory(entries):
    return [
        {
            "story_id": e.story.story_id,
            "query": e.query_kinds[0],
            "family_id": e.story.source_family,
            "scene_key": historical.story_key(e.story.objects, e.story.premise),
        }
        for e in entries
    ]


def input_paths():
    return [
        EXP / "entailment-config-frozen-v1.json",
        EXP / "COMBINED-ALL-SCENE-FEASIBILITY-v4.json",
        EXP / "unique-only-selection-v1/selection.json",
        EXP / "spatialmap-unique-v1/manifest.json",
        EXP / "spatialmap-unique-v1/evidence.jsonl",
        Path(__file__),
        ROOT / "spatial/v2/entailment_dataset.py",
        ROOT / "spatial/v2/entailment_unique_study.py",
        ROOT / "spatial/v2/composition_dataset.py",
        ROOT / "spatial/v2/composition_targeted_sampler.py",
        ROOT / "spatial/v2/entailment_evidence.py",
        EXP / "audit_entailment_candidate.py",
    ]


def prepare(path):
    if path.exists():
        raise ValueError("manifest already exists; never replace the fixed batch")
    frozen = json.loads(input_paths()[0].read_text())
    receipt = json.loads(input_paths()[1].read_text())
    if digest(input_paths()[0]) != receipt["config_sha256"]:
        raise ValueError("original configuration hash mismatch")
    source = Path(receipt["intermediate_metadata_directory"]) / "candidates.jsonl"
    if digest(source) != receipt["candidates_sha256"]:
        raise ValueError("original candidate hash mismatch")
    policy = training_allocation_policy(frozen)
    seeds = {
        s
        for b in frozen["source_batches"]
        for s in b.get("seeds", range(b["seed"], b["seed"] + b["count"]))
    }
    for batch in batches():
        if seeds.intersection(range(batch["seed"], batch["seed"] + batch["count"])):
            raise ValueError("supplementary seeds overlap original inventory")
    original = historical.schedule(production.common_policy(policy))
    overrides = {
        e.story.story_id: 2 + stable_int(f"{e.story.story_id}/cardinality") % 3
        for e in original
        if e.depth is not None
        and "which" in e.query_kinds
        and stable_int(f"{e.story.story_id}/multi-supply") % 2 == 0
    }
    mapped, _ = map_fixed_slots(original, policy, selection_matches=overrides)
    family_roles, scene_roles = defaultdict(set), defaultdict(set)
    for e in (*original, *mapped):
        family_roles[historical.source_lineage(e.story.source_family)].add(e.split)
        scene_roles[historical.story_key(e.story.objects, e.story.premise)].add(e.split)
    # Include the external benchmark without depending on its answer semantics.
    with input_paths()[4].open() as stream:
        for line in stream:
            problem = json.loads(line)["semantics"]["problem"]
            atoms = tuple(
                direction_constraint(
                    a["subject"], a["reference"], Direction(a["allowed"][0])
                )
                for a in problem["premise"]["operands"]
            )
            scene_roles[historical.story_key(tuple(problem["objects"]), And(atoms))].add(
                "external"
            )
    entries = entries_for(policy)
    slots = inventory(entries)
    if len(slots) != 2048 or len({s["story_id"] for s in slots}) != 2048:
        raise ValueError("supplementary cap or identity mismatch")
    for slot in slots:
        roles = family_roles[historical.source_lineage(slot["family_id"])]
        scene_splits = scene_roles[slot["scene_key"]]
        slot["preflight_rejection"] = (
            "held-out-family-overlap"
            if roles - {"train"}
            else "held-out-scene-overlap"
            if scene_splits - {"train"}
            else None
        )
    exclusions = EXP / frozen["development_exclusions"]["path"]
    if digest(exclusions) != frozen["development_exclusions"]["sha256"]:
        raise ValueError("development exclusions changed")
    hashes = {str(p): digest(p) for p in [*input_paths(), source, exclusions]}
    manifest = {
        "version": "unique-training-repair-v1",
        "approved_query_cap": 2048,
        "approval": "user approved both repairs; no automatic refill",
        "batches": batches(),
        "slots": slots,
        "input_sha256": hashes,
        "policy": policy,
        "old_candidate_path": str(source),
        "evaluation_changed": False,
        "sealed": False,
        "training_started": False,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(path),
                "sha256": digest(path),
                "slots": len(slots),
                "preflight_rejections": dict(
                    Counter(s["preflight_rejection"] for s in slots)
                ),
            }
        ),
        flush=True,
    )


def initialize(tokenizer, policy):
    global TOKENIZER, POLICY
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    POLICY = policy


def admit(entry):
    kind = entry.query_kinds[0]
    problem = entry.story.problems()[historical.KINDS.index(kind)]
    try:
        all_scene_problem(problem)
        sizes = {
            "objects": len(entry.story.objects),
            "premises": len(entry.story.premise.operands),
            "candidates": len(entry.story.candidates),
        }
        if any(n > POLICY["scene_limits"][key] for key, n in sizes.items()):
            raise ValueError("scene limits exceeded")
    except ValueError as exc:
        return {
            "base_id": f"{entry.story.story_id}-{kind}",
            "story_id": entry.story.story_id,
            "query_kind": kind,
            "source": historical.source_for(entry),
            "status": "pre-admission-rejected",
            "reason": str(exc),
        }
    record = production.admit(entry, problem, POLICY, TOKENIZER, QuestionSolver())
    record.update(
        stratum=historical.stratum(entry, kind),
        source_stratum=f"train/{kind}/{historical.source_for(entry)}",
    )
    if record["status"] == "accepted":
        try:
            audit_record(record)
            record["independent_audit"] = "passed"
            record["query_key"] = query_profile(all_scene_problem(problem))["key"]
        except (ValueError, RuntimeError) as exc:
            record.update(status="verification-failed", reason=str(exc))
    return record


def run(manifest_path, output, tokenizer, workers):
    if output.exists():
        raise ValueError("receipt exists; no rerun or refill")
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    for path, expected in manifest["input_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"fixed input changed: {path}")
    if manifest["batches"] != batches() or manifest["approved_query_cap"] != 2048:
        raise ValueError("fixed batch changed")
    if tokenizer.name != REVISION or not tokenizer.is_dir() or not 1 <= workers <= 8:
        raise ValueError("pinned tokenizer and 1-8 workers required")
    entries = entries_for(manifest["policy"])
    slots = manifest["slots"]
    if inventory(entries) != [
        {k: v for k, v in s.items() if k != "preflight_rejection"} for s in slots
    ]:
        raise ValueError("manifest replay differs")
    # A failed/interrupted batch still consumes its authorization. Never silently
    # restart it merely because no completion receipt was produced.
    with output.open("x") as marker:
        json.dump({"status": "running", "manifest_sha256": manifest_hash}, marker)
    spool = Path(tempfile.mkdtemp(prefix="spatial-unique-repair-"))
    records = []
    seen = set()
    with Path(manifest["old_candidate_path"]).open() as stream:
        for line in stream:
            seen.add(json.loads(line)["query_key"])
    rejected = {
        s["story_id"]: s["preflight_rejection"]
        for s in slots
        if s["preflight_rejection"]
    }
    jobs = [e for e in entries if e.story.story_id not in rejected]
    for entry in entries:
        if entry.story.story_id in rejected:
            records.append(
                {
                    "base_id": f"{entry.story.story_id}-{entry.query_kinds[0]}",
                    "status": "preflight-rejected",
                    "reason": rejected[entry.story.story_id],
                }
            )
    print(
        json.dumps({"phase": "admission", "jobs": len(jobs), "spool": str(spool)}),
        flush=True,
    )
    with (
        (spool / "full-records.jsonl").open("w") as full,
        (spool / "records.jsonl").open("w") as compact,
        get_context("spawn").Pool(
            workers,
            initializer=initialize,
            initargs=(str(tokenizer), manifest["policy"]),
            maxtasksperchild=100,
        ) as pool,
    ):
        for record in records:
            compact.write(json.dumps(record) + "\n")
        for record in pool.imap(admit, jobs, chunksize=1):
            if record["status"] == "accepted":
                if record["query_key"] in seen:
                    record.update(status="duplicate-query-rejected")
                else:
                    seen.add(record["query_key"])
            full.write(json.dumps(record) + "\n")
            small = historical.summary_record(record)
            small.pop("graph", None)
            compact.write(json.dumps(small) + "\n")
            records.append(small)
            if len(records) % 100 == 0:
                full.flush()
                compact.flush()
                print(
                    json.dumps(
                        {
                            "checked": len(records),
                            "total": 2048,
                            "statuses": dict(Counter(r["status"] for r in records)),
                        }
                    ),
                    flush=True,
                )
    if len(records) != 2048 or digest(manifest_path) != manifest_hash:
        raise ValueError("incomplete or changed inventory")
    for path, expected in manifest["input_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"input changed during run: {path}")
    report = {
        "status": "bounded-admission-complete-not-selected",
        "attempted_slots": len(records),
        "manifest_sha256": manifest_hash,
        "metadata_directory": str(spool),
        "records_sha256": digest(spool / "records.jsonl"),
        "full_records_sha256": digest(spool / "full-records.jsonl"),
        "statuses": dict(Counter(r["status"] for r in records)),
        "sealed": False,
        "training_started": False,
        "evaluation_changed": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


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
