"""Re-audit only changed depth-one scenes against the complete previous inventory."""

import argparse
import json
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

from check_repaired_fixed_reserve import initialize, inspect

from spatial.v2 import composition_production as historical
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_scene_recipes import VERSION, controlled_scene
from spatial.v2.solver import Direction


def overlap_rejections(keys, policy, exclusions):
    groups, rejected = defaultdict(list), {}
    for story_id, key in keys.items():
        batch = policy["source_batches"][int(story_id.split("-")[1][1:])]
        groups[key].append((story_id, batch["split"]))
        if batch["split"] != "train" and key in set(exclusions["story_keys"]):
            rejected[story_id] = "development-exclusion-overlap"
    for group in groups.values():
        if len({role for _, role in group}) > 1:
            rejected.update(
                (story_id, "canonical-scene-cross-split-overlap")
                for story_id, _ in group
            )
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh receipt under an existing directory")
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("provide pinned tokenizer")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1-8")
    start = perf_counter()
    policy = json.loads(args.config.read_text())
    previous = json.loads(args.previous.read_text())
    config_hash = digest(args.config)
    if (
        previous["recipe_version"] != "all-scene-forest-v2"
        or previous["config_sha256"] != config_hash
        or previous["status"] != "repaired-controlled-check-complete-not-full-retention"
    ):
        raise ValueError("previous completed guarded audit does not match fixed inputs")
    exclusion_path = args.config.parent / policy["development_exclusions"]["path"]
    exclusion_hash = digest(exclusion_path)
    if exclusion_hash != previous["exclusion_sha256"]:
        raise ValueError("exclusions differ from previous audit")
    exclusions = json.loads(exclusion_path.read_text())
    keys = dict(previous["mapped_story_keys"])
    manifest = [r for r in previous["mapping_manifest"] if r["depth"] == 1]
    entries = {}
    for slot in manifest:
        story = controlled_scene(
            slot["seed"],
            1,
            Direction(slot["direction"]),
            slot["matches"],
            story_id=slot["story_id"],
            source_family=slot["family"],
        )
        entries[slot["story_id"]] = historical.Entry(
            story, slot["split"], 1, tuple(slot["query_kinds"])
        )
        keys[slot["story_id"]] = historical.story_key(story.objects, story.premise)
    rejected = overlap_rejections(keys, policy, exclusions)
    jobs, results, selection_cells = [], [], set()
    for slot in manifest:
        entry = entries[slot["story_id"]]
        if slot["story_id"] in rejected:
            if "count" in entry.query_kinds:
                results.append(
                    {
                        "base_id": f"{slot['story_id']}-count",
                        "kind": "count",
                        "stratum": f"{entry.split}/count/1",
                        "family": slot["family"],
                        "status": "overlap-rejected",
                        "reason": rejected[slot["story_id"]],
                    }
                )
            continue
        slot = {**slot, "scene_limits": policy["scene_limits"]}
        if "count" in entry.query_kinds:
            jobs.append((entry, slot, "count", True))
        if "which" in entry.query_kinds:
            cell = (entry.split, slot["direction"])
            if cell in selection_cells:
                continue
            selection_cells.add(cell)
            for matches in range(1, 5):
                story = controlled_scene(
                    slot["seed"],
                    1,
                    Direction(slot["direction"]),
                    matches,
                    story_id=f"{slot['story_id']}-probe-m{matches}",
                    source_family=slot["family"],
                )
                jobs.append(
                    (
                        historical.Entry(story, entry.split, 1, ("which",)),
                        {**slot, "matches": matches},
                        "which",
                        True,
                    )
                )
    print(
        json.dumps(
            {
                "depth_one_slots": len(manifest),
                "jobs": len(jobs),
                "overlap_rejections": len(rejected),
            }
        ),
        flush=True,
    )
    with get_context("spawn").Pool(
        args.workers,
        initializer=initialize,
        initargs=(str(args.tokenizer),),
        maxtasksperchild=250,
    ) as pool:
        for i, result in enumerate(pool.imap(inspect, jobs, chunksize=4), 1):
            results.append(result)
            if i % 100 == 0:
                print(
                    json.dumps(
                        {
                            "checked": i,
                            "total": len(jobs),
                            "statuses": dict(Counter(r["status"] for r in results)),
                        }
                    ),
                    flush=True,
                )
    seen, verified = set(), []
    for result in results:
        if (
            result["kind"] == "count"
            and result["status"] == "passed"
            and result["query_key"] not in seen
        ):
            seen.add(result["query_key"])
            verified.append(result)
    supply = Counter(f"{r['stratum']}/{r['entailed_count_value']}" for r in verified)
    common = {**policy, "schema": historical.SCHEMA}
    deficits = [
        {
            "cell": cell,
            "required": n,
            "verified_unique_supply": supply[cell],
            "shortfall": n - supply[cell],
        }
        for cell, n in historical.count_quotas(common).items()
        if cell.split("/")[2] == "1" and supply[cell] < n
    ]
    nondepth = set(previous["mapped_story_keys"]) - set(entries)
    changed_overlap = [
        story_id
        for story_id in nondepth
        if rejected.get(story_id) != previous["overlap_rejected_slots"].get(story_id)
    ]
    if digest(args.config) != config_hash or digest(exclusion_path) != exclusion_hash:
        raise ValueError("fixed inputs changed during audit")
    report = {
        "status": "depth-one-repair-audited-not-full-retention",
        "recipe_version": VERSION,
        "seconds": perf_counter() - start,
        "previous_audit": str(args.previous),
        "previous_audit_sha256": digest(args.previous),
        "config_sha256": config_hash,
        "exclusion_sha256": exclusion_hash,
        "changed_scope": "depth-one controlled scene context only",
        "depth_one_fixed_slots": len(manifest),
        "records": results,
        "verified_unique_count_bases": len(verified),
        "statuses": dict(Counter(r["status"] for r in results)),
        "count_quota_shortfalls": deficits,
        "non_depth_one_overlap_status_changes": changed_overlap,
        "higher_depths_reverified": False,
        "full_retention_tested": False,
        "new_seeds": False,
        "quota_relaxations": False,
        "replacement_generated": False,
        "sealed": False,
        "training_started": False,
        "max_target_tokens": {
            style: max(
                (r.get("max_target_tokens", {}).get(style, 0) for r in results),
                default=0,
            )
            for style in ("natural", "symbolic")
        },
        "software_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("check_repaired_fixed_reserve.py"),
                Path("spatial/v2/entailment_scene_recipes.py"),
            )
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
