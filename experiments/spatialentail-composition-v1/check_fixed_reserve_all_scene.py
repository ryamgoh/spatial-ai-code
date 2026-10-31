"""Necessary-condition audit of the unchanged fixed reserve, never generation."""

import argparse
import json
import time
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import z3
from audit_entailment_candidate import checked_sat, independent_claims

from spatial.v2 import composition_production as historical
from spatial.v2.composition_training import digest
from spatial.v2.entailment_scene_protocol import all_scene_problem


def check_count(job):
    story_id, split, depth, family, target, problem = job
    solver, claims = independent_claims(problem)
    solver.set(timeout=5000)
    result = {
        "base_id": f"{story_id}-count",
        "stratum": f"{split}/count/{depth}",
        "family": family,
        "target": target,
        "entailed_values": [],
        "unresolved_values": [],
    }
    try:
        consistent = checked_sat(solver)
        for value in range(5):
            if not consistent:
                break
            try:
                if not checked_sat(solver, z3.Not(claims[value])):
                    result["entailed_values"].append(value)
                    break
            except ValueError:
                result["unresolved_values"].append(value)
        result.update(
            status="allowed-count-entailed"
            if result["entailed_values"]
            else "unresolved"
            if result["unresolved_values"]
            else "no-allowed-count-entailed",
            consistent=consistent,
        )
    except ValueError as exc:
        # Unknowns do not become labels. Include them optimistically in bounds.
        result.update(
            status="unresolved", error=str(exc), unresolved_values=list(range(5))
        )
    return result


def shortage_table(results, targets):
    supply = Counter(
        f"{r['stratum']}/{value}"
        for r in results
        for value in (
            r["entailed_values"] if r["entailed_values"] else r["unresolved_values"]
        )
    )
    return [
        {
            "cell": key,
            "required": n,
            "optimistic_supply": supply[key],
            "shortfall": n - supply[key],
        }
        for key, n in targets.items()
        if supply[key] < n
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh report path under an existing parent")
    if not 1 <= args.workers <= 8:
        parser.error("audit workers must be in 1-8")
    config_sha = digest(args.config)
    policy = json.loads(args.config.read_text())
    common = {**policy, "schema": historical.SCHEMA}
    exclusion = args.config.parent / policy["development_exclusions"]["path"]
    if digest(exclusion) != policy["development_exclusions"]["sha256"]:
        raise ValueError("fixed exclusion inventory hash changed")
    start = time.perf_counter()
    entries = historical.schedule(common)
    historical.preflight(entries, json.loads(exclusion.read_text()))
    print(
        json.dumps({"scheduled_stories": len(entries), "split_preflight": "passed"}),
        flush=True,
    )
    inventory, jobs, selection_slots = Counter(), [], []
    for entry in entries:
        batch = policy["source_batches"][int(entry.story.story_id.split("-")[1][1:])]
        for kind in entry.query_kinds:
            inventory[f"{entry.split}/{kind}/{historical.source_for(entry)}"] += 1
        if batch["generator"] != "sampled-forest":
            continue
        if "which" in entry.query_kinds:
            selection_slots.append(
                {
                    "story_id": entry.story.story_id,
                    "split": entry.split,
                    "depth": entry.depth,
                    "family": entry.story.source_family,
                    "seed": batch.get(
                        "seeds",
                        list(range(batch["seed"], batch["seed"] + batch["count"])),
                    )[int(entry.story.story_id.rsplit("s", 1)[1])],
                    "matches_parameter": batch.get("matches", 1),
                }
            )
        if "count" in entry.query_kinds:
            problem = all_scene_problem(entry.story.problems()[2])
            jobs.append(
                (
                    entry.story.story_id,
                    entry.split,
                    entry.depth,
                    entry.story.source_family,
                    batch["matches"],
                    problem,
                )
            )
    results = []
    with get_context("spawn").Pool(args.workers, maxtasksperchild=250) as pool:
        for result in pool.imap(check_count, jobs, chunksize=8):
            results.append(result)
            if len(results) % 500 == 0:
                print(
                    json.dumps(
                        {
                            "count_slots_checked": len(results),
                            "total": len(jobs),
                            "statuses": dict(Counter(r["status"] for r in results)),
                        }
                    ),
                    flush=True,
                )
    deficits = shortage_table(results, historical.count_quotas(common))
    families = defaultdict(set)
    for row in selection_slots:
        families[f"{row['split']}/which/{row['depth']}"].add(row["family"])
    target_families = historical.depth_family_minima(common)
    raw_family_deficits = [
        {"stratum": key, "required": n, "raw_families": len(families[key])}
        for key, n in target_families.items()
        if "/which/" in key and len(families[key]) < n
    ]
    if digest(args.config) != config_sha:
        raise ValueError("fixed configuration changed during audit")
    report = {
        "status": "combined-infeasible-for-unchanged-recipes"
        if deficits
        else "necessary-count-gates-not-disproved",
        "scope": "complete frozen Count reserve; necessary conditions only",
        "config_sha256": config_sha,
        "exclusion_sha256": digest(exclusion),
        "software_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("audit_entailment_candidate.py"),
                Path("spatial/v2/composition_production.py"),
                Path("spatial/v2/composition_scene_sampler.py"),
                Path("spatial/v2/entailment_scene_protocol.py"),
            )
        },
        "seconds": time.perf_counter() - start,
        "workers": args.workers,
        "scheduled_stories": len(entries),
        "scheduled_query_slots": sum(inventory.values()),
        "inventory": dict(inventory),
        "split_isolation_preflight": "passed-for-original-family-identities",
        "count_slots_checked": len(results),
        "count_statuses": dict(Counter(r["status"] for r in results)),
        "count_quota_shortfalls": deficits,
        "selection_raw_family_shortfalls": raw_family_deficits,
        "selection_slot_parameters": dict(
            Counter(r["matches_parameter"] for r in selection_slots)
        ),
        "selection_seed_range": [
            min(r["seed"] for r in selection_slots),
            max(r["seed"] for r in selection_slots),
        ],
        "records": results,
        "interpretation": (
            "Supply is an upper bound: no dedup, depth, evidence, token, or other coverage rejections. "
            "Unresolved SMT results count optimistically. Any quota shortage proves no combined retained set "
            "can meet the unchanged Count assignment. A changed construction needs new analysis, not old labels."
        ),
        "new_recipe_retention_tested": False,
        "multi_selection_25_percent_retention_tested": False,
        "milp_run": False,
        "milp_reason": "unnecessary: a mandatory Count gate has insufficient optimistic supply"
        if deficits
        else "full admission has not been reverified",
        "no_new_seeds": True,
        "quota_relaxations": False,
        "replacement_generated": False,
        "sealed": False,
        "training_started": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "count_slots_checked",
                    "count_statuses",
                    "selection_slot_parameters",
                    "selection_seed_range",
                    "seconds",
                )
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "count_cells_with_shortfalls": len(deficits),
                "largest_shortfalls": sorted(deficits, key=lambda r: -r["shortfall"])[
                    :10
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
