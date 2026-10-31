"""Check repaired fixed recipes; never exports or seals a replacement dataset."""

import argparse
import json
import time
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import z3
from audit_entailment_candidate import (
    audit_record,
    checked_sat,
    independent_claims,
    replay_paths,
)

from spatial.v2 import composition_production as historical
from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_dataset import (
    build_attempt,
    measure_depth,
    uniqueness_failure_depth,
    verify_transformations,
)
from spatial.v2.entailment_evidence import build_evidence
from spatial.v2.entailment_names import rename_structure, story_name_map
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import VERSION as PROMPT_VERSION
from spatial.v2.entailment_scene_protocol import all_scene_problem
from spatial.v2.entailment_scene_recipes import (
    VERSION,
    controlled_scene,
    map_fixed_slots,
)
from spatial.v2.solver import Direction

TOKENIZER = None


def initialize(tokenizer_path):
    global TOKENIZER
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def selection_balance_conflicts(policy):
    """Necessary arithmetic only: multi-entailment Unique answers cannot be G1."""
    conflicts = []
    for role in historical.ROLES:
        total = historical.allocate(
            policy["base_targets"][role], dict.fromkeys(historical.KINDS, 1)
        )["which"]
        proposed_multi = total // 4
        for rule in policy["coverage_minima"]:
            match = rule.get("match", {})
            if (
                match.get("split") == role
                and match.get("query_kind") == "which"
                and match.get("contract") == "unique-entailed"
                and match.get("menu_role") == "primary"
                and match.get("substantive_depth_support") is True
                and rule.get("minimum_bases", 0) > total - proposed_multi
            ):
                conflicts.append(
                    {
                        "split": role,
                        "selection_bases": total,
                        "proposed_multi_bases": proposed_multi,
                        "maximum_unique_substantive_bases": total - proposed_multi,
                        "unchanged_unique_substantive_minimum": rule["minimum_bases"],
                    }
                )
    return conflicts


def mapping_preflight(entries, exclusions):
    """Quarantine every overlapping slot; never move it to another split."""
    keys, groups, families, rejected = {}, defaultdict(list), defaultdict(set), {}
    excluded_families = {
        historical.source_lineage(family) for family in exclusions["family_ids"]
    }
    excluded_keys = set(exclusions["story_keys"])
    for entry in entries:
        story = entry.story
        if story.story_id in keys:
            raise ValueError("duplicate mapped story identity")
        key = historical.story_key(story.objects, story.premise)
        keys[story.story_id] = key
        groups[key].append(entry)
        lineage = historical.source_lineage(story.source_family)
        families[lineage].add(entry.split)
        if entry.split != "train" and (
            lineage in excluded_families or key in excluded_keys
        ):
            rejected[story.story_id] = "development-exclusion-overlap"
    if any(len(roles) > 1 for roles in families.values()):
        raise ValueError("related source family crosses splits")
    for group in groups.values():
        if len({entry.split for entry in group}) > 1:
            for entry in group:
                rejected[entry.story.story_id] = "canonical-scene-cross-split-overlap"
    eligible = [entry for entry in entries if entry.story.story_id not in rejected]
    if any(
        len({entry.split for entry in group if entry.story.story_id not in rejected})
        > 1
        for group in groups.values()
    ):
        raise ValueError("eligible mapped scenes still cross splits")
    return keys, rejected, tuple(eligible)


def inspect(job):
    entry, slot, kind, full = job
    story = entry.story
    names = story_name_map(story.objects, story.story_id)
    named = rename_structure(story, names)
    problem = all_scene_problem(named.problems()[historical.KINDS.index(kind)])
    result = {
        "base_id": f"{story.story_id}-{kind}",
        "split": entry.split,
        "depth": entry.depth,
        "stratum": f"{entry.split}/{kind}/{entry.depth}",
        "family": story.source_family,
        "kind": kind,
        "matches": slot["matches"],
        "seed": slot["seed"],
        "direction": slot["direction"],
        "objects": len(story.objects),
        "premises": len(story.premise.operands),
        "full_render_audit": full,
    }
    try:
        limits = slot["scene_limits"]
        sizes = {
            "objects": len(story.objects),
            "premises": len(story.premise.operands),
            "candidates": len(story.candidates),
        }
        if any(n > limits[key] for key, n in sizes.items()):
            raise ValueError("unchanged scene limits exceeded")
        solver = QuestionSolver()
        if full:
            record = build_attempt(
                named,
                problem,
                entry.depth,
                TOKENIZER,
                solver=solver,
                omission_percent=100,
                prompt_version=PROMPT_VERSION,
            )
            if record["status"] != "accepted":
                raise ValueError(f"{record['status']}: {record.get('reason')}")
            evidence = load_evidence(record)
            audit_record(record)
            result["max_target_tokens"] = {
                style: max(
                    t["target_tokens"] for t in record["tokens"] if t["style"] == style
                )
                for style in ("natural", "symbolic")
            }
            result["primary_answers"] = {
                row["contract"]: row["answer"]
                for row in record["rows"]
                if row["menu_role"] == "primary"
            }
        else:
            evidence = build_evidence(problem, solver=solver)
            verify_transformations(
                named, problem, evidence.semantics, solver, scene_scope=True
            )
            if measure_depth(problem, solver=solver)["depth"] != entry.depth:
                raise ValueError("requested depth mismatch")
        values = evidence.semantics.entailed
        if not evidence.semantics.consistent:
            raise ValueError("repair introduced inconsistency")
        if kind == "count":
            if values != frozenset((slot["matches"],)):
                raise ValueError("all-scene Count differs from fixed slot target")
            independent, claims = independent_claims(problem)
            independent.set(timeout=5000)
            if not checked_sat(independent) or checked_sat(
                independent, z3.Not(claims[slot["matches"]])
            ):
                raise ValueError("independent Count entailment check failed")
            support = evidence.positive_support[slot["matches"]]
            for member in support["memberships"]:
                replay_paths(
                    problem,
                    member["paths"],
                    member["candidate"],
                    problem.query.reference,
                    slot["direction"],
                    member["qualifies"],
                )
            result["entailed_count_value"] = slot["matches"]
        else:
            if len(values) != slot["matches"]:
                raise ValueError("all-scene Selection cardinality differs")
            if (
                slot["matches"] > 1
                and uniqueness_failure_depth(evidence) != entry.depth
            ):
                raise ValueError("Unique has a shallow two-member certificate")
            result["entailed_selection_cardinality"] = len(values)
        result.update(status="passed", query_key=query_profile(problem)["key"])
    except (ValueError, RuntimeError) as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh receipt path under an existing directory")
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("provide the pinned local tokenizer")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1-8")
    start = time.perf_counter()
    config_hash = digest(args.config)
    policy = json.loads(args.config.read_text())
    common = {**policy, "schema": historical.SCHEMA}
    exclusions_path = args.config.parent / policy["development_exclusions"]["path"]
    exclusions = json.loads(exclusions_path.read_text())
    if digest(exclusions_path) != policy["development_exclusions"]["sha256"]:
        raise ValueError("frozen exclusion inventory changed")
    report = {
        "recipe_version": VERSION,
        "config_sha256": config_hash,
        "exclusion_sha256": digest(exclusions_path),
        "selection_25_percent_arithmetic_conflicts": selection_balance_conflicts(
            policy
        ),
        "new_seeds": False,
        "quota_relaxations": False,
        "production_allocation_changed": False,
        "replacement_generated": False,
        "sealed": False,
        "training_started": False,
        "full_retention_tested": False,
        "noncontrolled_supply_reverified": False,
        "milp_run": False,
    }
    entries = historical.schedule(common)
    historical.preflight(entries, exclusions)
    print(
        json.dumps({"original_preflight": "passed", "stories": len(entries)}),
        flush=True,
    )
    try:
        mapped, manifest = map_fixed_slots(entries, policy)
        keys, overlap_rejections, eligible = mapping_preflight(mapped, exclusions)
        report["mapping_preflight"] = "passed-after-explicit-overlap-rejections"
        report["overlap_rejected_slots"] = overlap_rejections
        report["overlap_rejection_counts"] = dict(Counter(overlap_rejections.values()))
    except ValueError as exc:
        report.update(
            status="fixed-recipe-mapping-blocked",
            mapping_preflight="failed",
            mapping_error=str(exc),
            seconds=time.perf_counter() - start,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    slots = {
        r["story_id"]: {**r, "scene_limits": policy["scene_limits"]} for r in manifest
    }
    jobs, render_cells, selection_cells, results = [], set(), set(), []
    for entry in mapped:
        slot = slots.get(entry.story.story_id)
        if slot is None:
            continue
        if entry.story.story_id in overlap_rejections:
            if "count" in entry.query_kinds:
                results.append(
                    {
                        "base_id": f"{entry.story.story_id}-count",
                        "split": entry.split,
                        "depth": entry.depth,
                        "stratum": f"{entry.split}/count/{entry.depth}",
                        "family": entry.story.source_family,
                        "kind": "count",
                        "matches": slot["matches"],
                        "seed": slot["seed"],
                        "direction": slot["direction"],
                        "status": "overlap-rejected",
                        "reason": overlap_rejections[entry.story.story_id],
                    }
                )
            continue
        if "count" in entry.query_kinds:
            cell = (entry.split, entry.depth, slot["direction"], slot["matches"])
            full = cell not in render_cells
            render_cells.add(cell)
            jobs.append((entry, slot, "count", full))
        if "which" in entry.query_kinds:
            cell = (entry.split, entry.depth, slot["direction"])
            if cell in selection_cells:
                continue
            selection_cells.add(cell)
            for matches in range(1, 5):
                # Alternative constructions are probes, not four new fixed bases.
                story = controlled_scene(
                    slot["seed"],
                    entry.depth,
                    Direction(slot["direction"]),
                    matches,
                    story_id=f"{entry.story.story_id}-probe-m{matches}",
                    source_family=entry.story.source_family,
                )
                jobs.append(
                    (
                        historical.Entry(story, entry.split, entry.depth, ("which",)),
                        {**slot, "matches": matches},
                        "which",
                        True,
                    )
                )
    print(
        json.dumps(
            {
                "mapping_preflight": report["mapping_preflight"],
                "overlap_rejected_slots": len(overlap_rejections),
                "eligible_stories": len(eligible),
                "fixed_count_slots": sum(j[2] == "count" for j in jobs),
                "selection_probes": sum(j[2] == "which" for j in jobs),
            }
        ),
        flush=True,
    )
    checked_jobs = 0
    with get_context("spawn").Pool(
        args.workers,
        initializer=initialize,
        initargs=(str(args.tokenizer),),
        maxtasksperchild=250,
    ) as pool:
        for result in pool.imap(inspect, jobs, chunksize=4):
            results.append(result)
            checked_jobs += 1
            if checked_jobs % 250 == 0:
                print(
                    json.dumps(
                        {
                            "checked": checked_jobs,
                            "total": len(jobs),
                            "statuses": dict(Counter(r["status"] for r in results)),
                        }
                    ),
                    flush=True,
                )
    seen, retained_count = set(), []
    for result in results:
        if result["status"] != "passed" or result["kind"] != "count":
            continue
        if result["query_key"] not in seen:
            seen.add(result["query_key"])
            retained_count.append(result)
    supply = Counter(
        f"{r['stratum']}/{r['entailed_count_value']}" for r in retained_count
    )
    deficits = [
        {
            "cell": cell,
            "required": n,
            "verified_unique_supply": supply[cell],
            "shortfall": n - supply[cell],
        }
        for cell, n in historical.count_quotas(common).items()
        if supply[cell] < n
    ]
    families = defaultdict(set)
    for result in retained_count:
        families[result["stratum"]].add(result["family"])
    family_deficits = [
        {"cell": cell, "required": n, "verified_families": len(families[cell])}
        for cell, n in historical.depth_family_minima(common).items()
        if "/count/" in cell and len(families[cell]) < n
    ]
    if digest(args.config) != config_hash:
        raise ValueError("frozen configuration changed during audit")
    if digest(exclusions_path) != report["exclusion_sha256"]:
        raise ValueError("frozen exclusion inventory changed during audit")
    report.update(
        status="repaired-controlled-check-complete-not-full-retention",
        seconds=time.perf_counter() - start,
        mapping_manifest=manifest,
        mapped_story_keys=keys,
        records=results,
        checked=len(results),
        statuses=dict(Counter(r["status"] for r in results)),
        verified_unique_count_bases=len(retained_count),
        count_quota_shortfalls=deficits,
        count_family_shortfalls=family_deficits,
        selection_probe_count=sum(r["kind"] == "which" for r in results),
        max_target_tokens={
            style: max(
                (r.get("max_target_tokens", {}).get(style, 0) for r in results),
                default=0,
            )
            for style in ("natural", "symbolic")
        },
        software_sha256={
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("spatial/v2/entailment_scene_recipes.py"),
                Path("spatial/v2/entailment_dataset.py"),
                Path(__file__).with_name("audit_entailment_candidate.py"),
            )
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "records",
                    "mapping_manifest",
                    "mapped_story_keys",
                    "overlap_rejected_slots",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
