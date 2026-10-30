"""Compile a bounded, label-blind production schedule and development exclusions."""

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from functools import cache
from pathlib import Path

from spatial.v2.composition_dataset import random_stories
from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_pilot import RECIPES, story_key
from spatial.v2.composition_production import (
    CONTRACTS,
    KINDS,
    ROLES,
    allocate,
    artifact_digest,
    count_quotas,
    depth_family_minima,
    plan,
    preflight,
    quotas,
    schedule,
    source_for,
    source_lineage,
    source_quotas,
    stratum,
)
from spatial.v2.composition_scene_sampler import (
    _core,
    family_id,
    sample_scaffold,
    sample_scene,
)
from spatial.v2.composition_targeted_sampler import PATTERNS, sample_targeted
from spatial.v2.solver import And, Direction, direction_constraint

VERSION = "composition-allocation-v1"
ATTEMPT_MULTIPLIER = 3
POOL_START = 2026110200
POOL_SIZE = 4096
TARGETED_WEIGHTS = {
    "empty": 3,
    "multiple": 3,
    "contingent": 2,
    "singleton": 1,
    "consistent-cycle": 1,
    "inconsistent-cycle": 1,
    "coincidence": 1,
}


def historical_paths(experiment):
    """Original inputs and replay attempts, including rejected development cases."""
    paths = {
        *experiment.glob("*/pilot.json"),
        *experiment.glob("*/attempts.jsonl"),
        *experiment.glob("*/manifest.json"),
        *experiment.glob("*smoke*.json"),
        *experiment.glob("research-analysis/*PREVIEW*.json"),
    }
    return sorted(p for p in paths if not p.parent.name.startswith("production-"))


def development_scenes():
    """Conservative reconstruction of the unsaved sampler fixtures, never fresh families."""
    seen = set()

    def forest(seed, depth, **kwargs):
        spec = ("forest", seed, depth, tuple(sorted(kwargs.items())))
        if spec not in seen:
            seen.add(spec)
            yield sample_scene(seed, depth, **kwargs)

    def targeted(seed, pattern, **kwargs):
        spec = ("targeted", seed, pattern, tuple(sorted(kwargs.items())))
        if spec not in seen:
            seen.add(spec)
            yield sample_targeted(seed, pattern, **kwargs)

    directions = tuple(Direction)
    for seed in range(32):
        yield _core(sample_scaffold(seed), 5, Direction.NORTH)
        for depth in (2, 8, 10):
            for pattern in ("homogeneous", "axis-split", "mixed-chain"):
                yield from forest(
                    seed, depth, direction=Direction.NORTHEAST, proof_pattern=pattern
                )
        yield from targeted(seed, "contingent", span=3, direction=Direction.NORTH)
    for depth in range(1, 11):
        for index, direction in enumerate(directions):
            yield from forest((depth + index) % 32, depth, direction=direction)
    for seed, direction in enumerate(directions):
        yield from forest(seed, 10, direction=direction)
        for depth in (1, 2, 3, 8, 9, 10):
            yield from forest(seed, depth)
        for pattern in PATTERNS:
            yield from targeted(seed, pattern, direction=direction, span=2 + seed % 2)
            yield from targeted(seed, pattern)
    # Include all related fixtures for the two heavily exercised scaffolds.
    for seed in (3, 7):
        for depth in (1, 2, 3, 5, 8, 9, 10):
            for pattern, direction in proof_options(depth):
                for matches in range(5):
                    yield from forest(
                        seed,
                        depth,
                        direction=Direction(direction),
                        proof_pattern=pattern,
                        matches=matches,
                    )
        for direction in directions:
            for depth in (1, 8, 9, 10):
                for variant in (0, 1):
                    yield from forest(seed, depth, direction=direction, variant=variant)
            for pattern in PATTERNS:
                for span in (2, 3):
                    for variant in (0, 1):
                        yield from targeted(
                            seed,
                            pattern,
                            direction=direction,
                            span=span,
                            variant=variant,
                        )
        for matches in (2, 3, 4):
            yield from targeted(seed, "multiple", matches=matches)


def development_inventory(experiment):
    families, keys = set(), set()
    sources = {}

    @cache
    def canonical(serialized):
        objects, atoms = json.loads(serialized)
        return story_key(
            tuple(objects),
            And(tuple(direction_constraint(a, b, Direction(d)) for a, b, d in atoms)),
        )

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for field in ("family_id", "source_family"):
                if isinstance(value.get(field), str):
                    families.add(source_lineage(value[field]))
            if isinstance(value.get("story_key"), str):
                keys.add(value["story_key"])
            if "objects" in value and "premise" in value:
                atoms = value["premise"].get("operands")
                if atoms is None or any(
                    set(a) != {"subject", "reference", "allowed"}
                    or len(a["allowed"]) != 1
                    for a in atoms
                ):
                    raise ValueError(
                        "historical scene is not a flat atomic conjunction"
                    )
                serialized = json.dumps(
                    [
                        sorted(value["objects"]),
                        sorted(
                            (a["subject"], a["reference"], a["allowed"][0])
                            for a in atoms
                        ),
                    ]
                )
                keys.add(canonical(serialized))
            for item in value.values():
                if isinstance(item, (list, dict)):
                    visit(item)

    for path in historical_paths(experiment):
        before = artifact_digest(path)
        with path.open() as stream:
            if path.suffix == ".jsonl":
                for line in stream:
                    visit(json.loads(line))
            else:
                visit(json.load(stream))
        if artifact_digest(path) != before:
            raise ValueError(f"development artifact changed: {path}")
        sources[str(path.relative_to(experiment))] = before

    # These bounded software fixtures were inspected without saved candidate files.
    families.update(f"{recipe}-{i}" for recipe in RECIPES for i in range(5))
    families.update(f"controlled-depth/sample-{i}" for i in range(6))
    families.update(family_id(sample_scaffold(seed)) for seed in range(32))
    fixture_count = 0
    for story in development_scenes():
        fixture_count += 1
        keys.add(story_key(story.objects, story.premise))
    for seed in (1, 2):
        for story in random_stories(seed, 2, 2):
            keys.add(story_key(story.objects, story.premise))
    test_root = experiment.parents[1] / "spatial/v2/tests"
    return {
        "schema": VERSION,
        "status": "assembled-from-development-artifacts",
        "family_ids": sorted(families),
        "story_keys": sorted(keys),
        "sources_sha256": sources,
        "software_fixture_scope": {
            "sampled_scaffold_seeds": [0, 31],
            "reconstructed_sampler_scenes": fixture_count,
            "reconstruction": "Conservative bounded superset of recorded sampler test parameters; existing seeds only.",
            "legacy_recipe_samples": [0, 4],
            "legacy_depth_samples": [0, 5],
            "additional_random_draws": [
                {"seed": seed, "world_count": 2, "atom_count": 2} for seed in (1, 2)
            ],
            "tests_sha256": {
                p.name: artifact_digest(p)
                for p in sorted(test_root.glob("test_composition*.py"))
            },
        },
        "limitation": "Bounded canonical identity plus declared lineages, not arbitrary semantic equivalence.",
    }


def allocate_pool(policy, exclusions, *, start=POOL_START, size=POOL_SIZE):
    """Partition one fixed scaffold pool; never draw replacement seeds."""
    groups = defaultdict(list)
    excluded = set(exclusions["family_ids"])
    for seed in range(start, start + size):
        groups[family_id(sample_scaffold(seed))].append(seed)
    ordered = sorted(
        set(groups) - excluded,
        key=lambda family: (
            stable_int(f"{policy['seed']}/allocation/{family}"),
            family,
        ),
    )
    needs = {
        role: ATTEMPT_MULTIPLIER
        * max(
            n
            for cell, n in quotas(policy).items()
            if cell.startswith(role + "/") and not cell.endswith("/semantic")
        )
        for role in ROLES
    }
    sizes = allocate(len(ordered), needs)
    assigned, offset = {}, 0
    for role, count in sizes.items():
        assigned[role] = ordered[offset : offset + count]
        offset += count
        if count < needs[role]:
            raise ValueError(f"fixed scaffold pool too small for {role}; no refill")
    seeds = {role: [min(groups[f]) for f in fs] for role, fs in assigned.items()}
    return seeds, {
        "seed_start": start,
        "seed_count": size,
        "distinct_families": len(groups),
        "excluded_development_families": sorted(set(groups) & excluded),
        "family_roles": {f: role for role, fs in assigned.items() for f in fs},
        "representative_seeds": {f: min(ss) for f, ss in sorted(groups.items())},
        "minimum_pool_sizes": needs,
        "allocated_pool_sizes": sizes,
    }


def proof_options(depth):
    directions = tuple(Direction)
    options = [("homogeneous", d.value) for d in directions]
    if depth > 1:
        options += [("axis-split", d.value) for d in directions]
        options += [("mixed-chain", d.value) for d in directions[1::2]]
    return options


def source_batches(policy, seeds):
    batches = []

    def forest(role, depth, count, kinds, matches, offset):
        grouped = defaultdict(list)
        options = proof_options(depth)
        pool = seeds[role]
        if count > len(pool):
            raise ValueError("fixed role pool exhausted")
        for i in range(count):
            pattern, direction = options[(i + depth + offset) % len(options)]
            grouped[pattern, direction].append(pool[(i + offset) % len(pool)])
        for (pattern, direction), selected in grouped.items():
            batches.append(
                {
                    "generator": "sampled-forest",
                    "split": role,
                    "seed": selected[0],
                    "seeds": selected,
                    "count": len(selected),
                    "query_kinds": list(kinds),
                    "depth": depth,
                    "matches": matches,
                    "proof_pattern": pattern,
                    "direction": direction,
                }
            )

    targets, counts = quotas(policy), count_quotas(policy)
    for cell, n in targets.items():
        role, kind, depth = cell.split("/")
        if kind != "direction" or depth == "semantic":
            continue
        if targets[f"{role}/which/{depth}"] != n:
            raise ValueError(
                "paired schedule requires equal Direction/Selection quotas"
            )
        forest(role, int(depth), ATTEMPT_MULTIPLIER * n, ("direction", "which"), 1, 0)
        offset = 0
        for value in policy["controlled_count_weights"]:
            count = ATTEMPT_MULTIPLIER * counts[f"{role}/count/{depth}/{value}"]
            if count:
                forest(role, int(depth), count, ("count",), int(value), offset)
            offset += count

    sources = source_quotas(policy)
    for role_index, role in enumerate(ROLES[:-1]):
        for source_index, generator in enumerate(("random-world", "random-atoms")):
            count = ATTEMPT_MULTIPLIER * sources[f"{role}/direction/{generator}"]
            batches.append(
                {
                    "generator": generator,
                    "seed": policy["seed"] * 100 + role_index * 10 + source_index,
                    "count": count,
                    "split": role,
                    "query_kinds": list(KINDS),
                }
            )
        count = ATTEMPT_MULTIPLIER * sources[f"{role}/direction/targeted"]
        offset = 0
        for pattern, n in allocate(count, TARGETED_WEIGHTS).items():
            if not n:
                continue
            pool = seeds[role]
            selected = [pool[(offset + i) % len(pool)] for i in range(n)]
            batches.append(
                {
                    "generator": "sampled-targeted",
                    "seed": selected[0],
                    "seeds": selected,
                    "count": n,
                    "split": role,
                    "query_kinds": list(KINDS),
                    "pattern": pattern,
                    "span": 2,
                }
            )
            offset += n
    return batches


def structural_exclusions(entries, exclusions):
    """Quarantine whole related families on collision, never move or refill them."""
    family_roles, key_families = defaultdict(set), defaultdict(set)
    key_roles, keys = defaultdict(set), {}
    blocked = defaultdict(set)
    excluded_families = set(exclusions["family_ids"])
    excluded_keys = set(exclusions["story_keys"])
    for entry in entries:
        story = entry.story
        family = source_lineage(story.source_family)
        key = story_key(story.objects, story.premise)
        keys[story.story_id] = key
        family_roles[family].add(entry.split)
        key_families[key].add(family)
        key_roles[key].add(entry.split)
        if family in excluded_families:
            blocked[family].add("development-family")
        if key in excluded_keys:
            blocked[family].add("development-story")
    for family, roles in family_roles.items():
        if len(roles) > 1:
            blocked[family].add("cross-split-family")
    for key, roles in key_roles.items():
        if len(roles) > 1:
            for family in key_families[key]:
                blocked[family].add("cross-split-story")
    # Propagate through same-split copies too, so no related copy survives.
    changed = True
    while changed:
        changed = False
        for families in key_families.values():
            if families & blocked.keys():
                for family in families - blocked.keys():
                    blocked[family].add("related-to-quarantined-family")
                    changed = True
    removed = [
        {
            "story_id": e.story.story_id,
            "family_id": e.story.source_family,
            "split": e.split,
            "query_kinds": list(e.query_kinds),
            "story_key": keys[e.story.story_id],
            "reasons": sorted(blocked[source_lineage(e.story.source_family)]),
        }
        for e in entries
        if source_lineage(e.story.source_family) in blocked
    ]
    return removed


def apply_exclusions(batches, entries, removed):
    removed_ids = {r["story_id"] for r in removed}
    indices = defaultdict(list)
    for entry in entries:
        batch, index = entry.story.story_id.removeprefix("production-b").split("-s")
        if entry.story.story_id not in removed_ids:
            indices[int(batch)].append(int(index))
    result = deepcopy(batches)
    for i, batch in enumerate(result):
        kept = sorted(indices[i])
        if len(kept) != batch["count"]:
            batch["indices"] = kept
    return result


def substantive_minima(policy):
    """Explicit base floors already implied by the approved controlled cohorts."""
    sources = source_quotas(policy)
    return [
        {
            "match": {
                "split": role,
                "query_kind": kind,
                "contract": contract,
                "menu_role": "primary",
                "substantive_depth_support": True,
            },
            "minimum_bases": sources[f"{role}/{kind}/controlled"],
        }
        for role in ROLES
        for kind in KINDS
        for contract in sorted(CONTRACTS)
    ]


def supply_report(entries, policy):
    counts, sources, families = Counter(), Counter(), defaultdict(set)
    count_values, varied = Counter(), Counter()
    bounds_failures = []
    for entry in entries:
        batch_index = int(
            entry.story.story_id.split("-s")[0].removeprefix("production-b")
        )
        batch = policy["source_batches"][batch_index]
        sizes = {
            "objects": len(entry.story.objects),
            "premises": len(entry.story.premise.operands),
            "candidates": len(entry.story.candidates),
        }
        if any(n > policy["scene_limits"][key] for key, n in sizes.items()):
            bounds_failures.append(entry.story.story_id)
        for kind in entry.query_kinds:
            cell = stratum(entry, kind)
            counts[cell] += 1
            sources[f"{entry.split}/{kind}/{source_for(entry)}"] += 1
            families[cell].add(source_lineage(entry.story.source_family))
            if entry.depth is not None:
                if kind == "count":
                    count_values[f"{cell}/{batch['matches']}"] += 1
                if batch["proof_pattern"] != "homogeneous":
                    varied[cell] += 1
    proof_percent = policy["diversity_checks"]["nonhomogeneous_controlled_percent"]
    proof_targets = {
        cell: (n * proof_percent + 99) // 100
        for cell, n in quotas(policy).items()
        if not cell.endswith("/semantic") and int(cell.rsplit("/", 1)[1]) >= 2
    }
    return {
        "scheduled_bases": sum(counts.values()),
        "scheduled_stories": len(entries),
        "scheduled_by_split": dict(
            Counter(e.split for e in entries for _ in e.query_kinds)
        ),
        "scheduled_by_stratum": dict(sorted(counts.items())),
        "scheduled_by_source": dict(sorted(sources.items())),
        "families_by_stratum": {k: len(v) for k, v in sorted(families.items())},
        "insufficient_base_supply": {
            k: n - counts[k] for k, n in quotas(policy).items() if counts[k] < n
        },
        "insufficient_source_supply": {
            k: n - sources[k]
            for k, n in source_quotas(policy).items()
            if sources[k] < n
        },
        "insufficient_family_supply": {
            k: n - len(families[k])
            for k, n in depth_family_minima(policy).items()
            if len(families[k]) < n
        },
        "requested_count_value_supply": dict(sorted(count_values.items())),
        "insufficient_count_value_supply": {
            k: n - count_values[k]
            for k, n in count_quotas(policy).items()
            if count_values[k] < n
        },
        "nonhomogeneous_supply": dict(sorted(varied.items())),
        "insufficient_nonhomogeneous_supply": {
            k: n - varied[k] for k, n in proof_targets.items() if varied[k] < n
        },
        "scene_bounds_failures": bounds_failures,
        "boundary": "Structural supply only; no solver answers, menus, explanations or token admission.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh output directory under an existing parent")
    policy = json.loads(args.config.read_text())
    if policy["diversity_checks"]["status"] != "approved":
        parser.error("approve diversity settings before compiling the schedule")
    source_paths = [args.config, *sorted(Path(__file__).parent.glob("*.py"))]
    hashes = {str(p.resolve()): artifact_digest(p) for p in source_paths}
    print("Collecting historical development identities...", flush=True)
    exclusions = development_inventory(args.config.parent)
    print("Partitioning the fixed scaffold pool without solver labels...", flush=True)
    seeds, allocation = allocate_pool(policy, exclusions)
    policy["source_batches"] = source_batches(policy, seeds)
    raw = schedule(policy)
    raw_budget = sum(len(e.query_kinds) for e in raw)
    if raw_budget != ATTEMPT_MULTIPLIER * sum(policy["base_targets"].values()):
        raise ValueError("raw attempt schedule does not match the fixed multiplier")
    print(
        f"Checking {len(raw)} scheduled scenes for structural collisions...", flush=True
    )
    removed = structural_exclusions(raw, exclusions)
    policy["source_batches"] = apply_exclusions(policy["source_batches"], raw, removed)
    removed_ids = {r["story_id"] for r in removed}
    entries = tuple(e for e in raw if e.story.story_id not in removed_ids)
    preflight(entries, exclusions)
    policy["attempt_budget"] = sum(len(e.query_kinds) for e in entries)
    policy["coverage_minima"] = substantive_minima(policy)
    policy["status"] = "numeric-approved"
    policy["approval"]["production_freeze"] = False
    report = supply_report(entries, policy)
    report.update(
        schema=VERSION,
        status="allocation-preflight-complete-not-execution-approved",
        raw_attempt_budget=raw_budget,
        structural_excluded_bases=raw_budget - policy["attempt_budget"],
        structural_excluded_stories=len(removed),
        structural_exclusions=removed,
        targeted_attempt_weights=TARGETED_WEIGHTS,
        controlled_attempt_cycle={
            "depth_1": proof_options(1),
            "depth_2_through_10": proof_options(2),
        },
        literal_gold_policy="At least one base per feasible literal string per in-range split/contract, enforced by required_coverage.",
        inventory_counts={
            "files": len(exclusions["sources_sha256"]),
            "families": len(exclusions["family_ids"]),
            "story_keys": len(exclusions["story_keys"]),
        },
    )
    insufficient = any(
        value
        for key, value in report.items()
        if key.startswith("insufficient_") or key == "scene_bounds_failures"
    )
    if insufficient:
        report["status"] = "allocation-shortfalls-no-refill"
    if any(artifact_digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("source changed during allocation")
    args.output.mkdir()

    def write(name, value):
        path = args.output / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        return artifact_digest(path)

    exclusion_hash = write("development-exclusions.json", exclusions)
    policy["development_exclusions"] = {
        "path": "development-exclusions.json",
        "sha256": exclusion_hash,
    }
    report["remaining_execution_gates"] = plan(policy)["blockers"]
    write("config.json", policy)
    write("family-allocation.json", allocation)
    write("allocation-report.json", report)
    write(
        "manifest.json",
        {
            "schema": VERSION,
            "status": report["status"],
            "sources_sha256": hashes,
            "outputs_sha256": {
                p.name: artifact_digest(p) for p in sorted(args.output.iterdir())
            },
            "solver_invocations": 0,
            "training_authorized": False,
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in (
                    "status",
                    "raw_attempt_budget",
                    "scheduled_bases",
                    "scheduled_stories",
                    "structural_excluded_bases",
                    "scheduled_by_split",
                    "inventory_counts",
                    "insufficient_base_supply",
                    "insufficient_source_supply",
                    "insufficient_family_supply",
                    "remaining_execution_gates",
                    "insufficient_count_value_supply",
                    "insufficient_nonhomogeneous_supply",
                    "scene_bounds_failures",
                )
            },
            indent=2,
        )
    )
    if insufficient:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
