"""Bounded composition candidate assembly. Never seals or launches training."""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, replace
from hashlib import file_digest
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

from spatial.v2.composition_balance import (
    KINDS,
    build_attempt,
    menu_inventory,
    query_profile,
)
from spatial.v2.composition_contracts import (
    answer_domain,
    encode,
    parse_prompt,
    render_prompt,
)
from spatial.v2.composition_dataset import random_stories
from spatial.v2.composition_depth import measure_depth
from spatial.v2.composition_feasibility import depth_story, stable_int
from spatial.v2.composition_pilot import Story, construct_stories, story_key
from spatial.v2.composition_scene_sampler import VERSION as FOREST_VERSION
from spatial.v2.composition_scene_sampler import sample_scene
from spatial.v2.composition_selection_expansion import make_story
from spatial.v2.composition_targeted_sampler import sample_targeted
from spatial.v2.composition_training import REVISION, digest, messages
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import Direction, SpatialSolverV2, WhichQuery

SCHEMA = "composition-production-v1"
ROLES = ("train", "dev", "in-range-test", "extrapolation-test")
CONTRACTS = {mode.value for mode in AnswerMode}
CARDINALS = {"North", "East", "South", "West"}
SOURCES = ("controlled", "random-world", "random-atoms", "targeted")
CONTRACT_PAIRS = {
    "unique_listed_differ": ("single", "visible-possible"),
    "unique_exhaustive_differ": ("single", "all-possible"),
    "listed_exhaustive_differ": ("visible-possible", "all-possible"),
}


@dataclass(frozen=True)
class Entry:
    story: Story
    split: str
    depth: int | None
    query_kinds: tuple[str, ...]


def allocate(total, weights):
    """Largest-remainder allocation, with input order breaking ties."""
    if type(total) is not int or total < 0 or not weights:
        raise ValueError("expected a nonnegative total and positive integer weights")
    if any(type(w) is not int or w <= 0 for w in weights.values()):
        raise ValueError("weights must be positive integers")
    denominator = sum(weights.values())
    counts = {key: total * w // denominator for key, w in weights.items()}
    order = sorted(weights, key=lambda key: -(total * weights[key] % denominator))
    for key in order[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def quotas(policy):
    if policy["schema"] != SCHEMA or set(policy["base_targets"]) != set(ROLES):
        raise ValueError("unexpected schema or split roles")
    share = policy["depth_cohort_percent"]
    if type(share) is not int or not 0 < share < 100:
        raise ValueError("depth_cohort_percent must be an integer between 1 and 99")
    result = {}
    for role in ROLES:
        total = policy["base_targets"][role]
        if type(total) is not int or total <= 0:
            raise ValueError("base targets must be positive integers")
        controlled = (
            total
            if role == "extrapolation-test"
            else allocate(total, {"depth": share, "semantic": 100 - share})["depth"]
        )
        depths = range(9, 11) if role == "extrapolation-test" else range(1, 9)
        weights = policy["depth_weights"][role]
        if set(weights) != {str(d) for d in depths}:
            raise ValueError(f"incorrect depth range for {role}")
        for kind, count in allocate(controlled, dict.fromkeys(KINDS, 1)).items():
            for depth, n in allocate(count, weights).items():
                result[f"{role}/{kind}/{depth}"] = n
        for kind, n in allocate(total - controlled, dict.fromkeys(KINDS, 1)).items():
            if n:
                result[f"{role}/{kind}/semantic"] = n
    return result


def depth_family_minima(policy):
    """Require an explicit approved floor for every populated depth stratum."""
    minima = policy.get("depth_family_minima")
    if minima is None:
        return None
    targets = {
        key: n
        for key, n in quotas(policy).items()
        if n and not key.endswith("/semantic")
    }
    if not isinstance(minima, dict) or set(minima) != set(targets):
        raise ValueError("depth_family_minima must cover exactly the depth strata")
    if any(
        type(n) is not int or not 1 <= n <= targets[key] for key, n in minima.items()
    ):
        raise ValueError("family minima must be positive integers within base quotas")
    return minima


def source_quotas(policy):
    """Retained-base quotas, not attempt proportions or rendered-view counts."""
    mix = policy.get("source_mix_percent")
    if (
        not isinstance(mix, dict)
        or set(mix) != set(SOURCES)
        or any(type(n) is not int or n <= 0 for n in mix.values())
        or sum(mix.values()) != 100
        or mix["controlled"] != policy["depth_cohort_percent"]
    ):
        raise ValueError("source mix must total 100 and match the controlled share")
    targets = quotas(policy)
    result = {}
    for role in ROLES:
        for kind in KINDS:
            prefix = f"{role}/{kind}/"
            result[prefix + "controlled"] = sum(
                n
                for key, n in targets.items()
                if key.startswith(prefix) and not key.endswith("/semantic")
            )
            if role != "extrapolation-test":
                counts = allocate(
                    targets.get(prefix + "semantic", 0),
                    {source: mix[source] for source in SOURCES[1:]},
                )
                result.update({prefix + source: n for source, n in counts.items()})
    return result


def source_for(entry):
    if entry.depth is not None:
        return "controlled"
    recipe = entry.story.recipe
    return recipe if recipe in ("random-world", "random-atoms") else "targeted"


def count_quotas(policy):
    """Allocate invariant Count values per controlled split/depth stratum."""
    weights = policy.get("controlled_count_weights")
    if (
        not isinstance(weights, dict)
        or len(weights) < 2
        or any(k not in {str(n) for n in range(9)} for k in weights)
    ):
        raise ValueError("controlled_count_weights needs at least two values in 0-8")
    return {
        f"{cell}/{value}": n
        for cell, total in quotas(policy).items()
        if "/count/" in cell and not cell.endswith("/semantic")
        for value, n in allocate(total, weights).items()
    }


def diversity_coverage(policy):
    """Strengthen presence-only gates; never change labels or refill a pool."""
    spec = policy["diversity_checks"]
    percent = spec["contract_difference_semantic_percent"]
    floors = spec["contract_difference_families"]
    if type(percent) is not int or not 1 <= percent <= 100:
        raise ValueError("contract difference percent must be in 1-100")
    if set(floors) != set(ROLES[:-1]) or any(
        type(n) is not int or n <= 0 for n in floors.values()
    ):
        raise ValueError("contract difference family floors must cover in-range roles")
    proof_percent = spec["nonhomogeneous_controlled_percent"]
    if type(proof_percent) is not int or not 1 <= proof_percent <= 100:
        raise ValueError("nonhomogeneous percent must be in 1-100")
    semantic_percent = spec["semantic_minimum_percent"]
    if type(semantic_percent) is not int or not 1 <= semantic_percent <= 100:
        raise ValueError("semantic minimum percent must be in 1-100")
    rules = []
    for cell, total in quotas(policy).items():
        role, kind, depth = cell.split("/")
        if not total:
            continue
        if depth == "semantic":
            cases = ["unique", "underdetermined", "inconsistent"]
            if kind == "which":
                cases += ["no-match", "multiple-certain-match"]
            for case in cases:
                rules.append(
                    {
                        "match": {
                            "split": role,
                            "query_kind": kind,
                            "semantic_case": case,
                        },
                        "minimum_bases": (total * semantic_percent + 99) // 100,
                        "minimum_families": floors[role],
                    }
                )
            for flag in CONTRACT_PAIRS:
                rules.append(
                    {
                        "match": {"split": role, "query_kind": kind, flag: True},
                        "minimum_bases": (total * percent + 99) // 100,
                        "minimum_families": floors[role],
                    }
                )
        elif int(depth) >= 2:
            rules.append(
                {
                    "match": {
                        "split": role,
                        "query_kind": kind,
                        "depth": int(depth),
                        "source": "controlled",
                        "menu_role": "primary",
                        "nonhomogeneous_proof": True,
                    },
                    "minimum_bases": (total * proof_percent + 99) // 100,
                }
            )
    return rules


def annotate_contract_differences(rows):
    """Compare identical-menu golds, not a control against its omission."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["base_id"], row["bundle_id"])].append(row)
    for group in groups.values():
        gold = {r["contract"]: r["answer"] for r in group}
        if (
            len(group) != 3
            or set(gold) != CONTRACTS
            or len({tuple(r["menu"]) for r in group}) != 1
        ):
            raise ValueError(
                "contract differences require three matched contracts per menu"
            )
        flags = {name: gold[a] != gold[b] for name, (a, b) in CONTRACT_PAIRS.items()}
        for row in group:
            row.update(flags, contracts_differ=any(flags.values()))


def omission_percent(policy):
    value = policy["paired_omission_percent"]
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("paired_omission_percent must be between zero and 100")
    return value


def plan(policy):
    targets = quotas(policy)
    sources = source_quotas(policy)
    percent = omission_percent(policy)
    count_targets = count_quotas(policy)
    diversity_rules = diversity_coverage(policy)
    blockers = []
    if policy.get("status") != "frozen":
        blockers.append(
            "Numeric settings are approved; the executable source schedule is not frozen."
            if policy.get("status") == "numeric-approved"
            else "Numeric policy has not been frozen after user approval."
        )
    if not policy.get("source_batches"):
        blockers.append("Source batches and family allocation are not specified.")
    if not policy.get("development_exclusions"):
        blockers.append(
            "A reviewed development-family exclusion inventory is required."
        )
    if not policy.get("coverage_minima"):
        blockers.append("Semantic and literal-gold coverage minima are not specified.")
    if policy["diversity_checks"].get("status") != "approved":
        blockers.append("New Count weights and diversity thresholds await approval.")
    minima = depth_family_minima(policy)
    if minima is None:
        blockers.append(
            "Distinct-family minima for every depth stratum are not specified."
        )
    return {
        "status": "blocked-before-generation" if blockers else "ready-for-preflight",
        "blockers": blockers,
        "base_quotas": targets,
        "source_base_quotas": sources,
        "controlled_count_quotas": count_targets,
        "diversity_coverage_rules": diversity_rules,
        "source_totals_by_split": {
            role: {
                source: sum(
                    n
                    for key, n in sources.items()
                    if key.startswith(role + "/") and key.endswith("/" + source)
                )
                for source in SOURCES
            }
            for role in ROLES
        },
        "primary_views_per_style": {
            role: 3 * n for role, n in policy["base_targets"].items()
        },
        "maximum_views_per_style": {
            role: (6 if percent else 3) * n
            for role, n in policy["base_targets"].items()
        },
        "depth_family_minima": minima,
        "menu_policy": "one primary menu plus at most one matched omission; all three contracts per menu",
        "paired_omission_percent": percent,
        "direction_policy": "all eight directions per split/query; no per-depth direction quota",
        "extrapolation": "controlled depths 9-10 only; no semantic/null-depth cohort",
    }


def required_coverage():
    """Existing study minima; numeric production minima may strengthen them."""
    rules = []
    for role in ROLES[:-1]:
        for contract in sorted(CONTRACTS):
            for kind in KINDS:
                cases = ["unique", "underdetermined", "inconsistent"]
                if kind == "which":
                    cases += ["no-match", "multiple-certain-match"]
                for case in cases:
                    rules.append(
                        {
                            "match": {
                                "split": role,
                                "contract": contract,
                                "query_kind": kind,
                                "semantic_case": case,
                            },
                            "minimum_bases": 1,
                        }
                    )
            golds = (
                list("ABCDEF")
                if contract == "single"
                else [
                    *("".join(c) for n in range(1, 5) for c in combinations("ABCD", n)),
                    "E",
                    "F",
                ]
            )
            for gold in golds:
                rules.append(
                    {
                        "match": {"split": role, "contract": contract, "answer": gold},
                        "minimum_bases": 1,
                    }
                )
            rules.append(
                {
                    "match": {
                        "split": role,
                        "contract": contract,
                        "query_kind": "count",
                        "semantic_case": "unique",
                        "contains_zero": True,
                        "returned_count": 1,
                    },
                    "minimum_bases": 1,
                }
            )
    return rules


def schedule(policy):
    """Reuse scene constructors; seeds never rename a targeted source family."""
    entries = []
    for batch_index, batch in enumerate(policy["source_batches"]):
        generator, seed = batch["generator"], batch["seed"]
        if type(seed) is not int:
            raise ValueError("batch seed must be an integer")
        role, kinds = batch["split"], tuple(batch["query_kinds"])
        if role not in ROLES or not kinds or len(set(kinds)) != len(kinds):
            raise ValueError("invalid role or duplicate/empty query kinds")
        if set(kinds) - set(KINDS):
            raise ValueError("unknown query kind")
        depth = batch.get("depth")
        allowed = range(9, 11) if role == "extrapolation-test" else range(1, 9)
        if depth is not None and (type(depth) is not int or depth not in allowed):
            raise ValueError("requested depth outside split range")
        if role == "extrapolation-test" and depth is None:
            raise ValueError("extrapolation requires a requested depth")
        count = batch["count"]
        if type(count) is not int or count <= 0:
            raise ValueError("batch count must be a positive integer")
        seeds = batch.get("seeds", list(range(seed, seed + count)))
        if "seeds" in batch and generator not in ("sampled-forest", "sampled-targeted"):
            raise ValueError("explicit seeds are only supported for sampled batches")
        if (
            not isinstance(seeds, list)
            or len(seeds) != count
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("batch seeds must be distinct nonnegative integers")
        if generator in ("random-world", "random-atoms"):
            if depth is not None:
                raise ValueError("unconditioned random batches cannot claim a depth")
            stories = [
                replace(s, source_family=f"random/{seed}/{s.source_family}")
                for s in random_stories(
                    seed,
                    world_count=count if generator == "random-world" else 0,
                    atom_count=count if generator == "random-atoms" else 0,
                )
            ]
        elif generator == "sampled-forest":
            if depth is None:
                raise ValueError("sampled forest batches require a requested depth")
            if "count" in kinds and "matches" not in batch:
                raise ValueError("sampled Count batches require explicit matches")
            stories = [
                sample_scene(
                    draw_seed,
                    depth,
                    direction=Direction(batch["direction"])
                    if "direction" in batch
                    else None,
                    variant=batch.get("variant", 0),
                    matches=batch.get("matches", 1),
                    proof_pattern=batch.get("proof_pattern", "homogeneous"),
                )
                for draw_seed in seeds
            ]
        elif generator == "sampled-targeted":
            if depth is not None:
                raise ValueError("sampled targeted batches cannot claim a depth")
            stories = [
                sample_targeted(
                    draw_seed,
                    batch["pattern"],
                    span=batch.get("span", 2),
                    direction=Direction(batch["direction"])
                    if "direction" in batch
                    else None,
                    variant=batch.get("variant", 0),
                    matches=batch.get("matches"),
                )
                for draw_seed in seeds
            ]
        elif generator == "targeted":
            if depth is not None:
                raise ValueError("targeted semantic batches cannot claim a depth")
            stories = [
                s
                for i in range(count)
                for s in construct_stories(seed + i, batch["per_recipe"])
            ]
        elif generator in ("controlled-depth", "unique-selection"):
            if depth is None:
                raise ValueError("controlled batches require a requested depth")
            # Existing sample IDs describe development lineages, not fresh
            # independent families. Do not lift these bounds to fill quotas.
            sample = batch["sample"]
            if type(sample) is not int or sample not in range(
                6 if generator == "controlled-depth" else 2
            ):
                raise ValueError("unsupported controlled source sample")
            stories = [
                depth_story(depth, sample, random.Random(seed + i))
                if generator == "controlled-depth"
                else make_story(
                    batch["pattern"],
                    depth,
                    sample,
                    random.Random(seed + i),
                    direction=Direction(batch["direction"]),
                )
                for i in range(count)
            ]
            if generator == "controlled-depth":
                # The development constructor names depth bands separately.
                # Production must group related short/long versions together.
                stories = [
                    replace(s, source_family=f"controlled-depth/sample-{sample}")
                    for s in stories
                ]
        else:
            raise ValueError(f"unknown generator: {generator}")
        indices = batch.get("indices", list(range(len(stories))))
        if (
            not isinstance(indices, list)
            or any(type(i) is not int or not 0 <= i < len(stories) for i in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("batch indices must be sorted distinct valid indices")
        selected_indices = set(indices)
        for index, story in enumerate(stories):
            if index not in selected_indices:
                continue
            entries.append(
                Entry(
                    replace(story, story_id=f"production-b{batch_index}-s{index}"),
                    role,
                    depth,
                    kinds,
                )
            )
    return tuple(
        sorted(
            entries, key=lambda e: stable_int(f"{policy['seed']}/{e.story.story_id}")
        )
    )


def preflight(entries, exclusions):
    """Check the complete fixed schedule before constructing any query views."""
    families, identities = defaultdict(set), defaultdict(set)
    keys, ids = {}, set()
    excluded_families = {source_lineage(f) for f in exclusions["family_ids"]}
    excluded_keys = set(exclusions["story_keys"])
    for entry in entries:
        story = entry.story
        allowed = range(9, 11) if entry.split == "extrapolation-test" else range(1, 9)
        if entry.split not in ROLES or (
            entry.depth is not None
            and (type(entry.depth) is not int or entry.depth not in allowed)
        ):
            raise ValueError("invalid split or requested depth")
        if entry.split == "extrapolation-test" and entry.depth is None:
            raise ValueError("extrapolation requires a requested depth")
        if not entry.query_kinds or set(entry.query_kinds) - set(KINDS):
            raise ValueError("invalid query kinds")
        if len(set(entry.query_kinds)) != len(entry.query_kinds):
            raise ValueError("duplicate query kinds")
        if len(set(story.distractors)) != len(story.distractors) or any(
            type(i) is not int or not 0 <= i < len(story.premise.operands)
            for i in story.distractors
        ):
            raise ValueError("invalid distractor indices")
        if story.story_id in ids:
            raise ValueError("duplicate scheduled story ID")
        ids.add(story.story_id)
        key = story_key(story.objects, story.premise)
        keys[story.story_id] = key
        lineage = source_lineage(story.source_family)
        families[lineage].add(entry.split)
        identities[key].add(entry.split)
        if entry.split != "train" and (
            lineage in excluded_families or key in excluded_keys
        ):
            raise ValueError(f"development source reused: {story.story_id}")
    if any(len(v) > 1 for v in families.values()):
        raise ValueError("related source family crosses split roles")
    if any(len(v) > 1 for v in identities.values()):
        raise ValueError("canonical story crosses split roles")
    return keys


def source_lineage(family):
    match = re.search(r"depth-(?:in-range|extrapolation)-family-(\d+)$", family)
    return f"controlled-depth/sample-{match[1]}" if match else family


def check_omission_pair(semantics, full, partial):
    """Check single-slot omission, stable shared positions and contract effects."""
    if not semantics.consistent or not semantics.possible <= set(full):
        raise ValueError("omission pair requires a consistent full-coverage control")
    # Encoding independently validates domain membership and distinct menu values.
    before = {mode: encode(semantics, full, mode) for mode in AnswerMode}
    after = {mode: encode(semantics, partial, mode) for mode in AnswerMode}
    changed = [i for i, (a, b) in enumerate(zip(full, partial)) if a != b]
    if len(changed) != 1:
        raise ValueError("omission pair must change exactly one slot")
    index = changed[0]
    if full[index] not in semantics.possible or partial[index] in semantics.possible:
        raise ValueError(
            "omission must replace a possible value with an impossible one"
        )
    if semantics.possible.intersection(partial) != semantics.possible - {full[index]}:
        raise ValueError("omission changed more than one possible value")
    for mode in AnswerMode:
        old, new = before[mode].letters, after[mode].letters
        if mode.value == "all-possible":
            expected = "F"
        elif mode.value == "visible-possible":
            expected = old.replace("ABCD"[index], "") or "F"
        else:
            expected = "F" if old == "ABCD"[index] else old
        if new != expected:
            raise ValueError("omission changed contract gold unexpectedly")


def select_menu(semantics, seed, *, controlled, paired_percent, solver):
    menus = menu_inventory(semantics, seed)
    # Inventory is ordered by increasing displayed-possible cardinality.
    menu = menus[-1]
    if controlled and isinstance(semantics.problem.query, WhichQuery):
        measured = measure_depth(semantics.problem, solver=solver)
        deep_rivals = {
            m["candidate"]
            for m in measured["memberships"]
            if not m["qualifies"] and m["depth"] == measured["depth"]
        }
        if deep_rivals and not deep_rivals.intersection(menu):
            values = list(menu)
            index = next(
                (i for i, v in enumerate(values) if v not in semantics.possible), None
            )
            if index is not None:
                values[index] = min(deep_rivals)
                menu = tuple(values)
    replacements = [
        v
        for v in answer_domain(semantics.problem)
        if v not in semantics.possible and v not in menu
    ]
    if (
        seed % 100 < paired_percent
        and semantics.consistent
        and semantics.possible
        and semantics.possible <= set(menu)
        and replacements
    ):
        rng = random.Random(seed)
        partial = list(menu)
        index = rng.choice([i for i, v in enumerate(menu) if v in semantics.possible])
        partial[index] = rng.choice(replacements)
        check_omission_pair(semantics, menu, tuple(partial))
        return (menu, tuple(partial))
    return (menu,)


def admit(entry, problem, policy, tokenizer, solver):
    paired_percent = omission_percent(policy)

    def menu_selector(semantics, seed):
        menus = select_menu(
            semantics,
            stable_int(f"{policy['seed']}/{seed}"),
            controlled=entry.depth is not None,
            paired_percent=paired_percent,
            solver=solver,
        )
        if len(menus) == 2:
            check_omission_pair(semantics, *menus)
        for menu in menus:
            permuted = tuple(reversed(menu))
            for mode in AnswerMode:
                before = encode(semantics, menu, mode).letters
                after = encode(semantics, permuted, mode).letters
                selected = {menu["ABCD".index(c)] for c in before if c in "ABCD"}
                other = {permuted["ABCD".index(c)] for c in after if c in "ABCD"}
                if selected != other or (before in ("E", "F") and before != after):
                    raise ValueError("menu permutation changed semantic selection")
                if parse_prompt(render_prompt(problem, permuted, mode)) != (
                    problem,
                    permuted,
                    mode,
                ):
                    raise ValueError("permuted prompt roundtrip mismatch")
        return menus

    record = build_attempt(
        entry.story,
        problem,
        entry.depth,
        tokenizer,
        solver,
        menu_selector=menu_selector,
    )
    record["split"] = entry.split
    record["source"] = source_for(entry)
    recipe = entry.story.recipe
    record["proof_pattern"] = (
        "homogeneous"
        if recipe == FOREST_VERSION
        else recipe.removeprefix(FOREST_VERSION + "/")
        if recipe.startswith(FOREST_VERSION + "/")
        else None
    )
    semantics = record.get("semantics", {})
    direction = None
    if semantics.get("consistent"):
        if record["query_kind"] == "direction":
            if len(semantics["possible"]) == 1:
                direction = semantics["possible"][0]
        else:
            direction = next(iter(problem.query.directions)).value
    record["direction"] = direction
    record["direction_group"] = (
        None
        if direction is None
        else "cardinal"
        if direction in CARDINALS
        else "diagonal"
    )
    bundles = defaultdict(list)
    for row in record.get("rows", []):
        menu_role = (
            "primary" if row["bundle_id"] == f"{record['base_id']}-m0" else "omission"
        )
        row.update(
            split=entry.split,
            story_id=entry.story.story_id,
            family_id=entry.story.source_family,
            recipe=entry.story.recipe,
            source=record["source"],
            menu_role=menu_role,
            direction=direction,
            direction_group=record["direction_group"],
            proof_pattern=record["proof_pattern"],
            nonhomogeneous_proof=record["proof_pattern"]
            in ("mixed-chain", "axis-split"),
        )
        bundles[row["bundle_id"]].append(row)
    if record["status"] != "accepted":
        return record
    if len(bundles) not in (1, 2) or any(
        len(rows) != 3
        or {r["contract"] for r in rows} != CONTRACTS
        or len({tuple(r["menu"]) for r in rows}) != 1
        for rows in bundles.values()
    ):
        raise ValueError("admission did not preserve three matched contracts per menu")
    primary = [r for r in record["rows"] if r["menu_role"] == "primary"]
    if len(primary) != 3:
        raise ValueError("admission missing primary control menu")
    annotate_contract_differences(record["rows"])
    record["paired_omission"] = len(bundles) == 2
    depth = record["measurement"]["depth"]
    if entry.split != "extrapolation-test" and depth is not None and depth > 8:
        record.update(status="depth-rejected", reason="incidental depth exceeds eight")
    elif entry.depth is not None and not all(
        row["substantive_depth_support"] for row in primary
    ):
        record.update(
            status="depth-rejected",
            reason="not every primary contract has substantive printed support at requested depth",
        )
    return record


def stratum(entry, kind):
    return (
        f"{entry.split}/{kind}/{entry.depth if entry.depth is not None else 'semantic'}"
    )


def generate(
    entries,
    policy,
    exclusions,
    tokenizer,
    solver,
    *,
    admission=None,
    keys=None,
    apply_quotas=True,
    admission_ids=None,
):
    """Stream every scheduled outcome; never retry, top up, or move a rejected base."""
    keys = preflight(entries, exclusions) if keys is None else keys
    targets, used, seen = quotas(policy), Counter(), set()
    source_targets, source_used = source_quotas(policy), Counter()
    count_targets, count_used = count_quotas(policy), Counter()
    for entry in entries:
        story = entry.story
        for kind, problem in zip(KINDS, story.problems()):
            if kind not in entry.query_kinds:
                continue
            cell = stratum(entry, kind)
            source = source_for(entry)
            source_cell = f"{entry.split}/{kind}/{source}"
            record = {
                "base_id": f"{story.story_id}-{kind}",
                "story_id": story.story_id,
                "family_id": story.source_family,
                "story_key": keys[story.story_id],
                "query_kind": kind,
                "split": entry.split,
                "requested_depth": entry.depth,
                "stratum": cell,
                "source": source,
                "source_stratum": source_cell,
                "problem": tagged_dataclass_to_dict(problem),
            }
            sizes = {
                "objects": len(story.objects),
                "premises": len(story.premise.operands),
                "candidates": len(story.candidates),
            }
            if any(n > policy["scene_limits"][key] for key, n in sizes.items()):
                record.update(
                    status="bounds-rejected", reason="declared scene limit exceeded"
                )
                yield record
                continue
            query_key = query_profile(problem)["key"] if admission_ids is None else None
            duplicate = (
                query_key in seen
                if admission_ids is None
                else record["base_id"] not in admission_ids
            )
            if duplicate:
                record.update(
                    status="duplicate", reason="canonical query already scheduled"
                )
            else:
                if query_key is not None:
                    seen.add(query_key)
                admitted = (
                    admit(entry, problem, policy, tokenizer, solver)
                    if admission is None
                    else admission(entry, problem)
                )
                record.update(admitted)
                if record["status"] == "accepted":
                    count_cell = None
                    if kind == "count" and entry.depth is not None:
                        field = policy.get("answer_set_field", "possible")
                        if field not in ("possible", "entailed"):
                            raise ValueError("unknown Count answer-set field")
                        values = record["semantics"][field]
                        if len(values) != 1 or type(values[0]) is not int:
                            raise ValueError(
                                "controlled Count must have one verified integer value"
                            )
                        record["controlled_count_value"] = values[0]
                        count_cell = f"{cell}/{values[0]}"
                    if not apply_quotas:
                        yield record
                        continue
                    if used[cell] >= targets.get(cell, 0):
                        record["status"] = "quota-excess"
                    elif source_used[source_cell] >= source_targets.get(source_cell, 0):
                        record["status"] = "source-quota-excess"
                    elif count_cell is not None and count_used[
                        count_cell
                    ] >= count_targets.get(count_cell, 0):
                        record["status"] = "count-quota-excess"
                    else:
                        used[cell] += 1
                        source_used[source_cell] += 1
                        if count_cell is not None:
                            count_used[count_cell] += 1
            yield record


def summarize(records, policy):
    accepted = [r for r in records if r["status"] == "accepted"]
    rows = [row for r in accepted for row in r["rows"]]
    # Recompute from matched golds so stale annotations cannot satisfy gates.
    annotate_contract_differences(rows)
    counts = Counter(r["stratum"] for r in accepted)
    missing = {k: n - counts[k] for k, n in quotas(policy).items() if counts[k] < n}
    source_counts = Counter(r["source_stratum"] for r in accepted)
    source_targets = source_quotas(policy)
    source_shortfalls = {
        key: n - source_counts[key]
        for key, n in source_targets.items()
        if source_counts[key] < n
    }
    count_counts = Counter(
        f"{r['stratum']}/{r['controlled_count_value']}"
        for r in accepted
        if "controlled_count_value" in r
    )
    count_shortfalls = {
        key: n - count_counts[key]
        for key, n in count_quotas(policy).items()
        if count_counts[key] < n
    }
    families = defaultdict(set)
    for record in accepted:
        families[record["stratum"]].add(source_lineage(record["family_id"]))
    minima = depth_family_minima(policy)
    family_shortfalls = [
        {"stratum": key, "minimum_families": n, "actual_families": len(families[key])}
        for key, n in (minima or {}).items()
        if len(families[key]) < n
    ]
    coverage_failures = []
    for rule in [
        *required_coverage(),
        *diversity_coverage(policy),
        *(policy["coverage_minima"] or []),
    ]:
        limits = {
            k: rule[k] for k in ("minimum_bases", "minimum_families") if k in rule
        }
        if (
            not rule["match"]
            or not limits
            or any(type(n) is not int or n <= 0 for n in limits.values())
        ):
            raise ValueError(
                "coverage rules need a nonempty match and positive minimum"
            )
        matches = [
            row
            for row in rows
            if all(row.get(k) == v for k, v in rule["match"].items())
        ]
        actual = {
            "minimum_bases": len({r["base_id"] for r in matches}),
            "minimum_families": len({source_lineage(r["family_id"]) for r in matches}),
        }
        if any(actual[k] < n for k, n in limits.items()):
            coverage_failures.append(
                {
                    **rule,
                    "actual_bases": actual["minimum_bases"],
                    "actual_families": actual["minimum_families"],
                }
            )
    tables = {}
    for fields in (
        ("split", "query_kind", "contract", "semantic_case"),
        ("split", "contract", "answer"),
        ("split", "query_kind", "depth"),
        ("split", "contract", "escape_reason"),
        ("split", "query_kind", "coverage"),
        ("split", "query_kind", "source"),
        ("split", "query_kind", "proof_pattern"),
        ("split", "query_kind", "menu_role", "depth", "substantive_depth_support"),
    ):
        tables["/".join(fields)] = dict(
            Counter("/".join(str(row[f]) for f in fields) for row in rows)
        )
    baseline = {}
    for contract in sorted(CONTRACTS):
        train = Counter(
            r["answer"]
            for r in rows
            if r["split"] == "train" and r["contract"] == contract
        )
        prediction = min(train, key=lambda x: (-train[x], x)) if train else None
        baseline[contract] = {
            "prediction_from_train_only": prediction,
            "roles": {
                role: {
                    "total": len(group),
                    "correct": sum(r["answer"] == prediction for r in group),
                }
                for role in ROLES
                if (
                    group := [
                        r
                        for r in rows
                        if r["split"] == role and r["contract"] == contract
                    ]
                )
            },
        }
    return {
        "status": "unsealed-candidate-requires-review",
        "attempts": len(records),
        "statuses": dict(Counter(r["status"] for r in records)),
        "accepted_bases": len(accepted),
        "accepted_stories": len({r["story_id"] for r in accepted}),
        "accepted_families": len({source_lineage(r["family_id"]) for r in accepted}),
        "family_counts_by_stratum": {key: len(families[key]) for key in quotas(policy)},
        "views_per_style": len(rows),
        "base_counts": dict(counts),
        "quota_shortfalls": missing,
        "source_counts_base_denominators": {
            key: source_counts[key] for key in source_targets
        },
        "source_shortfalls": source_shortfalls,
        "controlled_count_counts": dict(count_counts),
        "controlled_count_shortfalls": count_shortfalls,
        "contract_differences_base_denominators": {
            f"{role}/{kind}": {
                "bases": len(
                    {
                        r["base_id"]
                        for r in rows
                        if r["split"] == role and r["query_kind"] == kind
                    }
                ),
                **{
                    flag: len(
                        {
                            r["base_id"]
                            for r in rows
                            if r["split"] == role
                            and r["query_kind"] == kind
                            and r[flag]
                        }
                    )
                    for flag in ("contracts_differ", *CONTRACT_PAIRS)
                },
            }
            for role in ROLES
            for kind in KINDS
        },
        "proof_patterns_base_denominators": dict(
            Counter(
                f"{r['split']}/{r['query_kind']}/{r['measurement']['depth']}/{r.get('proof_pattern')}"
                for r in accepted
            )
        ),
        "family_shortfalls": family_shortfalls,
        "family_minima_configured": minima is not None,
        "coverage_shortfalls": coverage_failures,
        **direction_coverage(accepted),
        "paired_omission_bases": sum(r["paired_omission"] for r in accepted),
        "menu_roles_view_counts": dict(Counter(r["menu_role"] for r in rows)),
        "unresolved_verification_failures": sum(
            r["status"] == "verification-failed" for r in records
        ),
        "tables_view_denominators": tables,
        "baseline": baseline,
        "scene_counts_base_denominators": {
            field: dict(Counter(str(r["graph"][field]) for r in accepted))
            for field in ("objects", "premises", "candidates", "verified_distractors")
        },
        "tokens": {
            style: {
                "min": min(values, default=0),
                "max": max(values, default=0),
                "mean": sum(values) / len(values) if values else 0,
            }
            for style in ("natural", "symbolic")
            if (
                values := [
                    t["target_tokens"]
                    for r in accepted
                    for t in r["tokens"]
                    if t["style"] == style
                ]
            )
            or not accepted
        },
    }


def direction_coverage(records):
    """Presence uses bases, never menu slots or all possible ambiguous answers."""
    cells = defaultdict(list)
    by_depth = Counter()
    groups = Counter()
    for record in records:
        direction = record["direction"]
        if direction is None:
            continue
        key = f"{record['split']}/{record['query_kind']}/{direction}"
        cells[key].append(record)
        by_depth[f"{key}/{record['measurement']['depth']}"] += 1
        groups[
            f"{record['split']}/{record['query_kind']}/{record['direction_group']}"
        ] += 1
    keys = [
        f"{role}/{kind}/{direction.value}"
        for role in ROLES
        for kind in KINDS
        for direction in Direction
    ]
    return {
        "direction_counts_base_denominators": {key: len(cells[key]) for key in keys},
        "direction_family_counts": {
            key: len({source_lineage(r["family_id"]) for r in cells[key]})
            for key in keys
        },
        "direction_group_counts_base_denominators": dict(groups),
        "direction_by_depth_base_denominators": dict(by_depth),
        "direction_shortfalls": [key for key in keys if not cells[key]],
    }


def has_shortfalls(summary):
    return bool(
        summary["quota_shortfalls"]
        or summary["controlled_count_shortfalls"]
        or summary["source_shortfalls"]
        or summary["family_shortfalls"]
        or not summary["family_minima_configured"]
        or summary["coverage_shortfalls"]
        or summary["direction_shortfalls"]
        or summary["unresolved_verification_failures"]
    )


def write_exports(records, output):
    """Gold/references stay separate from every model-input file."""
    with ExitStack() as stack:
        references = stack.enter_context((output / "references.jsonl").open("x"))
        streams = {
            (role, style): stack.enter_context(
                (output / f"{role}-{style}-inputs.jsonl").open("x")
            )
            for role in ROLES
            for style in ("natural", "symbolic")
        }
        for record in records:
            if record["status"] != "accepted":
                continue
            for row in record["rows"]:
                references.write(json.dumps(row) + "\n")
                for style in ("natural", "symbolic"):
                    streams[row["split"], style].write(
                        json.dumps({"id": row["id"], "messages": messages(row, style)})
                        + "\n"
                    )


def summary_record(record):
    """Keep compact statistics in memory; full evidence is streamed to disk."""
    return {
        **{k: v for k, v in record.items() if k not in ("problem", "rows", "evidence")},
        "rows": [
            {
                k: v
                for k, v in row.items()
                if k not in ("prompt", "references", "explanation_outline")
            }
            for row in record.get("rows", [])
        ],
    }


def artifact_digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--plan", action="store_true", help="Show quotas without sampling"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check the scheduled scenes and exclusions without solver admission",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    policy = json.loads(args.config.read_text())
    config_hash = digest(args.config)
    planned = plan({**policy, "status": "frozen"} if args.preflight else policy)
    if args.plan:
        print(json.dumps(planned, indent=2))
        return
    if planned["blockers"]:
        parser.error("; ".join(planned["blockers"]))
    if not args.preflight and (
        not args.output or args.output.exists() or not args.output.parent.is_dir()
    ):
        parser.error("provide a fresh output directory under an existing parent")
    if not args.preflight and (not args.tokenizer or args.tokenizer.name != REVISION):
        parser.error("provide the pinned tokenizer snapshot")
    exclusion_spec = policy["development_exclusions"]
    exclusion_path = args.config.parent / exclusion_spec["path"]
    if digest(exclusion_path) != exclusion_spec["sha256"]:
        parser.error("development exclusion inventory hash mismatch")
    exclusions = json.loads(exclusion_path.read_text())
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "spatial/v2").glob("*.py")), args.config, exclusion_path]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    if hashes[str(args.config.resolve())] != config_hash:
        parser.error("configuration changed before scheduling")
    entries = schedule(policy)
    if sum(len(e.query_kinds) for e in entries) != policy["attempt_budget"]:
        parser.error("source batches do not match the frozen base-attempt budget")
    preflight(entries, exclusions)
    if args.preflight:
        if any(digest(Path(p)) != h for p, h in hashes.items()):
            parser.error("inputs changed during preflight")
        print(
            json.dumps(
                {
                    "status": "structural-preflight-passed",
                    "scheduled_stories": len(entries),
                    "scheduled_bases": policy["attempt_budget"],
                    "sources_sha256": hashes,
                    "solver_admission": "not-run",
                    "production_freeze": policy.get("status") == "frozen",
                },
                indent=2,
            )
        )
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    solver = SpatialSolverV2(backend="z3", timeout_ms=policy["solver_timeout_ms"])
    tokenizer_hashes = {
        str(p.resolve()): digest(p) for p in args.tokenizer.iterdir() if p.is_file()
    }
    args.output.mkdir()
    manifest = {
        "status": "running-unsealed-candidate",
        "policy": policy,
        "sources_sha256": hashes,
        "tokenizer_sha256": tokenizer_hashes,
        "versions": {
            p: version(p) for p in ("z3-solver", "transformers", "tokenizers")
        },
        "scheduled_stories": len(entries),
        "attempt_budget": sum(len(e.query_kinds) for e in entries),
        "training_authorized": False,
        "manual_audit": "pending",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output / "schedule.jsonl").open("x") as stream:
        for entry in entries:
            stream.write(
                json.dumps(
                    {
                        "story": tagged_dataclass_to_dict(entry.story),
                        "split": entry.split,
                        "requested_depth": entry.depth,
                        "query_kinds": entry.query_kinds,
                    }
                )
                + "\n"
            )
    records = []
    with (args.output / "attempts.jsonl").open("x") as stream:
        for record in generate(entries, policy, exclusions, tokenizer, solver):
            records.append(summary_record(record))
            stream.write(json.dumps(record) + "\n")
            stream.flush()
    if any(digest(Path(p)) != h for p, h in {**hashes, **tokenizer_hashes}.items()):
        raise ValueError("source, configuration or tokenizer changed during generation")
    summary = summarize(records, policy)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "attempts.jsonl").open() as stream:
        write_exports((json.loads(line) for line in stream), args.output)
    manifest.update(
        status="completed-unsealed-candidate",
        output_sha256={
            p.name: artifact_digest(p)
            for p in args.output.iterdir()
            if p != manifest_path
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if has_shortfalls(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
