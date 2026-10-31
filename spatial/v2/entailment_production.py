"""Versioned entailment production policy using shared source and quota machinery."""

from collections import Counter, defaultdict
from copy import deepcopy

from spatial.v2 import composition_production as historical
from spatial.v2.composition_scene_sampler import VERSION as FOREST_VERSION
from spatial.v2.entailment_contracts import VERSION, Contract
from spatial.v2.entailment_dataset import build_attempt
from spatial.v2.entailment_names import rename_structure, story_name_map
from spatial.v2.entailment_scene_protocol import (
    PLAIN_VERSIONS,
    SCOPE,
    SHORT_VERSION,
    all_scene_problem,
)
from spatial.v2.entailment_scene_protocol import VERSION as SCENE_VERSION
from spatial.v2.entailment_scene_recipes import (
    PREVIOUS_VERSION as PREVIOUS_FOREST_VERSION,
)
from spatial.v2.entailment_scene_recipes import VERSION as ALL_SCENE_FOREST_VERSION
from spatial.v2.solver import DirectionQuery, WhichQuery

SCHEMA = "entailment-production-v1"
MULTI_SELECTION_DEPTH = "controlled-multi-selection-depth-v1"
PAIRS = {
    "unique_listed_differ": (Contract.UNIQUE.value, Contract.LISTED.value),
    "unique_exhaustive_differ": (Contract.UNIQUE.value, Contract.EXHAUSTIVE.value),
    "listed_exhaustive_differ": (Contract.LISTED.value, Contract.EXHAUSTIVE.value),
}
CONTRACTS = {c.value for c in Contract}


def all_scene_diagnostic_policy(policy):
    """Apply approved Unique decision-depth coverage in every split, not a freeze."""
    common_policy(policy)
    revised = deepcopy(policy)
    revised.update(
        status="diagnostic-not-frozen",
        prompt_version=SCENE_VERSION,
        scope_policy=SCOPE,
        controlled_multi_selection=MULTI_SELECTION_DEPTH,
    )
    for rule in revised["coverage_minima"]:
        match = rule["match"]
        if (
            match.get("split") in historical.ROLES
            and match.get("query_kind") == "which"
            and match.get("contract") == Contract.UNIQUE.value
            and match.get("menu_role") == "primary"
            and match.get("substantive_depth_support") is True
        ):
            del match["substantive_depth_support"]
            match["decision_depth_support"] = True
            rule["numeric_origin"] = "approved Unique all-split decision-depth rule"
    return revised


def common_policy(policy):
    """Translate only configuration plumbing, never query semantics or labels."""
    if policy.get("schema") != SCHEMA or policy.get("interface") != VERSION:
        raise ValueError("expected versioned entailment production configuration")
    if policy.get("controlled_multi_selection") not in (None, MULTI_SELECTION_DEPTH):
        raise ValueError("unknown controlled multi-Selection depth policy")
    if (policy.get("prompt_version"), policy.get("scope_policy")) not in (
        (None, None),
        *((version, SCOPE) for version in PLAIN_VERSIONS),
    ):
        raise ValueError("unknown or mismatched production scene protocol")
    if policy.get("prompt_version") == SHORT_VERSION:
        from spatial.v2.entailment_unique_study import STUDY_SCOPE

        if policy.get("study_scope") != STUDY_SCOPE or policy.get(
            "active_contracts"
        ) != [Contract.UNIQUE.value]:
            raise ValueError("short questions require an explicit Unique-only study")
    return {
        **policy,
        "schema": historical.SCHEMA,
        "answer_set_field": "entailed",
    }


def plan(policy):
    shared = common_policy(policy)
    targets = historical.quotas(shared)
    historical.source_quotas(shared)
    historical.count_quotas(shared)
    historical.omission_percent(shared)
    if policy.get("retention", "first-accepted") not in (
        "first-accepted",
        "coverage-constrained-v1",
    ):
        raise ValueError("unknown entailment retention policy")
    families = historical.depth_family_minima(shared)
    blockers = []
    if policy.get("prompt_version") in PLAIN_VERSIONS:
        blockers.append(
            "All-scene allocation and depth admission require review; "
            "the scoped frozen allocation cannot authorize replacement generation."
        )
    if policy.get("status") != "frozen":
        blockers.append("Entailment execution configuration is not frozen.")
    if policy.get("coverage_review", {}).get("status") != "approved":
        blockers.append("Revised coverage matrix has not been approved.")
    if not policy.get("source_batches") or not policy.get("development_exclusions"):
        blockers.append("Source schedule and hashed exclusion inventory are required.")
    if families is None:
        blockers.append("Depth-family floors are required.")
    rules = policy.get("coverage_minima")
    if not rules:
        blockers.append("Explicit entailment coverage minima are required.")
    for rule in rules or []:
        match = rule.get("match", {})
        limits = [rule[k] for k in ("minimum_bases", "minimum_families") if k in rule]
        if not match or not limits or any(type(n) is not int or n <= 0 for n in limits):
            raise ValueError("invalid entailment coverage minimum")
        if "contract" in match and match["contract"] not in CONTRACTS:
            raise ValueError("historical contract in entailment coverage")
        if match.get("listed_exhaustive_differ") and match.get("query_kind") in (
            "direction",
            "count",
        ):
            raise ValueError(
                "impossible Direction/Count Listed/Exhaustive difference gate"
            )
    return {
        "status": "blocked-before-generation" if blockers else "ready-for-preflight",
        "blockers": blockers,
        "interface": VERSION,
        "base_quotas": targets,
        "depth_family_minima": families,
    }


def schedule(policy):
    return historical.schedule(common_policy(policy))


def annotate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("interface") != VERSION:
            raise ValueError("historical row cannot enter entailment production")
        groups[(row["base_id"], row["bundle_id"])].append(row)
    for group in groups.values():
        gold = {r["contract"]: r["answer"] for r in group}
        if any(r.get("study_scope") is not None for r in group):
            from spatial.v2.entailment_unique_study import validate_row

            if len(group) != 1:
                raise ValueError("Unique-only requires one view per menu")
            validate_row(group[0])
            continue
        if (
            len(group) != 3
            or set(gold) != CONTRACTS
            or len({tuple(r["menu"]) for r in group}) != 1
        ):
            raise ValueError("entailment requires three matched contracts per menu")
        flags = {key: gold[a] != gold[b] for key, (a, b) in PAIRS.items()}
        if group[0]["query_kind"] != "which" and flags["listed_exhaustive_differ"]:
            raise ValueError("Direction/Count Listed and Exhaustive disagree")
        for row in group:
            row.update(flags)


def admit(entry, problem, policy, tokenizer, solver):
    common_policy(policy)
    names = None
    if policy.get("prompt_version") in PLAIN_VERSIONS:
        names = story_name_map(problem.objects, entry.story.story_id)
        problem = all_scene_problem(rename_structure(problem, names))
    record = build_attempt(
        entry.story,
        problem,
        entry.depth,
        tokenizer,
        solver=solver,
        omission_percent=policy["paired_omission_percent"],
        prompt_version=policy.get("prompt_version"),
    )
    if names is not None:
        record["entity_name_map"] = names
    recipe = entry.story.recipe
    pattern = (
        "homogeneous"
        if recipe == FOREST_VERSION
        else recipe.removeprefix(FOREST_VERSION + "/")
        if recipe.startswith(FOREST_VERSION + "/")
        else recipe.removeprefix(ALL_SCENE_FOREST_VERSION + "/")
        if recipe.startswith(ALL_SCENE_FOREST_VERSION + "/")
        else recipe.removeprefix(PREVIOUS_FOREST_VERSION + "/")
        if recipe.startswith(PREVIOUS_FOREST_VERSION + "/")
        else None
    )
    semantics = record.get("semantics", {})
    direction = None
    if semantics.get("consistent"):
        if isinstance(problem.query, DirectionQuery):
            if len(semantics["entailed"]) == 1:
                direction = semantics["entailed"][0]
        else:
            direction = next(iter(problem.query.directions)).value
    record.update(
        split=entry.split,
        source=historical.source_for(entry),
        proof_pattern=pattern,
        direction=direction,
        direction_group=None
        if direction is None
        else "cardinal"
        if direction in historical.CARDINALS
        else "diagonal",
        paired_omission=any(
            r["menu_role"] == "omission" for r in record.get("rows", [])
        ),
    )
    for row in record.get("rows", []):
        row.update(
            split=entry.split,
            source=record["source"],
            proof_pattern=pattern,
            nonhomogeneous_proof=pattern in ("mixed-chain", "axis-split"),
            direction=direction,
            direction_group=record["direction_group"],
        )
    if record["status"] == "accepted":
        annotate(record["rows"])
        depth = record["measurement"]["depth"]
        if entry.split != "extrapolation-test" and depth is not None and depth > 8:
            record.update(
                status="depth-rejected", reason="incidental depth exceeds eight"
            )
        elif entry.depth is not None:
            multi = (
                policy.get("controlled_multi_selection") == MULTI_SELECTION_DEPTH
                and isinstance(problem.query, WhichQuery)
                and 2 <= len(semantics["entailed"]) <= 4
            )
            field = "decision_depth_support" if multi else "substantive_depth_support"
            if not all(
                row[field] for row in record["rows"] if row["menu_role"] == "primary"
            ):
                record.update(
                    status="depth-rejected",
                    reason="primary contracts lack "
                    + ("decision" if multi else "substantive")
                    + " support at requested depth",
                )
    return record


def generate(
    entries,
    policy,
    exclusions,
    tokenizer,
    solver,
    *,
    admission=None,
    keys=None,
    admission_ids=None,
):
    shared = common_policy(policy)
    callback = admission or (
        lambda entry, problem: admit(entry, problem, policy, tokenizer, solver)
    )
    yield from historical.generate(
        entries,
        shared,
        exclusions,
        tokenizer,
        solver,
        admission=callback,
        keys=keys,
        apply_quotas=policy.get("retention") != "coverage-constrained-v1",
        admission_ids=admission_ids,
    )


def make_gate(policy):
    if policy.get("retention") == "coverage-constrained-v1":
        from spatial.v2.entailment_retention import CompletionGate

        return CompletionGate(policy)
    return None


def summarize(records, policy):
    shared = common_policy(policy)
    accepted = [r for r in records if r["status"] == "accepted"]
    rows = [row for record in accepted for row in record["rows"]]
    annotate(rows)
    counts = Counter(r["stratum"] for r in accepted)
    sources = Counter(r["source_stratum"] for r in accepted)
    count_values = Counter(
        f"{r['stratum']}/{r['controlled_count_value']}"
        for r in accepted
        if "controlled_count_value" in r
    )

    def deficits(targets, observed):
        return {
            key: n - observed[key] for key, n in targets.items() if observed[key] < n
        }

    family_counts = defaultdict(set)
    for record in accepted:
        family_counts[record["stratum"]].add(
            historical.source_lineage(record["family_id"])
        )
    minima = historical.depth_family_minima(shared)
    family_shortfalls = [
        {
            "stratum": key,
            "minimum_families": n,
            "actual_families": len(family_counts[key]),
        }
        for key, n in (minima or {}).items()
        if len(family_counts[key]) < n
    ]
    coverage = []
    for rule in policy["coverage_minima"]:
        matches = [
            row
            for row in rows
            if all(row.get(k) == v for k, v in rule["match"].items())
        ]
        actual = {
            "minimum_bases": len({r["base_id"] for r in matches}),
            "minimum_families": len(
                {historical.source_lineage(r["family_id"]) for r in matches}
            ),
        }
        if any(actual[k] < rule[k] for k in actual if k in rule):
            coverage.append(
                {
                    **rule,
                    "actual_bases": actual["minimum_bases"],
                    "actual_families": actual["minimum_families"],
                }
            )
    return {
        "interface": VERSION,
        "status": "unsealed-candidate-requires-review",
        "attempts": len(records),
        "statuses": dict(Counter(r["status"] for r in records)),
        "accepted_bases": len(accepted),
        "accepted_stories": len({r["story_id"] for r in accepted}),
        "accepted_families": len(
            {historical.source_lineage(r["family_id"]) for r in accepted}
        ),
        "views_per_style": len(rows),
        "base_counts": dict(counts),
        "source_counts": dict(sources),
        "controlled_count_counts": dict(count_values),
        "family_counts_by_stratum": {
            key: len(family_counts[key]) for key in historical.quotas(shared)
        },
        "paired_omission_bases": sum(r["paired_omission"] for r in accepted),
        "quota_shortfalls": deficits(historical.quotas(shared), counts),
        "source_shortfalls": deficits(historical.source_quotas(shared), sources),
        "controlled_count_shortfalls": deficits(
            historical.count_quotas(shared), count_values
        ),
        "family_shortfalls": family_shortfalls,
        "family_minima_configured": minima is not None,
        "coverage_shortfalls": coverage,
        "unresolved_verification_failures": sum(
            r["status"] == "verification-failed" for r in records
        ),
        **historical.direction_coverage(accepted),
        "tables_view_denominators": {
            "/".join(fields): dict(
                Counter("/".join(str(row.get(f)) for f in fields) for row in rows)
            )
            for fields in (
                ("split", "query_kind", "contract", "semantic_case"),
                ("split", "contract", "answer"),
                ("split", "query_kind", "depth"),
                ("split", "contract", "escape_reason"),
                ("split", "query_kind", "source"),
                ("split", "query_kind", "coverage"),
                ("split", "query_kind", "proof_pattern"),
                (
                    "split",
                    "query_kind",
                    "has_contingent_negative",
                    "has_impossible_negative",
                ),
            )
        },
        "tokens": {
            style: {
                "min": min(values, default=0),
                "max": max(values, default=0),
                "mean": sum(values) / len(values) if values else 0,
            }
            for style in ("natural", "symbolic")
            for values in [
                [
                    t["target_tokens"]
                    for r in accepted
                    for t in r["tokens"]
                    if t["style"] == style
                ]
            ]
        },
        "baseline": {
            contract: {
                "prediction_from_train_only": (
                    min(train, key=lambda answer: (-train[answer], answer))
                    if train
                    else None
                )
            }
            for contract in sorted(CONTRACTS)
            for train in [
                Counter(
                    row["answer"]
                    for row in rows
                    if row["split"] == "train" and row["contract"] == contract
                )
            ]
        },
    }


has_shortfalls = historical.has_shortfalls
write_exports = historical.write_exports
