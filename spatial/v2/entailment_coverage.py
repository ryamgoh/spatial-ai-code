"""Derive a reviewable entailment gate matrix, never a frozen execution policy."""

import argparse
import json
from itertools import combinations
from pathlib import Path

from spatial.v2.composition_execution import atomic_json
from spatial.v2.composition_production import KINDS, ROLES, quotas
from spatial.v2.composition_training import digest
from spatial.v2.entailment_contracts import VERSION, Contract


def derive(approved_policy):
    """Preserve numeric origins while declaring every semantic gate change."""
    targets = quotas(approved_policy)
    checks = approved_policy["diversity_checks"]
    rules, changes = [], []
    for cell, total in targets.items():
        role, kind, depth = cell.split("/")
        if not total:
            continue
        if depth != "semantic":
            if int(depth) >= 2:
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
                        "minimum_bases": (
                            total * checks["nonhomogeneous_controlled_percent"] + 99
                        )
                        // 100,
                        "numeric_origin": "approved nonhomogeneous proof percentage",
                    }
                )
            continue
        semantic_minimum = (total * checks["semantic_minimum_percent"] + 99) // 100
        difference_minimum = (
            total * checks["contract_difference_semantic_percent"] + 99
        ) // 100
        families = checks["contract_difference_families"][role]
        cases = ["unique-entailed", "no-entailed-answer", "inconsistent"]
        if kind == "which":
            cases.append("multiple-entailed")
        for case in cases:
            rules.append(
                {
                    "match": {"split": role, "query_kind": kind, "semantic_case": case},
                    "minimum_bases": semantic_minimum,
                    "minimum_families": families,
                    "numeric_origin": "approved semantic minimum percentage",
                }
            )
        for category in ("contingent", "impossible"):
            rules.append(
                {
                    "match": {
                        "split": role,
                        "query_kind": kind,
                        "contract": Contract.LISTED.value,
                        "has_" + category + "_negative": True,
                    },
                    "minimum_bases": semantic_minimum,
                    "minimum_families": families,
                    "numeric_origin": "proposed reuse of approved semantic minimum",
                }
            )
        for flag in (
            "unique_listed_differ",
            "unique_exhaustive_differ",
            "listed_exhaustive_differ",
        ):
            if flag == "listed_exhaustive_differ" and kind != "which":
                changes.append(
                    {
                        "split": role,
                        "query_kind": kind,
                        "gate": flag,
                        "previous_minimum_bases": difference_minimum,
                        "disposition": "structurally-inapplicable",
                        "reason": "Direction/Count entailed sets have size zero or one; Listed and Exhaustive always coincide.",
                    }
                )
                continue
            rules.append(
                {
                    "match": {"split": role, "query_kind": kind, flag: True},
                    "minimum_bases": families
                    if flag == "listed_exhaustive_differ"
                    else difference_minimum,
                    "minimum_families": families,
                    "numeric_origin": (
                        "user-approved rare-pair family floor as base minimum"
                        if flag == "listed_exhaustive_differ"
                        else "approved contract difference percentage"
                    ),
                }
            )
            if flag == "listed_exhaustive_differ":
                changes.append({
                    "split": role, "query_kind": kind, "gate": flag,
                    "previous_minimum_bases": difference_minimum,
                    "minimum_bases": families, "minimum_families": families,
                    "disposition": "user-approved-family-floor",
                    "reason": "The fixed training reserve has only 23 qualifying bases, below the former requirement of 120.",
                })
    # Carry substantive depth minima without translating old answer sets.
    mapping = {
        "single": Contract.UNIQUE.value,
        "visible-possible": Contract.LISTED.value,
        "all-possible": Contract.EXHAUSTIVE.value,
    }
    for rule in approved_policy["coverage_minima"]:
        match = dict(rule["match"])
        match["contract"] = mapping[match["contract"]]
        rules.append(
            {
                **rule,
                "match": match,
                "numeric_origin": "approved substantive depth minimum",
            }
        )
    for role in ROLES[:-1]:
        rules.append(
            {
                "match": {
                    "split": role,
                    "query_kind": "which",
                    "contract": Contract.UNIQUE.value,
                    "sole_entailed_with_contingent": True,
                },
                "minimum_bases": 1,
                "numeric_origin": "proposed presence gate for the revised Unique rule",
            }
        )
        for contract in Contract:
            golds = (
                list("ABCDEF")
                if contract is Contract.UNIQUE
                else [
                    *("".join(c) for n in range(1, 5) for c in combinations("ABCD", n)),
                    "E",
                    "F",
                ]
            )
            for gold in golds:
                rules.append(
                    {
                        "match": {
                            "split": role,
                            "contract": contract.value,
                            "answer": gold,
                        },
                        "minimum_bases": 1,
                        "numeric_origin": "historical feasible literal-gold presence gate",
                    }
                )
    changes.append(
        {
            "gate": "selection-no-match",
            "disposition": "retain-as-diagnostic-subtype",
            "reason": (
                "No entailed candidate can mean either all memberships are impossible "
                "or some are contingent. Retain separate observed subtype counts."
            ),
        }
    )
    return {
        "schema": "entailment-coverage-draft-v1",
        "interface": VERSION,
        "status": "draft-requires-coverage-review-not-executable",
        "frozen": False,
        "base_targets": approved_policy["base_targets"],
        "base_quotas": targets,
        "source_mix_percent": approved_policy["source_mix_percent"],
        "depth_family_minima": approved_policy["depth_family_minima"],
        "controlled_count_weights": approved_policy["controlled_count_weights"],
        "nonhomogeneous_controlled_percent": checks[
            "nonhomogeneous_controlled_percent"
        ],
        "paired_omission_percent": approved_policy["paired_omission_percent"],
        "rules": rules,
        "explicit_gate_changes": changes,
        "pending": [
            "Review proposed negative-category minima and changed gate semantics.",
            "Verify scheduled source/evidence supply for the new matrix.",
            "Include sole-entailed-plus-contingent Selection examples.",
            "Wire new rows, gates and exact quotas into ordered CPU execution.",
            "Freeze a new source/configuration/checkpoint identity; do not resume old runs.",
        ],
        "all_direction_presence": {
            "splits": list(ROLES),
            "queries": list(KINDS),
            "directions": 8,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh draft path under an existing parent")
    result = derive(json.loads(args.approved_config.read_text()))
    result["numeric_origin"] = {
        "path": str(args.approved_config),
        "sha256": digest(args.approved_config),
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rules": len(result["rules"]),
                "explicit_gate_changes": result["explicit_gate_changes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
