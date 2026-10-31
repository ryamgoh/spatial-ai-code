"""Explicit Unique-only study projection; historical contracts remain supported."""

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean

from spatial.v2.entailment_contracts import VERSION as INTERFACE
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_scene_protocol import PLAIN_VERSIONS, SCOPE, SHORT_VERSION

STUDY_SCOPE = "unique-only-entailment-v1"
UNIQUE = Contract.UNIQUE.value
PAIR_FLAGS = {
    "unique_listed_differ",
    "unique_exhaustive_differ",
    "listed_exhaustive_differ",
}


def validate_row(row):
    if (
        row.get("study_scope") != STUDY_SCOPE
        or row.get("interface") != INTERFACE
        or row.get("contract") != UNIQUE
        or row.get("prompt_version") not in PLAIN_VERSIONS
        or row.get("scope_policy") != SCOPE
    ):
        raise ValueError("Unique-only study requires all-scene unique-entailed rows")
    if row.get("answer") not in tuple("ABCDEF"):
        raise ValueError("Unique-only gold must be one A-F letter")


def project_record(record):
    """Preserve bases and Unique content; drop other contracts and their tokens."""
    if record.get("status") != "accepted":
        raise ValueError("only accepted records can enter the Unique-only study")
    projected = deepcopy(record)
    projected["study_scope"] = STUDY_SCOPE
    projected["rows"] = [
        {**row, "study_scope": STUDY_SCOPE}
        for row in record["rows"]
        if row["contract"] == UNIQUE
    ]
    rows = projected["rows"]
    if not rows or sum(r["menu_role"] == "primary" for r in rows) != 1:
        raise ValueError("each Unique-only base requires exactly one primary view")
    if len({r["bundle_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate Unique-only menu view")
    for row in rows:
        validate_row(row)
    ids = {r["id"] for r in rows}
    projected["tokens"] = [t for t in record["tokens"] if t["id"] in ids]
    expected = {(i, style) for i in ids for style in ("natural", "symbolic")}
    actual = [(t["id"], t["style"]) for t in projected["tokens"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Unique-only views require matched Natural/Symbolic tokens")
    if not all(t["admitted"] for t in projected["tokens"]):
        raise ValueError("Unique-only view exceeds its token admission")
    return projected


def study_policy(frozen):
    from spatial.v2.entailment_production import all_scene_diagnostic_policy

    policy = all_scene_diagnostic_policy(frozen)
    policy.update(
        study_scope=STUDY_SCOPE,
        active_contracts=[UNIQUE],
        prompt_version=SHORT_VERSION,
        status="unique-only-awaiting-artifact-review",
    )
    policy.pop("diagnostic_multi_selection_bases", None)
    policy.pop("approval", None)
    rules = []
    retired = []
    for original in policy["coverage_minima"]:
        rule = deepcopy(original)
        match = rule["match"]
        if PAIR_FLAGS.intersection(match):
            retired.append(original)
            continue
        if match.get("contract") in (Contract.LISTED.value, Contract.EXHAUSTIVE.value):
            if any(
                match.get(f"has_{kind}_negative")
                for kind in ("contingent", "impossible")
            ):
                match["contract"] = UNIQUE
                rule["numeric_origin"] = "same negative-evidence minimum in Unique view"
            else:
                retired.append(original)
                continue
        rules.append(rule)
    policy["coverage_minima"] = rules
    policy["retired_contract_requirements"] = retired
    policy["checkpoint_selection"] = (
        "Unique Stage-1 dev view accuracy; earliest epoch wins ties"
    )
    policy["training_authorized"] = False
    return policy


def training_allocation_policy(frozen):
    """Approved repair: selected explanations, not unseen primary-menu proofs."""
    policy = study_policy(frozen)
    revised = 0
    for rule in policy["coverage_minima"]:
        match = rule["match"]
        if (
            match.get("split") == "train"
            and match.get("contract") == UNIQUE
            and match.get("menu_role") == "primary"
            and (
                match.get("substantive_depth_support") is True
                or match.get("decision_depth_support") is True
            )
        ):
            del match["menu_role"]
            match.pop("substantive_depth_support", None)
            match.pop("decision_depth_support", None)
            match["selected_explanation_depth_support"] = True
            rule["numeric_origin"] = "approved selected-explanation depth minimum"
            revised += 1
    if revised != 3:
        raise ValueError("expected exactly three training-wide depth rules")
    policy["training_allocation"] = {
        kind: {
            "A-D": 1200,
            "unique-unlisted": 200,
            "no-entailed-answer": 225 if kind == "which" else 450,
            "inconsistent": 75 if kind == "which" else 150,
            **({"multiple-entailed": 300} if kind == "which" else {}),
        }
        for kind in ("direction", "which", "count")
    }
    policy["status"] = "approved-bounded-training-repair-not-sealed"
    policy["supplementary_query_cap"] = 2048
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("use a fresh output directory under an existing parent")
    from spatial.v2.composition_training import digest
    from spatial.v2.entailment_retention import validate

    receipt = json.loads(args.receipt.read_text())
    if not receipt.get("full_retention_passed"):
        raise ValueError("a completed verified selection receipt is required")
    if digest(args.config) != receipt["config_sha256"]:
        raise ValueError("configuration differs from selected-base provenance")
    source = Path(receipt["intermediate_metadata_directory"]) / "candidates.jsonl"
    if digest(source) != receipt["candidates_sha256"]:
        raise ValueError("candidate metadata hash mismatch")
    ids = receipt["retained_base_ids"]
    selected_ids = set(ids)
    if len(ids) != len(selected_ids):
        raise ValueError("duplicate selected base IDs")
    with source.open() as stream:
        records = [
            record
            for line in stream
            if (record := json.loads(line))["base_id"] in selected_ids
        ]
    if {r["base_id"] for r in records} != selected_ids:
        raise ValueError("selected base inventory is incomplete")
    if any(r.get("independent_audit") != "passed" for r in records):
        raise ValueError("selected base lacks independent evidence replay")
    policy = study_policy(json.loads(args.config.read_text()))
    projected = [project_record(r) for r in records]
    validate(projected, projected, policy)
    rows = [row for r in projected for row in r["rows"]]
    summary = []
    for split in ("train", "dev", "in-range-test", "extrapolation-test"):
        for kind in ("all", "direction", "which", "count"):
            bases = [
                r
                for r in projected
                if r["split"] == split and (kind == "all" or r["query_kind"] == kind)
            ]
            views = [row for r in bases for row in r["rows"]]
            summary.append(
                {
                    "split": split,
                    "query": kind,
                    "bases": len(bases),
                    "views_per_style": len(views),
                    "answers": dict(Counter(row["answer"] for row in views)),
                    "escape_reasons": dict(
                        Counter(
                            row["escape_reason"]
                            for row in views
                            if row["answer"] in ("E", "F")
                        )
                    ),
                    "target_tokens": {
                        style: {
                            "mean": round(
                                mean(
                                    t["target_tokens"]
                                    for r in bases
                                    for t in r["tokens"]
                                    if t["style"] == style
                                ),
                                2,
                            ),
                            "max": max(
                                t["target_tokens"]
                                for r in bases
                                for t in r["tokens"]
                                if t["style"] == style
                            ),
                        }
                        for style in ("natural", "symbolic")
                    },
                }
            )
    report = {
        "study_scope": STUDY_SCOPE,
        "status": "unique-only-selection-validated-not-training-export",
        "previous_receipt_sha256": digest(args.receipt),
        "source_metadata_sha256": digest(source),
        "base_count": len(projected),
        "views_per_style": len(rows),
        "removed_views_per_style": sum(len(r["rows"]) for r in records) - len(rows),
        "active_requirements_passed": True,
        "retired_requirement_count": len(policy["retired_contract_requirements"]),
        "selected_base_ids": ids,
        "summary": summary,
        "selection_origin": "unchanged v4 mixed-contract selected bases",
        "multi_selection_target": "historical selection characteristic, not an active quota",
        "references_exported": False,
        "sealed": False,
        "training_started": False,
    }
    args.output.mkdir()
    (args.output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (args.output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    table = [
        "# Unique-Only Selection",
        "",
        "Same verified bases and Unique menus; Listed/Exhaustive views are excluded.",
        "This is a validated selection/configuration, not an exported or sealed training package.",
        "",
        "| Split | Query | Bases | Views per style | A-D | E | F |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in summary:
        gold = cell["answers"]
        table.append(
            f"| {cell['split']} | {cell['query']} | {cell['bases']} | {cell['views_per_style']} | {sum(gold.get(a, 0) for a in 'ABCD')} | {gold.get('E', 0)} | {gold.get('F', 0)} |"
        )
    table += [
        "",
        f"Total: {len(projected):,} bases, {len(rows):,} views per style.",
        "Omission companions are retained; no repeat exposure is added to replace removed contracts.",
        "Multi-entailment Which bases remain valid E cases. The prior 25% target is no longer a requirement.",
        "Base/source/depth/Count/family, semantic-case, negative-evidence and Unique answer/depth requirements were checked after projection.",
        "Contract-pair comparisons and Listed/Exhaustive gold/depth requirements are retired.",
        "External all-scene Unique adaptation and full reference export still require separate artifact validation.",
    ]
    (args.output / "README.md").write_text("\n".join(table) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "selected_base_ids"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
