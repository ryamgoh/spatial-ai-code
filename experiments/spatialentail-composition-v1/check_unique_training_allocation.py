"""Necessary one-menu allocation bounds; never selects, regenerates or seals."""

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

TARGETS = {
    kind: {
        "A-D": 1200,
        "unique-unlisted": 200,
        "no-entailed-answer": 225 if kind == "which" else 450,
        "inconsistent": 75 if kind == "which" else 150,
        **({"multiple-entailed": 300} if kind == "which" else {}),
    }
    for kind in ("direction", "which", "count")
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def category(row):
    if row["answer"] in ("A", "B", "C", "D"):
        return "A-D"
    if row["answer"] == "F" and row["escape_reason"] == "unique-unlisted":
        return "unique-unlisted"
    if row["answer"] == "E" and row["escape_reason"] in (
        "no-entailed-answer",
        "multiple-entailed",
        "inconsistent",
    ):
        return row["escape_reason"]
    raise ValueError("unexpected Unique answer case")


def eligible(record, categories, match=None):
    return any(
        category(row) in categories
        and all(row.get(k) == v for k, v in (match or {}).items())
        for row in record["rows"]
    )


def supply_bound(records, categories, quotas, match=None):
    """Independent upper bounds, not an assertion that their minimum is attainable."""
    found = [r for r in records if eligible(r, categories, match)]
    bounds = {"raw": len(found)}
    for quota_kind, field in (("source", "source_stratum"), ("stratum", "stratum")):
        counts = Counter(r[field] for r in found)
        bounds[quota_kind] = sum(
            min(count, quotas.get((quota_kind, key), 0))
            for key, count in counts.items()
        )
    return {**bounds, "upper_bound": min(bounds.values())}


def analyze(records, policy):
    from spatial.v2.entailment_retention import quota_targets

    quotas = quota_targets(policy)
    census, subset_checks, coverage_checks = [], [], []
    for kind, targets in TARGETS.items():
        bases = [r for r in records if r["query_kind"] == kind]
        for source in ("controlled", "random-world", "random-atoms", "targeted"):
            found = [r for r in bases if r["source"] == source]
            census.append(
                {
                    "query": kind,
                    "source": source,
                    "bases": len(found),
                    "source_quota": quotas["source", f"train/{kind}/{source}"],
                    "eligible_bases": {
                        cat: sum(eligible(r, {cat}) for r in found) for cat in targets
                    },
                }
            )
        for size in range(1, len(targets) + 1):
            for cats in combinations(targets, size):
                bound = supply_bound(bases, cats, quotas)
                required = sum(targets[cat] for cat in cats)
                subset_checks.append(
                    {
                        "query": kind,
                        "categories": list(cats),
                        "required": required,
                        **bound,
                        "shortfall": max(0, required - bound["upper_bound"]),
                    }
                )
        for rule in policy["coverage_minima"]:
            match = rule["match"]
            if match.get("split") != "train" or match.get("query_kind") != kind:
                continue
            if "minimum_bases" not in rule:
                continue
            # Match the chosen row, not an unseen primary menu on the same base.
            by_category = {
                cat: min(n, supply_bound(bases, {cat}, quotas, match)["upper_bound"])
                for cat, n in targets.items()
            }
            bound = min(
                sum(by_category.values()),
                supply_bound(bases, targets, quotas, match)["upper_bound"],
            )
            coverage_checks.append(
                {
                    "rule": rule,
                    "category_capped_supply": by_category,
                    "upper_bound": bound,
                    "shortfall": max(0, rule["minimum_bases"] - bound),
                }
            )
    conflicts = [
        {"type": "answer-subset", **r} for r in subset_checks if r["shortfall"]
    ] + [{"type": "coverage", **r} for r in coverage_checks if r["shortfall"]]
    return {
        "status": (
            "infeasible-under-retained-requirements"
            if conflicts
            else "necessary-bounds-pass-not-full-feasibility"
        ),
        "targets": TARGETS,
        "training_bases_available": len(records),
        "source_census": census,
        "answer_subset_checks": subset_checks,
        "coverage_checks": coverage_checks,
        "conflicts": conflicts,
        "full_selection_attempted": False,
        "limitations": [
            "Bounds prove impossibility when violated, not feasibility when satisfied.",
            "Count-value, family and all coverage constraints still require a full selection check if these bounds pass.",
            "Historical prompt token counts do not certify a short-protocol export.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a fresh file in an existing directory")
    from spatial.v2.entailment_unique_study import project_record, study_policy

    receipt = json.loads(args.receipt.read_text())
    if not receipt.get("full_retention_passed"):
        raise ValueError("completed v4 receipt required")
    if digest(args.config) != receipt["config_sha256"]:
        raise ValueError("configuration provenance mismatch")
    source = Path(receipt["intermediate_metadata_directory"]) / "candidates.jsonl"
    if digest(source) != receipt["candidates_sha256"]:
        raise ValueError("candidate hash mismatch")
    records, ids, keys = [], set(), set()
    with source.open() as stream:
        for line in stream:
            record = json.loads(line)
            if (
                record.get("status") != "accepted"
                or record.get("independent_audit") != "passed"
                or record["base_id"] in ids
                or record["query_key"] in keys
            ):
                raise ValueError("unverified or duplicate candidate")
            ids.add(record["base_id"])
            keys.add(record["query_key"])
            if record["split"] == "train":
                records.append(project_record(record))
    policy = study_policy(json.loads(args.config.read_text()))
    report = analyze(records, policy)
    report.update(
        {
            "candidate_path": str(source),
            "candidate_sha256": digest(source),
            "receipt_sha256": digest(args.receipt),
            "config_sha256": digest(args.config),
            "diagnostic_sha256": digest(Path(__file__)),
            "candidate_bases": len(ids),
            "inherited_training_coverage": [
                r
                for r in policy["coverage_minima"]
                if r["match"].get("split") == "train"
            ],
            "sealed": False,
            "training_started": False,
            "evaluation_changed": False,
            "new_seeds_or_menus": False,
        }
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_bases": len(ids),
                "training_bases": len(records),
                "conflicts": report["conflicts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
