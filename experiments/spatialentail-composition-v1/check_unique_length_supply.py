"""Read-only optimistic bounds on weakening the observed premise-count shortcut."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def count_bucket(record):
    return (
        record["stratum"],
        record.get("controlled_count_value"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh diagnostic output")
    receipt = json.loads(args.selection.read_text())
    if receipt["status"] != "selection-validated-not-sealed":
        raise ValueError("completed selection required")
    full = Path(receipt["metadata_directory"]) / "full-records.jsonl"
    if digest(full) != receipt["full_records_sha256"]:
        raise ValueError("replayed candidate pool changed")
    manifest = json.loads((args.candidate / "manifest.json").read_text())
    if manifest["selection_sha256"] != digest(args.selection):
        raise ValueError("candidate does not match selected provenance")
    current_path = args.candidate / "evidence.jsonl"
    if digest(current_path) != manifest["payload_sha256"]["evidence.jsonl"]:
        raise ValueError("selected evidence changed")
    with current_path.open() as stream:
        selected = [json.loads(line) for line in stream]
    # These are the exact quotas already validated in the selected artifact.
    quotas = Counter(count_bucket(r) for r in selected if r["source"] == "controlled")
    supply, groups, ids = Counter(), defaultdict(list), set()
    with full.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["base_id"] in ids:
                raise ValueError("duplicate base in replayed pool")
            ids.add(record["base_id"])
            n = len(record["problem"]["premise"]["operands"])
            is_e = record["rows"][0]["answer"] == "E"
            if any((row["answer"] == "E") != is_e for row in record["rows"]):
                raise ValueError("E classification unexpectedly depends on menu")
            kind, source = record["query_kind"], record["source"]
            groups[kind, source, "E" if is_e else "nonE"].append(n)
            if source == "controlled":
                supply[kind, count_bucket(record), is_e, n <= 13] += 1
    results = []
    for kind in ("direction", "count"):
        current = [r for r in selected if r["query_kind"] == kind]
        controlled = [r for r in current if r["source"] == "controlled"]
        assert len(current) == 2000 and len(controlled) == 1400
        assert all(r["rows"][0]["answer"] != "E" for r in controlled)
        # Every controlled pool base is non-E. The 1,400 controlled slots
        # exhaust all 1,400 target non-E answers, excluding random non-E supply.
        assert not groups.get((kind, "controlled", "E"))
        assert sum(r["rows"][0]["answer"] != "E" for r in current) == 1400
        per_cell = []
        for bucket, target in sorted(quotas.items(), key=lambda item: str(item[0])):
            if not bucket[0].startswith(f"train/{kind}/"):
                continue
            available = supply[kind, bucket, False, True]
            per_cell.append(
                {
                    "stratum": bucket[0],
                    "count_value": bucket[1],
                    "quota": target,
                    "short_nonE_supply": available,
                    "optimistic_selected": min(target, available),
                }
            )
        short_bound = sum(c["optimistic_selected"] for c in per_cell)
        source_caps = Counter(
            r["source"] for r in current if r["source"] != "controlled"
        )
        long_e = {
            source: {
                "quota": cap,
                "long_E_supply": sum(n >= 14 for n in groups[kind, source, "E"]),
            }
            for source, cap in source_caps.items()
        }
        long_bound = sum(min(c["quota"], c["long_E_supply"]) for c in long_e.values())
        errors = sum(
            (len(r["problem"]["premise"]["operands"]) <= 13)
            != (r["rows"][0]["answer"] == "E")
            for r in current
        )
        results.append(
            {
                "query": kind,
                "current_rule_errors": errors,
                "current_rule_accuracy": 1 - errors / 2000,
                "short_controlled_nonE_upper_bound": short_bound,
                "long_noncontrolled_E_upper_bound": long_bound,
                "rule_errors_upper_bound": short_bound + long_bound,
                "rule_accuracy_lower_bound_after_any_allowed_reselection": 1
                - (short_bound + long_bound) / 2000,
                "controlled_cells": per_cell,
                "long_E_by_source": long_e,
            }
        )
    report = {
        "status": "optimistic-bound-complete-not-reselected",
        "fixed_rule": "premises <= 13 predicts E; premises >= 14 predicts non-E",
        "pool_bases": len(ids),
        "method": (
            "Distinct-base census across the entire replayed pool. Cap short controlled "
            "non-E supply by exact depth quotas and, for Count, exact depth/value quotas. "
            "Cap long E supply by noncontrolled source quotas. Ignore family, negative "
            "coverage, menu availability and E-subtype coupling, so these are optimistic "
            "error upper bounds, not attainable selections or joint optima."
        ),
        "results": results,
        "source_length_supply": [
            {
                "query": q,
                "source": s,
                "category": c,
                "bases": len(ns),
                "min_premises": min(ns),
                "max_premises": max(ns),
                "short": sum(n <= 13 for n in ns),
                "long": sum(n >= 14 for n in ns),
            }
            for (q, s, c), ns in sorted(groups.items())
        ],
        "which_note": (
            "Which has both E and non-E controlled cases. The Direction/Count forced-source "
            "argument does not apply; no constrained optimum is claimed for Which."
        ),
        "limitations": (
            "Fixed-rule accuracy, not SFT or full A-F accuracy. Reselecting against this "
            "threshold could expose other shortcuts. These bounds already show this one "
            "shortcut cannot be removed under all existing constraints."
        ),
        "selection_sha256": digest(args.selection),
        "full_pool_sha256": digest(full),
        "candidate_manifest_sha256": digest(args.candidate / "manifest.json"),
        "diagnostic_sha256": digest(Path(__file__)),
        "dataset_changed": False,
        "new_generation": False,
        "quota_changes": False,
        "evaluation_used": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("controlled_cells", "long_E_by_source")
                }
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
