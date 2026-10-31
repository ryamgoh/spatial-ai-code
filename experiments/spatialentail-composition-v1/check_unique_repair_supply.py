"""Check combined semantic supply after the single approved supplementary batch."""

import argparse
import json
from collections import Counter
from pathlib import Path

from check_unique_training_allocation import analyze, digest

from spatial.v2.entailment_unique_study import (
    project_record,
    training_allocation_policy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output receipt")
    manifest = json.loads(args.manifest.read_text())
    receipt = json.loads(args.receipt.read_text())
    if (
        receipt["status"] != "bounded-admission-complete-not-selected"
        or receipt["attempted_slots"] != 2048
        or receipt["manifest_sha256"] != digest(args.manifest)
    ):
        raise ValueError("a completed fixed-batch receipt is required")
    old_path = Path(manifest["old_candidate_path"])
    new_path = Path(receipt["metadata_directory"]) / "records.jsonl"
    if (
        digest(old_path) != manifest["input_sha256"][str(old_path)]
        or digest(new_path) != receipt["records_sha256"]
    ):
        raise ValueError("candidate metadata hash mismatch")
    config = args.manifest.parent / "entailment-config-frozen-v1.json"
    if digest(config) != manifest["input_sha256"][str(config.resolve())]:
        raise ValueError("frozen configuration changed")
    policy = training_allocation_policy(json.loads(config.read_text()))
    if policy != manifest["policy"]:
        raise ValueError("approved policy changed")
    candidates, new_records, seen = [], [], set()
    for path in (old_path, new_path):
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if path == new_path:
                    new_records.append(record)
                if record["status"] != "accepted":
                    continue
                if record.get("independent_audit") != "passed":
                    raise ValueError("accepted candidate lacks independent replay")
                if record["query_key"] in seen:
                    raise ValueError("duplicate canonical question")
                seen.add(record["query_key"])
                if record["split"] == "train":
                    candidates.append(project_record(record))
    # Old compact metadata does not contain the new selected-explanation audit.
    # Only source/answer bounds are evaluated here; do not invent depth evidence.
    supply_policy = {**policy, "coverage_minima": []}
    result = analyze(candidates, supply_policy)
    result.pop("coverage_checks")
    failures = sum(r["status"] == "verification-failed" for r in new_records)
    if failures:
        result["status"] = "verification-failure-requires-investigation"
    result.update(
        {
            "scope": "semantic source/answer necessary bounds only",
            "coverage_depth_family_count_selection": "not evaluated; no selected dataset",
            "manifest_sha256": digest(args.manifest),
            "admission_receipt_sha256": digest(args.receipt),
            "old_metadata_sha256": digest(old_path),
            "new_metadata_sha256": digest(new_path),
            "new_statuses": dict(Counter(r["status"] for r in new_records)),
            "new_accepted_semantic_supply": [
                {"source": source, "query": kind, "case": case, "bases": count}
                for (source, kind, case), count in sorted(
                    Counter(
                        (r["source"], r["query_kind"], r["semantics"]["case"])
                        for r in new_records
                        if r["status"] == "accepted"
                    ).items()
                )
            ],
            "approved_batch_exhausted": True,
            "automatic_refill_authorized": False,
            "evaluation_changed": False,
            "sealed": False,
            "training_started": False,
        }
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "new_statuses": result["new_statuses"],
                "conflicts": result["conflicts"],
                "new_accepted_semantic_supply": result["new_accepted_semantic_supply"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
