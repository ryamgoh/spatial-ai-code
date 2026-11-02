"""Historical artifact-audit CLI; verification lives in spatial.v2."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from spatial.v2.composition_training import digest, messages
from spatial.v2.entailment_artifact_audit import (
    SIGNS,
    audit_record,
    checked_sat,
    independent_claims,
    label,
    relation,
    replay_conflict,
    replay_counterexample,
    replay_paths,
    require,
)

__all__ = [
    "SIGNS",
    "audit_record",
    "checked_sat",
    "independent_claims",
    "label",
    "relation",
    "replay_conflict",
    "replay_counterexample",
    "replay_paths",
    "require",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.candidate / "manifest.json").read_text())
    require(
        manifest["status"] == "completed-unsealed-candidate" and not manifest["sealed"],
        "audit requires a completed unsealed candidate",
    )
    for name, sha in manifest["output_sha256"].items():
        require(digest(args.candidate / name) == sha, f"artifact hash mismatch: {name}")
    counts, families, ids = Counter(), {}, set()
    inputs = {}
    with (args.candidate / "attempts.jsonl").open() as stream:
        for index, line in enumerate(stream, 1):
            record = json.loads(line)
            require(record["base_id"] not in ids, "duplicate independent base")
            ids.add(record["base_id"])
            family = record["family_id"]
            legacy = re.search(
                r"depth-(?:in-range|extrapolation)-family-(\d+)$", family
            )
            if legacy:
                family = f"controlled-depth/sample-{legacy[1]}"
            require(
                families.setdefault(family, record["split"]) == record["split"],
                "family crosses splits",
            )
            positives, negatives = audit_record(record)
            counts["bases"] += 1
            counts["positive_claims"] += positives
            counts["counterexamples"] += negatives
            for row in record["rows"]:
                require(row["id"] not in inputs, "duplicate view ID")
                inputs[row["id"]] = (row["split"], row)
                counts["views"] += 1
            if index % 500 == 0:
                print(json.dumps(dict(counts)), flush=True)
    for split in ("train", "dev", "in-range-test", "extrapolation-test"):
        for style in ("natural", "symbolic"):
            seen = set()
            with (args.candidate / f"{split}-{style}-inputs.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    require(set(row) == {"id", "messages"}, "input leaks audit fields")
                    require(row["id"] not in seen, "duplicate input view")
                    seen.add(row["id"])
                    require(inputs[row["id"]][0] == split, "wrong input split")
                    require(
                        row["messages"] == messages(inputs[row["id"]][1], style),
                        "model input differs from matched prompt",
                    )
            require(
                seen == {key for key, (role, _) in inputs.items() if role == split},
                "missing model inputs",
            )
            counts["input_rows"] += len(seen)
    receipt = {
        "status": "passed",
        "scope": "all retained bases, claims, views and model inputs",
        "independent_of": [
            "production entailment analysis",
            "production contract encoder",
            "production path replay",
            "production counterexample checker",
        ],
        "counts": dict(counts),
        "families": len(families),
        "candidate_manifest_sha256": digest(args.candidate / "manifest.json"),
        "auditor_sha256": digest(Path(__file__)),
        "atomic_checker_sha256": digest(
            Path(__file__).resolve().parents[2]
            / "spatial/v2/entailment_artifact_audit.py"
        ),
        "not_covered": [
            "manual prose quality",
            "token remeasurement",
            "coverage quota re-solving",
        ],
        "sealed": False,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
