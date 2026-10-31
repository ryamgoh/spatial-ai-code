"""Replay external evidence with independent atomic checks and source legends."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from audit_entailment_candidate import audit_record, require

from spatial.v2.composition_training import digest, messages
from spatial.v2.entailment_contracts import VERSION, render_prompt
from spatial.v2.entailment_external import adapt_problem, parse_external_prompt
from spatial.v2.entailment_scene_protocol import (
    PLAIN_VERSIONS,
    SCOPE,
    all_scene_problem,
)
from spatial.v2.serialization import tagged_dataclass_to_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {}
    with (args.candidate / "evidence.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            require(
                item["source_id"] not in evidence, "duplicate external evidence base"
            )
            evidence[item["source_id"]] = item
    groups, original_prompts, stories = defaultdict(list), {}, set()
    with (args.candidate / "references.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            protocol = row.get("prompt_version")
            require(
                (protocol, row.get("scope_policy"))
                in ((None, None), *((version, SCOPE) for version in PLAIN_VERSIONS)),
                "unknown external protocol",
            )
            problem, menu, contract, legend = parse_external_prompt(
                row["prompt"], prompt_version=protocol, names=row["entity_names"]
            )
            source_problem, source_names, _ = adapt_problem(
                row["source_text"],
                row["selection_scope"],
                preserve_source_names=protocol in PLAIN_VERSIONS,
            )
            if protocol in PLAIN_VERSIONS:
                require(
                    row["selection_scope"] == "all-nonreference", "restricted scene"
                )
                source_problem = all_scene_problem(source_problem)
                require(
                    row.get("display_names") == source_names, "display mapping changed"
                )
                original_scene = row["source_text"].partition("Please answer")[0]
                original_scene = original_scene.partition(
                    "Consider a map with multiple objects:"
                )[2]
                original_scene = " ".join(original_scene.split())
                require(
                    row.get("scene_paragraph") == original_scene
                    and row["prompt"].splitlines()[1:3] == [original_scene, ""],
                    "external source paragraph changed",
                )
            require(
                problem == source_problem
                and legend == source_names == row["entity_names"],
                "external source/legend changed",
            )
            require(
                row["problem"] == tagged_dataclass_to_dict(problem)
                and row["menu"] == list(menu)
                and row["contract"] == contract.value,
                "external structured record differs from prompt",
            )
            require(row["id"] not in original_prompts, "duplicate external view")
            original_prompts[row["id"]] = row
            stories.add(row["story_id"])
            normalized = (
                row
                if protocol in PLAIN_VERSIONS
                else {**row, "prompt": render_prompt(problem, menu, contract)}
            )
            groups[row["base_id"]].append(normalized)
    require(set(groups) == set(evidence), "external evidence/domain mismatch")
    positives, negatives, views = 0, 0, 0
    for base, rows in groups.items():
        saved = evidence[base]
        record = {
            "interface": VERSION,
            "problem": rows[0]["problem"],
            "semantics": saved["semantics"],
            "evidence": saved["evidence"],
            "rows": rows,
        }
        if rows[0].get("prompt_version") is not None:
            record.update(
                prompt_version=rows[0]["prompt_version"],
                scope_policy=rows[0]["scope_policy"],
            )
        if rows[0].get("study_scope") is not None:
            record["study_scope"] = rows[0]["study_scope"]
        count_positive, count_negative = audit_record(record)
        positives += count_positive
        negatives += count_negative
        views += len(rows)
    input_rows = 0
    for style in ("natural", "symbolic"):
        seen = set()
        with (args.candidate / f"{style}-inputs.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                require(
                    set(row) == {"id", "messages"} and row["id"] not in seen,
                    "invalid or duplicate external model input",
                )
                seen.add(row["id"])
                require(
                    row["messages"] == messages(original_prompts[row["id"]], style),
                    "external input changed or contains audit data",
                )
        require(seen == set(original_prompts), "external input view missing")
        input_rows += len(seen)
    receipt = {
        "status": "passed",
        "scope": "all external bases, claims, views, legends and inputs",
        "bases": len(groups),
        "stories": len(stories),
        "views": views,
        "positive_claims": positives,
        "counterexamples": negatives,
        "input_rows": input_rows,
        "sealed": False,
        "candidate_manifest_sha256": digest(args.candidate / "manifest.json"),
        "artifact_sha256": {p.name: digest(p) for p in args.candidate.glob("*.jsonl")},
        "auditor_sha256": digest(Path(__file__)),
        "atomic_checker_sha256": digest(
            Path(__file__).with_name("audit_entailment_candidate.py")
        ),
        "independent_of": [
            "production solver",
            "production contract encoder",
            "production proof and counterexample replay",
        ],
        "normalization": "source adapter and prompt parsers; no label computation",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
