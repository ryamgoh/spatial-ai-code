"""Audit final payloads against the completed selection and replay provenance."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_entailment_candidate import label
from audit_selected_explanation_depth import certificate_depth

from spatial.v2.composition_training import digest, messages
from spatial.v2.entailment_scene_protocol import SHORT_VERSION, parse_row
from spatial.v2.entailment_training_selection import answer_category, validate
from spatial.v2.entailment_unique_study import project_record
from spatial.v2.serialization import tagged_dataclass_to_dict


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh audit receipt")
    manifest = json.loads((args.candidate / "manifest.json").read_text())
    selection = json.loads(args.selection.read_text())
    source = json.loads(args.source_manifest.read_text())
    assert manifest["status"] == "validated-unsealed-training-candidate"
    assert manifest["sealed"] is False and manifest["training_started"] is False
    assert selection["status"] == "selection-validated-not-sealed"
    assert manifest["selection_sha256"] == digest(args.selection)
    assert selection["source_manifest_sha256"] == digest(args.source_manifest)
    for name, expected in manifest["payload_sha256"].items():
        assert digest(args.candidate / name) == expected
    for path, expected in source["input_sha256"].items():
        assert digest(Path(path)) == expected, path
    assert source["policy"] == manifest["policy"]
    records = read_rows(args.candidate / "evidence.jsonl")
    rows = read_rows(args.candidate / "references.jsonl")
    assert len(records) == len(rows) == 6000
    assert len({r["base_id"] for r in records}) == 6000
    assert len({r["query_key"] for r in records}) == 6000
    assert len({r["id"] for r in rows}) == 6000
    assert {(r["base_id"], r["id"]) for r in rows} == {
        (r["base_id"], r["row_id"]) for r in selection["selected_menus"]
    }
    by_id = {r["id"]: r for r in rows}
    selected_metadata = []
    counts, sources, depths, decision_depths, cases = (Counter() for _ in range(5))
    families = defaultdict(set)
    f_depth = Counter()
    for record in records:
        assert len(record["rows"]) == 1
        row = record["rows"][0]
        assert row == by_id[row["id"]]
        assert row["prompt_version"] == SHORT_VERSION
        assert row["selected_depth_audit"] == "passed"
        problem, menu, contract = parse_row(row)
        assert record["problem"] == tagged_dataclass_to_dict(problem)
        sem = record["evidence"]["semantics"]
        # The serializer tags only the outer dataclass, not nested dataclasses.
        assert sem["problem"] == {
            k: v for k, v in record["problem"].items() if k != "type"
        }
        entailed = set(sem["entailed"])
        assert set(record["semantics"]["entailed"]) == entailed
        assert row["answer"] == label(sem["consistent"], entailed, menu, contract.value)
        expected_case = (
            "inconsistent"
            if not sem["consistent"]
            else "no-entailed-answer"
            if not entailed
            else "unique-entailed"
            if len(entailed) == 1
            else "multiple-entailed"
        )
        assert row["semantic_case"] == expected_case
        if row["answer"] == "F":
            assert sem["consistent"] and len(entailed) == 1
            assert not entailed.intersection(menu)
            positive = record["evidence"]["positive_support"]
            assert len(positive) == 1
            assert (
                certificate_depth(next(iter(positive.values())))
                == row["selected_explanation_depth"]
            )
            f_depth[f"{row['query_kind']}/{row['selected_explanation_depth']}"] += 1
        if expected_case in ("inconsistent", "no-entailed-answer"):
            assert row["selected_explanation_depth"] is None
            assert row["selected_explanation_depth_support"] is False
        assert row["depth"] is None or 1 <= row["depth"] <= 8
        assert len(record["tokens"]) == 2
        assert {t["style"] for t in record["tokens"]} == {"natural", "symbolic"}
        assert all(t["admitted"] and t["id"] == row["id"] for t in record["tokens"])
        assert all(t["stage2_reserved_tokens"] <= 8192 for t in record["tokens"])
        counts[f"{row['query_kind']}/{answer_category(row)}"] += 1
        cases[f"{row['query_kind']}/{expected_case}"] += 1
        sources[record["source_stratum"]] += 1
        depths[record["stratum"]] += 1
        families[record["stratum"]].add(record["family_id"])
        if row["selected_explanation_depth_support"]:
            decision_depths[row["query_kind"]] += 1
        selected_metadata.append(
            {
                **{
                    k: v
                    for k, v in record.items()
                    if k not in ("problem", "rows", "evidence", "graph")
                },
                "rows": [
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("prompt", "references", "explanation_outline")
                    }
                ],
            }
        )
    assert dict(counts) == manifest["answer_subtype_counts"]
    for style in ("natural", "symbolic"):
        inputs = read_rows(args.candidate / f"{style}-training.jsonl")
        assert len(inputs) == 6000 and {r["id"] for r in inputs} == set(by_id)
        for example in inputs:
            row = by_id[example["id"]]
            assert example["base_id"] == row["base_id"]
            assert example["messages"] == [
                *messages(row, style),
                {"role": "assistant", "content": row["references"][style]},
            ]
    # Revalidate all quotas, Count values, family floors and coverage together
    # with exactly the original held-out selections.
    pool_path = Path(selection["metadata_directory"]) / "candidates.jsonl"
    assert digest(pool_path) == selection["candidates_sha256"]
    pool = read_rows(pool_path)
    old = read_rows(Path(source["candidate_paths"][0]))
    old_ids = set(
        json.loads(
            (
                args.source_manifest.parent / "unique-only-selection-v1/selection.json"
            ).read_text()
        )["selected_base_ids"]
    )
    evaluation = [
        project_record(r)
        for r in old
        if r["split"] != "train" and r["base_id"] in old_ids
    ]
    assert len(evaluation) == 4800
    assert set(selection["evaluation_base_ids"]) == {r["base_id"] for r in evaluation}
    validate(pool, selected_metadata, evaluation, manifest["policy"])
    report = {
        "status": "passed",
        "bases": 6000,
        "examples_per_style": 6000,
        "styles": ["natural", "symbolic"],
        "answer_subtype_counts": dict(counts),
        "semantic_cases": dict(cases),
        "source_counts": dict(sources),
        "stratum_counts": dict(depths),
        "stratum_family_counts": {k: len(v) for k, v in families.items()},
        "selected_depth_qualifying_counts": dict(decision_depths),
        "f_positive_proof_depth_counts": dict(f_depth),
        "candidate_manifest_sha256": digest(args.candidate / "manifest.json"),
        "payload_sha256": manifest["payload_sha256"],
        "selection_sha256": digest(args.selection),
        "source_manifest_sha256": digest(args.source_manifest),
        "auditor_sha256": digest(Path(__file__)),
        "verification_scope": (
            "All exported rows and style messages, semantic label convention, F proof depth, "
            "exact quotas and coverage, token-admission metadata and immutable replay provenance. "
            "Full spatial evidence was independently replayed upstream; Z3 was not rerun here."
        ),
        "unchanged_evaluation_bases": 4800,
        "evaluation_input_hashes_verified": True,
        "sealed": False,
        "training_started": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
