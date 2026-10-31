"""Audit the published replacement against its admitted records and fixed inputs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_entailment_candidate import label, require
from review_unique_pretraining import digest

from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_pilot import story_key
from spatial.v2.composition_training import messages
from spatial.v2.entailment_replacement_allocation import validate_records
from spatial.v2.entailment_scene_protocol import parse_row
from spatial.v2.serialization import tagged_dataclass_to_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".md").exists():
        parser.error("audit output already exists")
    build = json.loads(args.receipt.read_text())
    require(build["status"] == "audited-unsealed-training-export", "build did not pass")
    prep_path = args.receipt.with_suffix(".manifest.json")
    require(
        digest(prep_path) == build["manifest_sha256"], "preparation fingerprint changed"
    )
    prep = json.loads(prep_path.read_text())
    for path, expected in {**prep["protected_sha256"], **prep["source_sha256"]}.items():
        require(digest(Path(path)) == expected, f"input/source changed: {path}")
    directory = Path(build["output"])
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["payload_sha256"].items():
        require(digest(directory / name) == expected, f"payload changed: {name}")
    spool = Path(build["spool"])
    for name, field in (
        ("held-out.json", "held_out_sha256"),
        ("selected-constructions.json", "selected_constructions_sha256"),
        ("admissions.jsonl", "admissions_sha256"),
    ):
        require(digest(spool / name) == build[field], f"build evidence changed: {name}")
    held_out = json.loads((spool / "held-out.json").read_text())
    with (directory / "evidence.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    validate_records(records, prep["policy"])
    admitted = {}
    with (spool / "admissions.jsonl").open() as stream:
        for line in stream:
            result = json.loads(line)
            require(result["status"] == "accepted", "unaccepted admission")
            record = result["record"]
            require(record["base_id"] not in admitted, "duplicate admitted base")
            admitted[record["base_id"]] = record
    require(len(admitted) == len(records) == 6000, "wrong admission/export counts")
    answers, depths, sizes, phrases = (
        defaultdict(Counter),
        defaultdict(Counter),
        defaultdict(Counter),
        Counter(),
    )
    seen = set()
    for record in records:
        require(
            record == admitted[record["base_id"]],
            "export differs from independently admitted record",
        )
        row = record["rows"][0]
        problem, menu, contract = parse_row(row)
        require(
            tagged_dataclass_to_dict(problem) == record["problem"],
            "parsed prompt differs",
        )
        key = query_profile(problem)["key"]
        scene = story_key(problem.objects, problem.premise)
        require(
            key == record["query_key"] and scene == record["scene_key"],
            "canonical key changed",
        )
        require(
            key not in seen and key not in held_out["query_keys"],
            "duplicate/held-out query",
        )
        require(scene not in held_out["scene_keys"], "held-out scene")
        require(
            record["anchor_key"] not in held_out["anchor_keys"]
            and record["anchor_key"] not in held_out["scene_keys"],
            "held-out anchor",
        )
        require(
            record["family_id"] == f"controlled-v2/{record['anchor_key']}",
            "family differs from construction anchor",
        )
        seen.add(key)
        expected = label(
            record["semantics"]["consistent"],
            set(record["semantics"]["entailed"]),
            menu,
            contract.value,
        )
        require(row["answer"] == expected, "independent answer encoding differs")
        kind = record["query_kind"]
        answers[kind][expected if expected in "EF" else "A-D"] += 1
        if not record["stratum"].endswith("/semantic"):
            depths[kind][str(record["measurement"]["depth"])] += 1
        n = len(problem.premise.operands)
        sizes[kind, "at-most-13" if n <= 13 else "over-13"][
            "E" if expected == "E" else "non-E"
        ] += 1
        for text in row["references"].values():
            for phrase in (
                "every contract requires",
                "Unique requires",
                "No minimality claim",
                "complete entailed-answer set",
            ):
                phrases[phrase] += phrase in text
    for kind in answers:
        require(
            answers[kind] == Counter({"A-D": 1200, "E": 600, "F": 200}),
            "answer totals differ",
        )
    require(not any(phrases.values()), "retired explanation boilerplate remains")
    for name in (
        "references.jsonl",
        "natural-training.jsonl",
        "symbolic-training.jsonl",
    ):
        with (directory / name).open() as stream:
            for record, line in zip(records, stream, strict=True):
                value, row = json.loads(line), record["rows"][0]
                if name == "references.jsonl":
                    require(value == row, "reference row differs")
                else:
                    style = name.split("-")[0]
                    require(
                        value["id"] == row["id"]
                        and value["base_id"] == record["base_id"],
                        "chat identity differs",
                    )
                    require(
                        value["messages"]
                        == [
                            *messages(row, style),
                            {"role": "assistant", "content": row["references"][style]},
                        ],
                        "chat messages differ",
                    )
    report = {
        "status": "passed",
        "bases": len(records),
        "examples_per_style": 6000,
        "answers": dict(answers),
        "depth_cohort_counts": dict(depths),
        "size_diagnostic": [
            {
                "query": kind,
                "range": bucket,
                **counts,
                "E_fraction": counts["E"] / sum(counts.values()),
            }
            for (kind, bucket), counts in sorted(sizes.items())
        ],
        "retired_phrase_counts": dict(phrases),
        "max_target_tokens": build["max_target_tokens"],
        "input_payloads_unchanged": True,
        "candidate_manifest_sha256": digest(directory / "manifest.json"),
        "audit_source_sha256": digest(Path(__file__)),
        "spatial_verification": "Independent SMT/path/counterexample replay was performed for every admitted question in the build. This artifact audit verifies exact exported equality and provenance; it does not rerun SMT.",
        "sealed": False,
        "training_started": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Replacement Training Audit",
        "",
        "Status: passed. Unsealed; no model training started.",
        "",
        "6,000 distinct canonical questions, with one selected menu and matched Natural/Symbolic supervision.",
        "",
        "| Query | A-D | E | F |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {kind} | {counts['A-D']} | {counts['E']} | {counts['F']} |"
        for kind, counts in sorted(answers.items())
    )
    lines.extend(
        [
            "",
            "All preserved depth, Count-value, family and evidence minima passed. Fixed evaluation queries, scenes and anchors were excluded. Existing payload hashes are unchanged.",
            "",
            f"Maximum target tokens: {report['max_target_tokens']}.",
            "",
            "## Scene Length Check",
            "",
            "Descriptive E shares, not a model prediction experiment. The previous shortcut used the 13-premise boundary.",
            "",
            "| Query | Premises | E share |",
            "|---|---|---:|",
        ]
    )
    lines.extend(
        f"| {r['query']} | {r['range']} | {r['E_fraction']:.1%} |"
        for r in report["size_diagnostic"]
    )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- The construction recipes are controlled and restricted; this audit does not establish absence of all structural shortcuts.",
            "- Reference cleanup preserves proof obligations; this is not a claim that every explanation reads like independently written human prose.",
            "- The training payload is exported as chat JSONL. No GPU loading, optimizer step, training smoke test or SFT run was performed.",
            "",
            report["spatial_verification"],
            "",
            f"Candidate: `{directory}`",
            "",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
