"""Export only a validated one-menu selection, without sealing or training."""

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import run_unique_training_repair as first

from spatial.v2.composition_feasibility import token_measurement
from spatial.v2.composition_training import messages
from spatial.v2.entailment_scene_protocol import SHORT_VERSION, parse_row
from spatial.v2.entailment_training_selection import (
    answer_category,
    menu_record,
    validate,
)
from spatial.v2.entailment_unique_study import project_record


def selected_inventory(receipt):
    if receipt.get("status") != "selection-validated-not-sealed":
        raise ValueError("completed validated selection required")
    items = receipt["selected_menus"]
    bases = {r["base_id"]: r["row_id"] for r in items}
    if len(items) != 6000 or len(bases) != 6000 or len(set(bases.values())) != 6000:
        raise ValueError("selection must contain 6,000 distinct bases and menus")
    return bases


def selected_record(record, row_id):
    rows = [row for row in record["rows"] if row["id"] == row_id]
    if len(rows) != 1:
        raise ValueError("selected row missing or duplicated")
    row = rows[0]
    if (
        record.get("independent_audit") != "passed"
        or row.get("selected_depth_audit") != "passed"
        or row.get("prompt_version") != SHORT_VERSION
    ):
        raise ValueError("selected row lacks short-protocol proof audit")
    parse_row(row)
    return menu_record(record, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("use a fresh directory under an existing parent")
    if args.tokenizer.name != first.REVISION or not args.tokenizer.is_dir():
        parser.error("pinned tokenizer required")
    receipt = json.loads(args.selection.read_text())
    wanted = selected_inventory(receipt)
    manifest = json.loads(args.manifest.read_text())
    if first.digest(args.manifest) != receipt["source_manifest_sha256"]:
        raise ValueError("selection manifest differs")
    hashes = dict(manifest["input_sha256"])
    spool = Path(receipt["metadata_directory"])
    hashes.update(
        {
            str(spool / "candidates.jsonl"): receipt["candidates_sha256"],
            str(spool / "full-records.jsonl"): receipt["full_records_sha256"],
            str(args.selection): first.digest(args.selection),
            str(args.manifest): first.digest(args.manifest),
        }
    )
    for path, expected in hashes.items():
        if first.digest(Path(path)) != expected:
            raise ValueError(f"provenance mismatch: {path}")
    candidates = [
        json.loads(line)
        for line in (spool / "candidates.jsonl").read_text().splitlines()
    ]
    old = Path(manifest["candidate_paths"][0])
    held_out = set(receipt["evaluation_base_ids"])
    expected_held_out = set()
    original_selection = set(
        json.loads(
            (
                args.manifest.parent / "unique-only-selection-v1/selection.json"
            ).read_text()
        )["selected_base_ids"]
    )
    evaluation = []
    with old.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["split"] != "train" and record["base_id"] in original_selection:
                expected_held_out.add(record["base_id"])
                evaluation.append(project_record(record))
    if held_out != expected_held_out or len(evaluation) != 4800:
        raise ValueError("evaluation inventory changed")
    selected = [
        menu_record(
            r, next(row for row in r["rows"] if row["id"] == wanted[r["base_id"]])
        )
        for r in candidates
        if r["base_id"] in wanted
    ]
    validate(candidates, selected, evaluation, manifest["policy"])
    by_base = {r["base_id"]: r for r in selected}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    args.output.mkdir()
    # An interrupted export has no completed manifest and cannot be a release.
    counts, seen = Counter(), set()
    max_tokens = {"natural": 0, "symbolic": 0}
    with ExitStack() as stack:
        source = stack.enter_context((spool / "full-records.jsonl").open())
        evidence = stack.enter_context((args.output / "evidence.jsonl").open("w"))
        references = stack.enter_context((args.output / "references.jsonl").open("w"))
        styles = {
            style: stack.enter_context(
                (args.output / f"{style}-training.jsonl").open("w")
            )
            for style in ("natural", "symbolic")
        }
        for line in source:
            record = json.loads(line)
            base_id = record["base_id"]
            if base_id not in wanted:
                continue
            if base_id in seen:
                raise ValueError("duplicate selected full record")
            chosen = selected_record(record, wanted[base_id])
            compact = first.historical.summary_record(chosen)
            compact.pop("graph", None)
            if compact != by_base[base_id]:
                raise ValueError(
                    "full evidence differs from validated selected metadata"
                )
            row = chosen["rows"][0]
            for style, stream in styles.items():
                measured = token_measurement(tokenizer, row, style)
                saved = next(t for t in chosen["tokens"] if t["style"] == style)
                if measured != saved or not measured["admitted"]:
                    raise ValueError("selected token admission changed")
                stream.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "base_id": base_id,
                            "messages": [
                                *messages(row, style),
                                {
                                    "role": "assistant",
                                    "content": row["references"][style],
                                },
                            ],
                        }
                    )
                    + "\n"
                )
                max_tokens[style] = max(max_tokens[style], measured["target_tokens"])
            evidence.write(json.dumps(chosen) + "\n")
            references.write(json.dumps(row) + "\n")
            seen.add(base_id)
            counts[f"{row['query_kind']}/{answer_category(row)}"] += 1
    if seen != set(wanted):
        raise ValueError("selected full evidence inventory incomplete")
    for path, expected in hashes.items():
        if first.digest(Path(path)) != expected:
            raise ValueError("input changed during export")
    report = {
        "status": "validated-unsealed-training-candidate",
        "bases": len(seen),
        "examples_per_style": len(seen),
        "answer_subtype_counts": dict(counts),
        "max_target_tokens": max_tokens,
        "selection_sha256": first.digest(args.selection),
        "policy": manifest["policy"],
        "payload_sha256": {
            p.name: first.digest(p) for p in sorted(args.output.glob("*.jsonl"))
        },
        "evaluation_base_ids": sorted(held_out),
        "evaluation_changed": False,
        "sealed": False,
        "training_started": False,
        "remaining": "manual prose review and separate release/training approval",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("policy", "evaluation_base_ids")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
