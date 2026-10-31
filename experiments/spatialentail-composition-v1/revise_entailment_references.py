"""Sparse reference-only repair; no scene generation, label changes or sealing."""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer

from spatial.v2.composition_execution import atomic_json
from spatial.v2.composition_feasibility import token_measurement
from spatial.v2.composition_production import summary_record, write_exports
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_evidence import build_evidence, render_pair
from spatial.v2.entailment_external import parse_external_prompt, verify_view
from spatial.v2.entailment_production import has_shortfalls, summarize
from spatial.v2.entailment_saved import load_evidence

VERSION = "unique-unlisted-selection-full-domain-v1"


def affected(row):
    return (
        row["query_kind"] == "which"
        and row["contract"] == "unique-entailed"
        and row["escape_reason"] == "unique-unlisted"
    )


def corrected(row, evidence, tokenizer, solver):
    result = deepcopy(row)
    result["references"] = render_pair(
        evidence, tuple(row["menu"]), Contract(row["contract"]), solver=solver
    )
    tokens = [
        token_measurement(tokenizer, result, style) for style in ("natural", "symbolic")
    ]
    if not all(token["admitted"] for token in tokens):
        raise ValueError("corrected explanation exceeds token limits")
    if result["references"] == row["references"]:
        raise ValueError("affected reference did not receive its missing justification")
    return result, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh output directory")
    if args.tokenizer.name != REVISION:
        parser.error("provide the pinned local tokenizer")
    source_manifest = args.source / "manifest.json"
    original_manifest = json.loads(source_manifest.read_text())
    if original_manifest.get("sealed") is not False:
        raise ValueError("reference revision requires an explicitly unsealed source")
    for name, sha in original_manifest.get("output_sha256", {}).items():
        if digest(args.source / name) != sha:
            raise ValueError(f"source artifact hash changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    solver = QuestionSolver()
    args.output.mkdir()
    manifest = {
        "status": "revising-unsealed-references",
        "sealed": False,
        "training_authorized": False,
        "source_manifest_sha256": digest(source_manifest),
        "source": str(args.source),
        "reference_revision": VERSION,
        "renderer_sha256": digest(Path("spatial/v2/entailment_evidence.py")),
        "manual_audit": "pending-corrected-reference-review",
    }
    atomic_json(args.output / "manifest.json", manifest)
    changed, measurements, records = [], [], []
    try:
        if args.external:
            with (
                (args.source / "references.jsonl").open() as source,
                (args.output / "references.jsonl").open("x") as output,
            ):
                for line in source:
                    row = json.loads(line)
                    if affected(row):
                        solver.clear()
                        problem, _, _, _ = parse_external_prompt(row["prompt"])
                        evidence = build_evidence(problem, solver=solver)
                        row, tokens = corrected(row, evidence, tokenizer, solver)
                        verify_view(row, solver=solver)
                        changed.append(row["id"])
                        measurements.extend(tokens)
                    output.write(json.dumps(row) + "\n")
            for name in (
                "natural-inputs.jsonl",
                "symbolic-inputs.jsonl",
                "evidence.jsonl",
            ):
                shutil.copy2(args.source / name, args.output / name)
        else:
            policy = json.loads((args.source / "configuration.json").read_text())
            with (
                (args.source / "attempts.jsonl").open() as source,
                (args.output / "attempts.jsonl").open("x") as output,
            ):
                for line in source:
                    record = json.loads(line)
                    targets = [row for row in record["rows"] if affected(row)]
                    if targets:
                        solver.clear()
                        evidence = load_evidence(record)
                        replacement = {}
                        for index, row in enumerate(record["rows"]):
                            if not affected(row):
                                continue
                            updated, tokens = corrected(
                                row, evidence, tokenizer, solver
                            )
                            record["rows"][index] = updated
                            changed.append(row["id"])
                            measurements.extend(tokens)
                            replacement.update(
                                {(t["id"], t["style"]): t for t in tokens}
                            )
                        record["tokens"] = [
                            replacement.get((token["id"], token["style"]), token)
                            for token in record["tokens"]
                        ]
                    output.write(json.dumps(record) + "\n")
                    records.append(summary_record(record))
            summary = summarize(records, policy)
            if has_shortfalls(summary):
                raise ValueError(
                    "reference-only revision changed quota/coverage acceptance"
                )
            atomic_json(args.output / "summary.json", summary)
            shutil.copy2(
                args.source / "configuration.json", args.output / "configuration.json"
            )
            with (args.output / "attempts.jsonl").open() as source:
                write_exports((json.loads(line) for line in source), args.output)
        unchanged_inputs = {}
        for path in args.source.glob("*inputs.jsonl"):
            new = args.output / path.name
            if digest(new) != digest(path):
                raise ValueError("reference revision changed model inputs")
            unchanged_inputs[path.name] = digest(new)
        receipt = {
            "revision": VERSION,
            "changed_view_ids": changed,
            "changed_views": len(changed),
            "questions_labels_splits_evidence_changed": False,
            "unchanged_model_inputs_sha256": unchanged_inputs,
            "changed_reference_tokens": measurements,
            "sealed": False,
        }
        atomic_json(args.output / "reference-revision.json", receipt)
        manifest.update(
            status="completed-unsealed-candidate",
            acceptance="automated-checks-passed",
            changed_views=len(changed),
            output_sha256={
                path.name: digest(path)
                for path in args.output.iterdir()
                if path.is_file() and path.name != "manifest.json"
            },
        )
        atomic_json(args.output / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "changed_views": len(changed),
                    "model_inputs_unchanged": True,
                    "sealed": False,
                }
            )
        )
    except BaseException as error:
        manifest.update(
            status="failed-unsealed-reference-revision",
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
