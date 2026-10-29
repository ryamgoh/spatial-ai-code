"""Replay saved pilot inputs with compact Direction proofs, without resampling."""

import argparse
import json
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_audit import aligned_pair
from spatial.v2.composition_compact import VERSION, render_pair
from spatial.v2.composition_contracts import encode, parse_answer, parse_prompt
from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_feasibility import summarize, token_measurement
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import DirectionQuery, SpatialSolverV2


def rerender_record(original, tokenizer, solver):
    record = deepcopy(original)
    if "data" not in original:
        return record
    data = record["data"]
    first = data["rows"][0]
    problem, _, _ = parse_prompt(first["prompt"])
    if tagged_dataclass_to_dict(problem) != record["problem"]:
        raise ValueError("saved prompt and attempted problem differ")
    if isinstance(problem.query, DirectionQuery) and first["semantic_case"] == "unique":
        evidence = build_evidence(problem, solver=solver)
        previous = data["evidence"]["semantics"]
        current = tagged_dataclass_to_dict(evidence)["semantics"]
        for field in ("consistent", "possible", "certain"):
            if previous[field] != current[field]:
                raise ValueError(f"rebuilt semantics differ: {field}")
        data["evidence"] = tagged_dataclass_to_dict(evidence)
        for row in data["rows"]:
            parsed, menu, mode = parse_prompt(row["prompt"])
            if (
                parsed != problem
                or list(menu) != row["menu"]
                or mode.value != row["contract"]
            ):
                raise ValueError("paired row differs from saved problem or menu")
            if encode(evidence.semantics, menu, mode).letters != row["answer"]:
                raise ValueError("rerendering changed the gold mapping")
            row["references"] = render_pair(evidence, menu, mode, solver=solver)
    tokens = []
    for row in data["rows"]:
        aligned_pair(row)
        for style, text in row["references"].items():
            if parse_answer(text, row["contract"], stage=1) != row["answer"]:
                raise ValueError("rerendered answer differs")
            tokens.append(token_measurement(tokenizer, row, style))
    data["tokens"] = tokens
    record["status"] = (
        "accepted" if all(t["admitted"] for t in tokens) else "token-rejected"
    )
    return record


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a rerendered pilot")
    if args.tokenizer.name != REVISION:
        raise ValueError("expected pinned tokenizer revision")
    manifest = json.loads((args.source / "manifest.json").read_text())
    originals = [
        json.loads(s) for s in (args.source / "attempts.jsonl").read_text().splitlines()
    ]
    if len(originals) != manifest["attempt_budget"]:
        raise ValueError("source pilot is incomplete")
    tokenizer_hashes = {
        p.name: digest(p) for p in sorted(args.tokenizer.iterdir()) if p.is_file()
    }
    if tokenizer_hashes != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from the source pilot")
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "spatial/v2").glob("*.py"))
    protocol = (
        root
        / "experiments/spatialentail-composition-v1/research-analysis/COMPACT-DIRECTION-PROTOCOL.md"
    )
    hashes = {str(p.relative_to(root)): digest(p) for p in [*sources, protocol]}
    original_hashes = {
        p.name: digest(p) for p in sorted(args.source.iterdir()) if p.is_file()
    }
    manifest.update(
        status="rerender-feasibility-only-not-production",
        renderer=VERSION,
        source_pilot=str(args.source),
        source_pilot_sha256=original_hashes,
        source_sha256=hashes,
        versions={p: version(p) for p in ("z3-solver", "transformers", "tokenizers")},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    solver = SpatialSolverV2(backend="z3", timeout_ms=5000)
    args.output.mkdir()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (args.output / "attempts.jsonl").open("w") as stream:
        for original in originals:
            record = rerender_record(original, tokenizer, solver)
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
    if any(digest(root / path) != h for path, h in hashes.items()):
        raise ValueError("source code changed during replay")
    if any(digest(args.source / path) != h for path, h in original_hashes.items()):
        raise ValueError("source pilot changed during replay")
    result = summarize(records)
    result["changed_reference_pairs"] = sum(
        a["references"] != b["references"]
        for old, new in zip(originals, records)
        for a, b in zip(
            old.get("data", {}).get("rows", []), new.get("data", {}).get("rows", [])
        )
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status_counts",
                    "views_per_style",
                    "changed_reference_pairs",
                    "token_maxima_before_admission",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
