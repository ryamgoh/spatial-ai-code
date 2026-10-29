"""Replay existing pilot inputs with readable pairs; never sample or seal data."""

import argparse
import json
from collections import Counter
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    encode,
    parse_prompt,
)
from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_feasibility import summarize, token_measurement
from spatial.v2.composition_readable import (
    VERSION,
    build_outline,
    check_pair,
    render_outline,
)
from spatial.v2.composition_readable_audit import check_row
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import SpatialSolverV2


def rerender_record(original, tokenizer, solver):
    """Preserve source metadata and certificates; add the new checked outline."""
    if original["status"] != "accepted" or not original.get("data", {}).get("rows"):
        raise ValueError("readable replay requires an accepted source base")
    record = deepcopy(original)
    data = record["data"]
    problem, _, _ = parse_prompt(data["rows"][0]["prompt"])
    if tagged_dataclass_to_dict(problem) != record["problem"]:
        raise ValueError("saved prompt and problem differ")
    evidence = build_evidence(problem, solver=solver)
    rebuilt = tagged_dataclass_to_dict(evidence)
    previous = data["evidence"]["semantics"]
    for field in ("problem", "consistent", "possible", "certain"):
        if rebuilt["semantics"][field] != previous[field]:
            raise ValueError(f"rebuilt semantics differ: {field}")
    data["readable_evidence"] = rebuilt
    tokens = []
    for row in data["rows"]:
        parsed, menu, mode = parse_prompt(row["prompt"])
        if (
            parsed != problem
            or list(menu) != row["menu"]
            or mode.value != row["contract"]
        ):
            raise ValueError(f"{row['id']}: saved question/menu/contract mismatch")
        decision = encode(evidence.semantics, menu, mode)
        if (
            decision.letters != row["answer"]
            or decision.semantic_case != row["semantic_case"]
            or decision.escape_reason != row["escape_reason"]
        ):
            raise ValueError(f"{row['id']}: gold or semantic reason changed")
        outline = build_outline(evidence, menu, mode, solver=solver)
        pair = render_outline(outline)
        check_pair(evidence, menu, mode, pair, solver=solver)
        row["references"] = pair
        row["explanation_outline"] = outline
        check_row(row, record["problem"])
        tokens.extend(token_measurement(tokenizer, row, style) for style in pair)
    data["tokens"] = tokens
    record["status"] = (
        "accepted" if all(t["admitted"] for t in tokens) else "token-rejected"
    )
    assert_preserved(original, record)
    return record


def assert_preserved(original, replay):
    restored = deepcopy(replay)
    restored["status"] = original["status"]
    data = restored["data"]
    data.pop("readable_evidence")
    data["tokens"] = original["data"]["tokens"]
    if len(data["rows"]) != len(original["data"]["rows"]):
        raise ValueError("replay changed the number of views")
    for old, new in zip(original["data"]["rows"], data["rows"]):
        new.pop("explanation_outline")
        new["references"] = old["references"]
    if restored != original:
        raise ValueError("replay changed protected source metadata")


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a readable replay")
    if args.tokenizer.name != REVISION:
        raise ValueError("expected pinned tokenizer revision")
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    originals = [
        json.loads(line)
        for line in (args.source / "attempts.jsonl").read_text().splitlines()
    ]
    if len(originals) != source_manifest["attempt_budget"] or any(
        r["status"] != "accepted" for r in originals
    ):
        raise ValueError("expected the complete accepted source pilot")
    ids = [row["id"] for r in originals for row in r["data"]["rows"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source view IDs")
    tokenizer_hashes = {
        p.name: digest(p) for p in sorted(args.tokenizer.iterdir()) if p.is_file()
    }
    if tokenizer_hashes != source_manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from the source pilot")
    source_hashes = {
        p.name: digest(p) for p in sorted(args.source.iterdir()) if p.is_file()
    }
    root = Path(__file__).resolve().parents[2]
    files = [
        *sorted((root / "spatial/v2").glob("*.py")),
        *sorted((root / "spatial/v2/tests").glob("test_composition_readable*.py")),
        root
        / "experiments/spatialentail-composition-v1/research-analysis/READABLE-REPLAY-PROTOCOL.md",
        root / "experiments/spatialentail-composition-v1/pivot_20261026.md",
    ]
    code_hashes = {str(p.relative_to(root)): digest(p) for p in files}
    manifest = {
        "status": "running-readable-replay-not-production",
        "renderer": VERSION,
        "attempt_budget": len(originals),
        "source_views_per_style": len(ids),
        "source_pilot": str(args.source.resolve()),
        "source_pilot_sha256": source_hashes,
        "source_sha256": code_hashes,
        "tokenizer_sha256": tokenizer_hashes,
        "versions": {
            p: version(p) for p in ("z3-solver", "transformers", "tokenizers")
        },
        "solver_timeout_ms": 5000,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    solver = SpatialSolverV2(backend="z3", timeout_ms=5000)
    args.output.mkdir()
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (args.output / "attempts.jsonl").open("w") as stream:
        for index, original in enumerate(originals, 1):
            try:
                record = rerender_record(original, tokenizer, solver)
            except (ValueError, RuntimeError) as exc:
                record = {k: deepcopy(v) for k, v in original.items() if k != "data"}
                record.update(
                    status="unsupported"
                    if isinstance(exc, UnsupportedEvidence)
                    else "verification-failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if index % 30 == 0 or index == len(originals):
                print(
                    f"{index}/{len(originals)}: {dict(Counter(r['status'] for r in records))}",
                    flush=True,
                )
    if any(digest(root / p) != h for p, h in code_hashes.items()):
        raise ValueError("source code changed during replay")
    if any(digest(args.source / p) != h for p, h in source_hashes.items()):
        raise ValueError("source pilot changed during replay")
    result = summarize(records)
    result["changed_reference_pairs"] = sum(
        old_row["references"] != row["references"]
        for old, new in zip(originals, records)
        if "data" in new
        for old_row, row in zip(old["data"]["rows"], new["data"]["rows"])
    )
    result["protected_source_metadata_unchanged"] = all("data" in r for r in records)
    result["outline_units"] = dict(
        Counter(
            unit["kind"]
            for r in records
            if "data" in r
            for row in r["data"]["rows"]
            for unit in row["explanation_outline"]["units"]
        )
    )
    result["failures"] = [
        {
            "story_id": r["story_id"],
            "query_kind": r["query_kind"],
            "reason": r.get("reason", r["status"]),
        }
        for r in records
        if r["status"] != "accepted"
    ]
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest["status"] = "readable-replay-complete-not-production"
    manifest["all_bases_admitted"] = not result["failures"]
    manifest["qualitative_text_audit_performed"] = True
    manifest["output_sha256"] = {
        name: digest(args.output / name) for name in ("attempts.jsonl", "summary.json")
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status_counts",
                    "views_per_style",
                    "changed_reference_pairs",
                    "token_maxima_before_admission",
                    "failures",
                )
            },
            indent=2,
        )
    )
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
