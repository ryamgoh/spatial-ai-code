"""Fixed development-only selection comparison and saved-evidence prose replay."""

import argparse
import json
from collections import Counter
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_contracts import encode, parse_answer, parse_prompt
from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_feasibility import token_measurement
from spatial.v2.composition_readable import VERSION, render_outline
from spatial.v2.composition_readable_audit import check_row
from spatial.v2.composition_selection import (
    describe,
    inventory,
    select_milp,
    validate_selection,
)
from spatial.v2.composition_selection_expansion import PATTERNS, deduplicate
from spatial.v2.composition_shared_paths import factor_shared_paths
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import SpatialSolverV2

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/spatialentail-composition-v1"


def load_inputs():
    sources = [
        EXPERIMENT / "balance-feasibility-v1",
        EXPERIMENT / "unique-selection-expansion-v1",
    ]
    records, hashes = [], {}
    for source in sources:
        manifest = json.loads((source / "manifest.json").read_text())
        for name, expected in manifest["output_sha256"].items():
            if digest(source / name) != expected:
                raise ValueError(f"source output hash mismatch: {source / name}")
        for path in source.iterdir():
            if path.is_file():
                hashes[str(path)] = digest(path)
        records.append(
            [
                json.loads(line)
                for line in (source / "attempts.jsonl").read_text().splitlines()
            ]
        )
    comparison = EXPERIMENT / "research-analysis/COVERAGE-SELECTION-RESULTS-v2.json"
    hashes[str(comparison)] = digest(comparison)
    previous = json.loads(comparison.read_text())
    expanded = json.loads((sources[1] / "summary.json").read_text())
    if (
        previous["selection"]["status"] != "optimal"
        or expanded["selection"]["status"] != "optimal"
    ):
        raise ValueError("expected completed baseline selections")
    return records, previous, expanded, hashes


def selection_comparison(old, additions, previous, expanded):
    kept, duplicates = deduplicate(old, additions)
    bundles = inventory(old + kept)
    old_ids = previous["selection"]["bundle_ids"]
    validate_selection(bundles, old_ids, 70)
    required = set().union(*(b["cells"] for b in bundles if b["id"] in old_ids))
    if len(required) != 123:
        raise ValueError("baseline fine-cell inventory changed")
    result = {
        "required_cells": sorted(required),
        "duplicate_exclusions": duplicates,
        "old_pool": describe(bundles, old_ids),
        "expanded_escape_first": describe(bundles, expanded["selection"]["bundle_ids"]),
        "policies": {},
    }
    for name, caps in (
        ("coverage_only", {}),
        ("coverage_and_pattern_caps", {("which", pattern): 7 for pattern in PATTERNS}),
    ):
        selection = select_milp(bundles, required_cells=required, recipe_caps=caps)
        entry = {
            "selection": selection,
            "recipe_caps": [
                {"query_kind": kind, "recipe": recipe, "maximum": cap}
                for (kind, recipe), cap in caps.items()
            ],
        }
        if "bundle_ids" in selection:
            ids = selection["bundle_ids"]
            validate_selection(
                bundles, ids, 70, required_cells=required, recipe_caps=caps
            )
            entry["metrics"] = describe(bundles, ids)
            entry["selection_recipes"] = dict(
                Counter(
                    b["recipe"]
                    for b in bundles
                    if b["id"] in ids and b["kind"] == "which"
                )
            )
        result["policies"][name] = entry
        print(name, selection["status"], flush=True)
    return result


def replay_record(original, tokenizer, solver):
    if original["status"] != "accepted":
        raise ValueError("replay requires accepted bases")
    record = deepcopy(original)
    problem, _, _ = parse_prompt(record["rows"][0]["prompt"])
    if tagged_dataclass_to_dict(problem) != record["problem"]:
        raise ValueError("saved problem/prompt mismatch")
    evidence = build_evidence(problem, solver=solver)
    rebuilt = tagged_dataclass_to_dict(evidence)
    for field in ("problem", "consistent", "possible", "certain"):
        if rebuilt["semantics"][field] != original["evidence"]["semantics"][field]:
            raise ValueError(f"rebuilt semantics changed: {field}")
    tokens = []
    for row in record["rows"]:
        parsed, menu, mode = parse_prompt(row["prompt"])
        if (
            parsed != problem
            or list(menu) != row["menu"]
            or mode.value != row["contract"]
        ):
            raise ValueError("question, menu or contract mismatch")
        decision = encode(evidence.semantics, menu, mode)
        if (decision.letters, decision.semantic_case, decision.escape_reason) != (
            row["answer"],
            row["semantic_case"],
            row["escape_reason"],
        ):
            raise ValueError("saved gold or semantics mismatch")
        check_row(row, record["problem"])
        if render_outline(row["explanation_outline"]) != row["references"]:
            raise ValueError("saved references differ from original outline")
        outline = factor_shared_paths(row["explanation_outline"])
        outline["version"] = VERSION
        # Removing display-only factoring must recover exactly the old evidence.
        restored = deepcopy(outline)
        restored["version"] = row["explanation_outline"]["version"]
        restored["units"] = [u for u in restored["units"] if "comparison" not in u]
        for unit in restored["units"]:
            unit.pop("shared_segments", None)
        if restored != row["explanation_outline"]:
            raise ValueError("factoring changed selected evidence")
        row["explanation_outline"] = outline
        row["references"] = render_outline(outline)
        check_row(row, record["problem"])
        supported = (
            row["returned_count"] is not None
            and row["depth"] is not None
            and any(
                u["kind"] == "path" and len(u["indices"]) == row["depth"]
                for u in outline["units"]
            )
        )
        if supported != row["substantive_depth_support"]:
            raise ValueError("printed substantive depth support changed")
        for style, text in row["references"].items():
            if parse_answer(text, mode, stage=1) != row["answer"]:
                raise ValueError("rendered gold changed")
            tokens.append(token_measurement(tokenizer, row, style))
    record["tokens"] = tokens
    if not all(t["admitted"] for t in tokens):
        raise ValueError("replayed text failed token admission")
    restored = deepcopy(record)
    restored["tokens"] = original["tokens"]
    for old, new in zip(original["rows"], restored["rows"]):
        new["references"] = old["references"]
        new["explanation_outline"] = old["explanation_outline"]
    if restored != original:
        raise ValueError("protected source metadata changed")
    return record


def replay(records, output, tokenizer):
    solver = SpatialSolverV2(backend="z3", timeout_ms=5000)
    result = {"bases": len(records), "pairs": 0, "changed_pairs": 0, "failures": []}
    lengths = {style: {"before": [], "after": []} for style in ("natural", "symbolic")}
    groups = {}
    with (output / "attempts.jsonl").open("w") as stream:
        for index, original in enumerate(records, 1):
            try:
                record = replay_record(original, tokenizer, solver)
            except (ValueError, RuntimeError) as exc:
                record = {
                    "base_id": original["base_id"],
                    "status": "verification-failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                result["failures"].append(record)
            else:
                result["pairs"] += len(record["rows"])
                result["changed_pairs"] += sum(
                    a["references"] != b["references"]
                    for a, b in zip(original["rows"], record["rows"])
                )
                for style in lengths:
                    before = [
                        t["target_tokens"]
                        for t in original["tokens"]
                        if t["style"] == style
                    ]
                    after = [
                        t["target_tokens"]
                        for t in record["tokens"]
                        if t["style"] == style
                    ]
                    lengths[style]["before"].extend(before)
                    lengths[style]["after"].extend(after)
                    key = f"{original['recipe']}/depth-{original['requested_depth']}/{style}"
                    group = groups.setdefault(key, {"before": [], "after": []})
                    group["before"].extend(before)
                    group["after"].extend(after)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if index % 30 == 0:
                print(
                    f"{index}/{len(records)} bases; failures={len(result['failures'])}",
                    flush=True,
                )

    def stats(values):
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    result["tokens"] = {
        style: {phase: stats(values) for phase, values in phases.items() if values}
        for style, phases in lengths.items()
    }
    result["groups"] = {
        key: {phase: stats(values) for phase, values in phases.items()}
        for key, phases in groups.items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("selection", "replay"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite refinement artifacts")
    records, previous, expanded, hashes = load_inputs()
    code = list((ROOT / "spatial/v2").glob("*.py"))
    protocol = (
        EXPERIMENT / "research-analysis/SELECTION-AND-PROSE-REFINEMENT-PROTOCOL-v1.md"
    )
    hashes.update({str(p): digest(p) for p in [*code, protocol]})
    tokenizer = None
    packages = (
        ["z3-solver", "scipy"]
        if args.task == "selection"
        else ["z3-solver", "transformers", "tokenizers"]
    )
    if args.task == "replay":
        from transformers import AutoTokenizer

        if args.tokenizer is None or args.tokenizer.name != REVISION:
            raise ValueError("expected pinned tokenizer")
        token_hashes = {
            p.name: digest(p) for p in args.tokenizer.iterdir() if p.is_file()
        }
        original = json.loads(
            (EXPERIMENT / "unique-selection-expansion-v1/manifest.json").read_text()
        )
        if token_hashes != original["tokenizer_sha256"]:
            raise ValueError("tokenizer hash mismatch")
        hashes.update(
            {str(args.tokenizer / name): h for name, h in token_hashes.items()}
        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output.mkdir()
    manifest = {
        "status": "running-development-only",
        "task": args.task,
        "input_and_source_sha256": hashes,
        "versions": {package: version(package) for package in packages},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    result = (
        selection_comparison(*records, previous, expanded)
        if args.task == "selection"
        else replay(records[0] + records[1], args.output, tokenizer)
    )
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("source or input changed during refinement")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    passed = (
        all(p["selection"]["status"] == "optimal" for p in result["policies"].values())
        if args.task == "selection"
        else not result["failures"]
    )
    manifest["status"] = "completed-development-only"
    manifest["all_checks_passed"] = passed
    manifest["output_sha256"] = {
        p.name: digest(p) for p in args.output.iterdir() if p.name != "manifest.json"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("completed:", args.task, "passed:", passed)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
