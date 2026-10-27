"""Measure per-stratum structural yield and paired token admission before release."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from spatial.v2.legacy.benchmark_registry import load_benchmark_registry
from spatial.v2.context_budget import ContextBudget, admit_row_group, load_tokenizer
from spatial.v2.legacy.generation import GenerationPolicy, SpatialGeneratorV2
from spatial.v2.supervision_controls import apply_supervision_arm


def audit_feasibility(config: Path, output: Path, *, candidates: int = 50) -> dict:
    if output.exists():
        raise FileExistsError(f"audit output exists: {output}")
    if candidates < 1:
        raise ValueError("candidate count must be positive")
    source = yaml.safe_load(config.read_text())
    base = config.parent
    cells = {
        c.id: c
        for filename in source["registries"]
        for c in load_benchmark_registry(base / filename).cells
    }
    budget = ContextBudget(**source["context_budget"])
    tokenizer = load_tokenizer(budget)
    report = []
    for request in source["roles"]["train"]:
        name = request.get("cell", request.get("stratum"))
        policy = (
            cells[name].generation_policy()
            if "cell" in request
            else GenerationPolicy(**request["policy"])
        )
        generator = SpatialGeneratorV2(source["seed"])
        seen = set()
        admitted_signatures = set()
        rejections = Counter()
        max_tokens = Counter()
        for _ in range(candidates):
            try:
                sample = generator.generate(policy, max_attempts=1)
            except RuntimeError:
                rejections["generation"] += 1
                continue
            natural = sample.as_sft_row()
            rows = [
                natural,
                sample.with_trace("symbolic").as_sft_row(),
                apply_supervision_arm(natural, "answer-only"),
            ]
            admission = admit_row_group(rows, tokenizer, budget)
            seen.add(sample.structure_signature)
            if admission.accepted:
                admitted_signatures.add(sample.structure_signature)
            else:
                rejections.update(admission.rejection_reasons)
            for row in admission.rows:
                package = (
                    row["metadata"]["trace_format"]
                    + ":"
                    + row["metadata"]["supervision_arm"]
                )
                max_tokens[package] = max(
                    max_tokens[package], row["metadata"]["training_model_tokens"]
                )
        result = {
            "stratum": name,
            "candidates": candidates,
            "unique_proposals": len(seen),
            "unique_admitted": len(admitted_signatures),
            "rejections": dict(rejections),
            "maximum_training_tokens": dict(max_tokens),
        }
        report.append(result)
        print(json.dumps(result), flush=True)
    receipt = {
        "schema": "spatialentail-feasibility-v1",
        "config": str(config),
        "context_budget": source["context_budget"],
        "strata": report,
        "claim": "bounded yield sample; does not establish full quotas",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=50)
    args = p.parse_args()
    audit_feasibility(args.config, args.out, candidates=args.candidates)


if __name__ == "__main__":
    main()
