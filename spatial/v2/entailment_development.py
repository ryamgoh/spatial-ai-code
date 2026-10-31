"""Fixed existing development workload: correctness, explanations, and timing only."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from spatial.v2.composition_execution import atomic_json
from spatial.v2.composition_execution_benchmark import development_jobs
from spatial.v2.composition_production import admit as legacy_admit
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_contracts import VERSION
from spatial.v2.entailment_dataset import build_attempt


def run(tokenizer, *, legacy_policy=None):
    from dataclasses import replace

    results, runs = [], []
    jobs = development_jobs()
    if legacy_policy is not None:
        started = time.perf_counter()
        legacy = [
            legacy_admit(
                replace(entry, depth=None),
                problem,
                {**legacy_policy, "paired_omission_percent": 100},
                tokenizer,
                QuestionSolver(),
            )
            for _, entry, problem in jobs
        ]
        runs.append(
            {
                "mode": "historical-possibility-full-admission",
                "seconds": time.perf_counter() - started,
                "statuses": dict(Counter(r["status"] for r in legacy)),
            }
        )
    started = time.perf_counter()
    for _, entry, problem in jobs:
        record = build_attempt(
            entry.story,
            problem,
            None,
            tokenizer,
            solver=QuestionSolver(),
            omission_percent=100,
        )
        results.append(record)
    runs.append(
        {
            "mode": "entailment-only-full-admission",
            "seconds": time.perf_counter() - started,
            "statuses": dict(Counter(r["status"] for r in results)),
        }
    )
    if legacy_policy is not None:
        for old, new in zip(legacy, results):
            if old.get("semantics", {}).get("consistent") != new.get(
                "semantics", {}
            ).get("consistent") or old.get("semantics", {}).get("certain") != new.get(
                "semantics", {}
            ).get("entailed"):
                raise ValueError("migration changed underlying entailment results")
    rows = [row for r in results for row in r.get("rows", [])]
    return {
        "interface": VERSION,
        "status": "development-check-not-production-candidate",
        "production_generation": False,
        "sealed": False,
        "training_authorized": False,
        "workload": "six queries from the two existing CPU development scenes",
        "fresh_families": 0,
        "base_attempts": len(jobs),
        "rows_per_style": len(rows),
        "runs": runs,
        "timing_caveat": (
            "The two admissions implement different answer tasks. Semantic "
            "entailed sets are compared, not exact reference text or golds. "
            "This is not a full-production or server throughput estimate."
        ),
        "max_target_tokens": {
            style: max(
                (
                    t["target_tokens"]
                    for r in results
                    for t in r.get("tokens", [])
                    if t["style"] == style
                ),
                default=0,
            )
            for style in ("natural", "symbolic")
        },
        "failures": [
            {k: r.get(k) for k in ("base_id", "stage", "status", "reason")}
            for r in results
            if r["status"] != "accepted"
        ],
        "records": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-legacy-config", type=Path)
    args = parser.parse_args()
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("the pinned local tokenizer snapshot is required")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh report file under an existing directory")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    policy = (
        json.loads(args.compare_legacy_config.read_text())
        if args.compare_legacy_config
        else None
    )
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    hashes = {str(p): digest(p) for p in sources}
    result = run(tokenizer, legacy_policy=policy)
    if hashes != {str(p): digest(p) for p in sources}:
        raise ValueError("sources changed during development check")
    result["source_sha256"] = hashes
    result["tokenizer_sha256"] = {
        p.name: digest(p) for p in args.tokenizer.iterdir() if p.is_file()
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("records", "source_sha256", "tokenizer_sha256")
            },
            indent=2,
        )
    )
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
