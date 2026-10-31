"""Bounded constructive balance feasibility; never writes training data."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from audit_entailment_candidate import audit_record

from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_balance_probe import VERSION, probe_scene
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_dataset import build_attempt, uniqueness_failure_depth
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import VERSION as PROMPT_VERSION
from spatial.v2.solver import Direction


def jobs():
    grid = [
        (3, depth, direction, matches)
        for depth in range(1, 11)
        for direction in Direction
        for matches in range(1, 5)
    ]

    families = [
        (seed, depth, direction, matches)
        for seed in range(8)
        if seed != 3
        for depth in (1, 4, 8, 10)
        for direction in (Direction.NORTH, Direction.NORTHEAST)
        for matches in range(1, 5)
    ]
    return grid + families


def inspect(seed, depth, direction, matches, tokenizer):
    story, problem = probe_scene(seed, depth, direction, matches)
    solver = QuestionSolver()
    # Force omissions in this development test so every applicable case is checked.
    record = build_attempt(
        story,
        problem,
        depth,
        tokenizer,
        solver=solver,
        omission_percent=100,
        prompt_version=PROMPT_VERSION,
    )
    result = {
        "seed": seed,
        "depth": depth,
        "direction": direction.value,
        "requested_entailed_count": matches,
        "family": story.source_family,
        "base_id": record["base_id"],
        "status": record["status"],
        "reason": record.get("reason"),
        "objects": len(problem.objects),
        "premises": len(problem.premise.operands),
    }
    if record["status"] != "accepted":
        return result, None
    evidence = load_evidence(record)
    if len(evidence.semantics.entailed) != matches:
        raise ValueError("construction cardinality differs from all-scene entailment")
    failure_depth = uniqueness_failure_depth(evidence)
    if matches > 1 and failure_depth != depth:
        raise ValueError("Unique has a shallow decision certificate")
    primary = [r for r in record["rows"] if r["menu_role"] == "primary"]
    if not all(r["decision_depth_support"] for r in primary):
        raise ValueError("primary decision lacks requested support depth")
    for row in primary:
        expected_size = 1 if matches == 1 else matches
        if row["contract"] == Contract.UNIQUE.value and matches > 1:
            if row["answer"] != "E":
                raise ValueError("Unique must escape on multiple entailments")
        elif len(row["answer"]) != expected_size or set(row["answer"]) - set("ABCD"):
            raise ValueError("primary substantive gold size differs")
    for row in record["rows"]:
        if row["menu_role"] != "omission":
            continue
        if row["contract"] == Contract.EXHAUSTIVE.value and row["answer"] != "F":
            raise ValueError("omission contract label differs")
        if row["contract"] == Contract.UNIQUE.value and row["answer"] != (
            "F" if matches == 1 else "E"
        ):
            raise ValueError("Unique omission label differs")
        if (
            row["contract"] == Contract.LISTED.value
            and matches > 1
            and len(row["answer"]) != matches - 1
        ):
            raise ValueError("Listed omission size differs")
        if (
            row["contract"] == Contract.LISTED.value
            and matches == 1
            and row["answer"] != "F"
        ):
            raise ValueError("Listed singleton omission must escape")
    claims = audit_record(record)
    result.update(
        actual_entailed_count=len(evidence.semantics.entailed),
        measured_depth=record["measurement"]["depth"],
        uniqueness_failure_depth=failure_depth,
        positive_member_depths=sorted(
            max(len(p["indices"]) for p in support["paths"])
            for support in evidence.positive_support.values()
        ),
        independent_claim_counts=claims,
        primary_answers={row["contract"]: row["answer"] for row in primary},
        omission_checked=any(r["menu_role"] == "omission" for r in record["rows"]),
        context_removal_checked=bool(story.distractors),
        max_target_tokens={
            style: max(
                t["target_tokens"] for t in record["tokens"] if t["style"] == style
            )
            for style in ("natural", "symbolic")
        },
    )
    return result, record


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh report path under an existing parent")
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("provide pinned local tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    start = time.perf_counter()
    results, examples = [], {}
    for seed, depth, direction, matches in jobs():
        result, record = inspect(seed, depth, direction, matches, tokenizer)
        results.append(result)
        if (
            record
            and seed == 3
            and depth == 8
            and direction.value == "North"
            and matches > 1
        ):
            examples[f"G{matches}-depth8"] = record["rows"][:3]
        if len(results) % 20 == 0:
            print(
                json.dumps(
                    {
                        "checked": len(results),
                        "statuses": dict(Counter(r["status"] for r in results)),
                    }
                ),
                flush=True,
            )
    accepted = [r for r in results if r["status"] == "accepted"]
    table = [
        {
            "depth": depth,
            "entailed_count": matches,
            "checked": sum(
                r["depth"] == depth and r["requested_entailed_count"] == matches
                for r in results
            ),
            "passed": sum(
                r["depth"] == depth and r["requested_entailed_count"] == matches
                for r in accepted
            ),
            "directions": sorted(
                {
                    r["direction"]
                    for r in accepted
                    if r["depth"] == depth and r["requested_entailed_count"] == matches
                }
            ),
            "families": len(
                {
                    r["family"]
                    for r in accepted
                    if r["depth"] == depth and r["requested_entailed_count"] == matches
                }
            ),
        }
        for depth in range(1, 11)
        for matches in range(1, 5)
    ]
    # A small 25% retention demonstration, not the 2,000-base training allocation:
    # retain six singleton plus two multi-answer bases at every depth.
    retained = []
    for depth in range(1, 11):
        singles = [
            r
            for r in accepted
            if r["seed"] == 3
            and r["depth"] == depth
            and r["actual_entailed_count"] == 1
        ]
        multi = [
            r
            for r in accepted
            if r["seed"] == 3
            and r["depth"] == depth
            and r["actual_entailed_count"] == 2 + (depth - 1) % 3
        ]
        if len(singles) >= 6 and len(multi) >= 2:
            retained.extend(singles[:6] + multi[:2])
    summary = {
        "attempts": len(results),
        "accepted": len(accepted),
        "statuses": dict(Counter(r["status"] for r in results)),
        "families": len({r["family"] for r in accepted}),
        "all_cells_supported": all(len(cell["directions"]) == 8 for cell in table),
        "max_target_tokens": {
            style: max((r["max_target_tokens"][style] for r in accepted), default=0)
            for style in ("natural", "symbolic")
        },
        "small_retention": {
            "bases": len(retained),
            "multi_bases": sum(r["actual_entailed_count"] > 1 for r in retained),
            "fraction": sum(r["actual_entailed_count"] > 1 for r in retained)
            / len(retained)
            if retained
            else None,
            "all_ten_depths": sorted({r["depth"] for r in retained}),
            "independent_families": len({r["family"] for r in retained}),
            "warning": "Demonstrates cardinality/depth compatibility only; not production family/source/quota feasibility.",
        },
    }
    report = {
        "status": "bounded-development-feasibility-not-production",
        "construction_version": VERSION,
        "sampling": "seed3 full 10x8x4 grid; seven additional seeds at depths1,4,8,10 in N/NE",
        "seconds": time.perf_counter() - start,
        "summary": summary,
        "table": table,
        "records": results,
        "retained_probe_ids": [r["base_id"] for r in retained],
        "examples": examples,
        "software_sha256": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("spatial/v2/entailment_balance_probe.py"),
                Path("spatial/v2/entailment_dataset.py"),
                Path("spatial/v2/entailment_evidence.py"),
                Path(__file__).with_name("audit_entailment_candidate.py"),
            )
        },
        "not_proven": [
            "500 multi-answer training bases within the fixed reserve",
            "train/dev/test family isolation and per-depth family floors",
            "70/15/10/5 source mixture and all existing coverage constraints",
            "controlled Count production adaptation",
            "full manual prose review",
        ],
        "production_allocation_changed": False,
        "replacement_generated": False,
        "sealed": False,
        "training_started": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
