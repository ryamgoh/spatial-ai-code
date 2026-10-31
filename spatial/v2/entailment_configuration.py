"""Compile a review-only entailment configuration from approved numeric origins."""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from spatial.v2.composition_execution import atomic_json
from spatial.v2.composition_training import digest
from spatial.v2.entailment_contracts import VERSION
from spatial.v2.entailment_coverage import derive
from spatial.v2.entailment_production import SCHEMA, plan
from spatial.v2.entailment_retention import RETENTION


def draft_policy(approved):
    coverage = derive(approved)
    result = deepcopy(approved)
    result.update(
        schema=SCHEMA,
        interface=VERSION,
        status="draft",
        coverage_review={
            "status": "pending",
            "explicit_gate_changes": coverage["explicit_gate_changes"],
            "negative_minima": "proposed reuse of approved semantic minima",
        },
        coverage_minima=coverage["rules"],
        retention=RETENTION,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path)
    parser.add_argument("--approve-reviewed-coverage", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh draft path under an existing parent")
    origin = args.approved_config.resolve()
    policy = draft_policy(json.loads(origin.read_text()))
    exclusion = (origin.parent / policy["development_exclusions"]["path"]).resolve()
    if digest(exclusion) != policy["development_exclusions"]["sha256"]:
        parser.error("original exclusion inventory hash mismatch")
    policy["development_exclusions"]["path"] = os.path.relpath(
        exclusion, args.output.parent.resolve()
    )
    policy["numeric_origin"] = {
        "path": os.path.relpath(origin, args.output.parent.resolve()),
        "sha256": digest(origin),
    }
    if args.approve_reviewed_coverage:
        if args.support_audit is None:
            parser.error("a completed training-source support audit is required")
        audit = json.loads(args.support_audit.read_text())
        if (
            audit.get("interface") != VERSION
            or not audit.get("training_source_only")
            or audit.get("held_out_questions_inspected") != 0
            or audit.get("statuses") != {"accepted": audit.get("attempted_questions")}
            or not audit.get("attempted_questions")
        ):
            parser.error("support audit has unresolved failures or incorrect scope")
        critical = (
            "solver.py",
            "entailment_contracts.py",
            "entailment_evidence.py",
            "entailment_dataset.py",
            "composition_scene_sampler.py",
            "composition_targeted_sampler.py",
            "composition_training.py",
        )
        for name in critical:
            recorded = [
                expected
                for path, expected in audit["sources_sha256"].items()
                if Path(path).name == name
            ]
            if recorded != [digest(Path(__file__).parent / name)]:
                parser.error(f"support-audit semantic source changed: {name}")
        policy.update(
            status="frozen",
            support_audit={
                "path": os.path.relpath(
                    args.support_audit.resolve(), args.output.parent.resolve()
                ),
                "sha256": digest(args.support_audit),
                "tokenizer_sha256": audit["tokenizer_sha256"],
                "scope": "bounded training-source support, not a completed dataset",
            },
            frozen_sources_sha256={
                p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        )
        policy["coverage_review"].update(
            status="approved",
            negative_minima="user approved 30/6/12 per query",
            rare_selection_difference="user approved 10/3/5 bases and families",
            selection_scope="user approved all five nonreference objects externally",
        )
    planned = plan(policy)
    if args.approve_reviewed_coverage and planned["blockers"]:
        parser.error("; ".join(planned["blockers"]))
    if not args.approve_reviewed_coverage and not planned["blockers"]:
        parser.error("a draft must not be executable before review")
    atomic_json(args.output, policy)
    print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
