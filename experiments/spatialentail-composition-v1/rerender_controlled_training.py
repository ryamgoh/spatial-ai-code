"""Re-render verified questions without changing a single problem or evidence object."""

import argparse
import json
import tempfile
from collections import Counter
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path

import run_all_controlled_prototype as tokenizer_loader
from audit_entailment_candidate import replay_counterexample, replay_paths
from build_controlled_training import export
from review_unique_pretraining import digest

from spatial.v2.composition_feasibility import token_measurement
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_dataset import audit_row
from spatial.v2.entailment_evidence import SPATIAL_PROSE_VERSION, render_pair
from spatial.v2.entailment_readable_v3 import audit_plan, make_plan
from spatial.v2.entailment_replacement_allocation import validate_records
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.solver import CountQuery, DirectionQuery

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]


def preserved_projection(record):
    """Only reference text, its version, and token measurements may change."""
    result = deepcopy(record)
    result.pop("tokens", None)
    for row in result["rows"]:
        row.pop("references")
        row.pop("reference_version")
    return result


def independent_plan_check(evidence, plan):
    problem = evidence.semantics.problem
    for value in plan["values"]:
        support = evidence.positive_support[value]
        if isinstance(problem.query, CountQuery):
            for member in support["memberships"]:
                replay_paths(
                    problem,
                    member["paths"],
                    member["candidate"],
                    problem.query.reference,
                    next(iter(problem.query.directions)).value,
                    member["qualifies"],
                )
        else:
            replay_paths(
                problem,
                support["paths"],
                problem.query.target
                if isinstance(problem.query, DirectionQuery)
                else value,
                problem.query.reference,
                value
                if isinstance(problem.query, DirectionQuery)
                else next(iter(problem.query.directions)).value,
                True,
            )
    for value in plan["witness_keys"]:
        replay_counterexample(problem, value, evidence.semantics.counterexamples[value])
    audit_plan(evidence, plan)


def rerender(original):
    result = {"id": original["base_id"]}
    try:
        if len(original["rows"]) != 1 or original["independent_audit"] != "passed":
            raise ValueError("input lacks one independently verified selected menu")
        evidence = load_evidence(original)
        plan = make_plan(evidence)
        independent_plan_check(evidence, plan)
        record = deepcopy(original)
        row = record["rows"][0]
        solver = QuestionSolver()
        row["references"] = render_pair(
            evidence,
            tuple(row["menu"]),
            Contract.UNIQUE,
            solver=solver,
            restate_premises=True,
            spatial_prose=True,
        )
        row["reference_version"] = SPATIAL_PROSE_VERSION
        audit_row(row, evidence, solver=solver)
        record["tokens"] = [
            token_measurement(tokenizer_loader.TOKENIZER, row, style)
            for style in ("natural", "symbolic")
        ]
        if not all(t["admitted"] for t in record["tokens"]):
            raise ValueError("new references exceed the unchanged token limits")
        if preserved_projection(record) != preserved_projection(original):
            raise ValueError("re-render changed non-reference content")
        result.update(
            status="passed",
            record=record,
            exact_ambiguity_cases=len(plan["outcomes"]),
            grouped_negative_claims=sum(map(len, plan["negative_groups"].values())),
            original_long_paths=sum(len(p["indices"]) > 1 for p in plan["paths"]),
            displayed_chains=len(plan["chains"]),
        )
    except (ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    return result


def write_readable_reports(path, report, records):
    examples = {}
    for record in records:
        row = record["rows"][0]
        key = (record["query_kind"], row["escape_reason"] or "A-D", "first")
        examples.setdefault(key, row)
        if row["depth"] == 8 and row["answer"] in "ABCD":
            examples.setdefault((record["query_kind"], "A-D", "L8"), row)
        if row["sole_entailed_with_contingent"]:
            examples.setdefault(
                (record["query_kind"], "sole with contingent alternative", "first"), row
            )
    lines = [
        "# Spatial-Prose Training Examples",
        "",
        "Actual v3 references for unchanged v2 questions. Both styles use the same checked evidence.",
        "",
    ]
    for (kind, case, depth), row in examples.items():
        lines.extend(
            [
                f"## {kind}: {case} ({depth})",
                "",
                f"ID: `{row['id']}`",
                "",
                row["prompt"],
                "",
                "### Natural",
                "",
                row["references"]["natural"],
                "",
                "### Symbolic",
                "",
                row["references"]["symbolic"],
                "",
            ]
        )
    path.with_suffix(".examples.md").write_text("\n".join(lines))
    lines = [
        "# Spatial-Prose Re-Rendering Audit",
        "",
        "Status: passed. Reference-only update; no question generation, sealing or training.",
        "",
        "- All 6,000 original records are preserved except references, reference version and token measurements.",
        "- Problems, menus, labels, evidence, IDs, canonical keys, depth metadata and allocation are unchanged.",
        "- Selected paths and counterexamples were replayed; shared-chain coverage and all Which alternatives were checked.",
        "- Current row audits recheck entailment and reference matching using the existing solver. This is not a new dataset selection.",
        "- Both style exports were read back and checked against the verified records.",
        "- All original training and external payload fingerprints are unchanged.",
        "",
        "## Token Comparison",
        "",
        "| Style | Previous mean | New mean | Previous max | New max |",
        "|---|---:|---:|---:|---:|",
    ]
    for style, values in report["tokens"].items():
        lines.append(
            f"| {style} | {values['old_mean']:.1f} | {values['new_mean']:.1f} | {values['old_max']} | {values['new_max']} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            (
                "Natural wording distinguishes 'further east/west' (one axis) from exact compass directions. "
                "Direct facts stay direct; Count reuses cited chains; ambiguous Direction/Count uses two "
                "saved arrangements with different outcomes; Which rejections are grouped without "
                "dropping any other object. Contradiction explanations use the same conflict certificate."
            ),
            "",
            (
                "The prose remains systematically generated, not independently human-authored. "
                "No claim of new structural diversity, new spatial coverage, or GPU/training validation is made."
            ),
            "",
            f"Candidate: `{report['output']}`",
            f"Examples: `{path.with_suffix('.examples.md').name}`",
            "",
        ]
    )
    path.with_suffix(".md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    manifest_path = args.receipt.with_suffix(".manifest.json")
    outputs = [
        args.output,
        args.receipt,
        manifest_path,
        args.receipt.with_suffix(".md"),
        args.receipt.with_suffix(".examples.md"),
    ]
    if any(p.exists() for p in outputs):
        parser.error("existing re-render output: do not overwrite or rerun")
    if (
        not args.output.parent.is_dir()
        or not args.receipt.parent.is_dir()
        or not args.tokenizer.is_dir()
        or args.tokenizer.name != tokenizer_loader.REVISION
    ):
        parser.error("existing parents and pinned local tokenizer required")
    source_manifest_path = args.source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    prior_audit_path = EXP / "CONTROLLED-TRAINING-EXPORT-AUDIT-v2.json"
    prior_audit = json.loads(prior_audit_path.read_text())
    if (
        prior_audit["status"] != "passed"
        or digest(source_manifest_path) != prior_audit["candidate_manifest_sha256"]
    ):
        raise ValueError("source is not the independently audited v2 candidate")
    for name, expected in source_manifest["payload_sha256"].items():
        if digest(args.source / name) != expected:
            raise ValueError("source payload fingerprint mismatch")
    policy = json.loads(
        (EXP / "CONTROLLED-TRAINING-BUILD-v2.manifest.json").read_text()
    )["policy"]
    with (args.source / "evidence.jsonl").open() as stream:
        original = [json.loads(line) for line in stream]
    validate_records(original, policy)
    protected = [
        prior_audit_path,
        EXP / "CONTROLLED-TRAINING-BUILD-v2.manifest.json",
        EXP / "entailment-config-frozen-v1.json",
        EXP / "unique-only-selection-v1/selection.json",
        *[
            p
            for d in (
                "unique-training-candidate-v1",
                "unique-training-candidate-v2",
                "spatialmap-unique-v1",
            )
            for p in (EXP / d).iterdir()
            if p.suffix == ".jsonl" or p.name == "manifest.json"
        ],
    ]
    before = {str(p): digest(p) for p in protected}
    sources = [
        Path(__file__),
        EXP / "build_controlled_training.py",
        EXP / "audit_entailment_candidate.py",
        *[
            ROOT / "spatial/v2" / name
            for name in (
                "entailment_readable_v3.py",
                "entailment_evidence.py",
                "entailment_dataset.py",
                "composition_scenarios.py",
                "composition_readable.py",
            )
        ],
    ]
    hashes = {str(p): digest(p) for p in sources}
    spool = Path(tempfile.mkdtemp(prefix="spatial-reference-v3-"))
    manifest = {
        "version": SPATIAL_PROSE_VERSION,
        "source": str(args.source),
        "records": 6000,
        "allowed_changes": ["rows[].references", "rows[].reference_version", "tokens"],
        "protected_sha256": before,
        "source_sha256": hashes,
        "tokenizer": str(args.tokenizer),
        "spool": str(spool),
        "question_generation": False,
        "training_started": False,
    }
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    args.receipt.write_text(
        json.dumps({"status": "running", "spool": str(spool)}) + "\n"
    )
    results = []
    with (
        get_context("spawn").Pool(
            args.workers,
            initializer=tokenizer_loader.initialize,
            initargs=(str(args.tokenizer),),
        ) as pool,
        (spool / "results.jsonl").open("x") as stream,
    ):
        for result in pool.imap(rerender, original):
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            results.append(result)
            if len(results) % 250 == 0:
                print(
                    json.dumps(
                        {
                            "checked": len(results),
                            "statuses": dict(Counter(r["status"] for r in results)),
                        }
                    ),
                    flush=True,
                )
    report = {
        "status": "re-render-failed-no-export",
        "manifest_sha256": digest(manifest_path),
        "spool": str(spool),
        "results_sha256": digest(spool / "results.jsonl"),
        "statuses": dict(Counter(r["status"] for r in results)),
        "failures": [r for r in results if r["status"] != "passed"],
    }
    try:
        if report["failures"]:
            raise ValueError("one or more reference updates failed verification")
        records = [r["record"] for r in results]
        for old, new in zip(original, records, strict=True):
            if preserved_projection(old) != preserved_projection(new):
                raise ValueError("unapproved dataset change")
        validate_records(records, policy)
        for p, sha in {**before, **hashes}.items():
            if digest(Path(p)) != sha:
                raise ValueError(f"protected input or source changed: {p}")
        stats = {}
        for style in ("natural", "symbolic"):
            old_tokens = [
                t["target_tokens"]
                for r in original
                for t in r["tokens"]
                if t["style"] == style
            ]
            new_tokens = [
                t["target_tokens"]
                for r in records
                for t in r["tokens"]
                if t["style"] == style
            ]
            stats[style] = {
                "old_mean": sum(old_tokens) / 6000,
                "new_mean": sum(new_tokens) / 6000,
                "old_max": max(old_tokens),
                "new_max": max(new_tokens),
            }
        report.update(
            status="references-rerendered-and-audited",
            tokens=stats,
            output=str(args.output),
            preserved_inputs_unchanged=True,
            non_reference_content_unchanged=True,
            evidence_unchanged=True,
            exact_ambiguity_with_two_cases=sum(
                r["exact_ambiguity_cases"] == 2 for r in results
            ),
        )
        export(records, args.output, report, manifest_path)
        report["candidate_manifest_sha256"] = digest(args.output / "manifest.json")
        write_readable_reports(args.receipt, report, records)
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status="re-render-failed-no-export", reason=str(exc))
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
