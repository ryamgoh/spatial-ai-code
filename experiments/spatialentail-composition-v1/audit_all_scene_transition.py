"""Bounded scope-transition diagnostics, not replacement dataset generation."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_entailment_candidate import audit_record

from spatial.v2.composition_contracts import UnsupportedEvidence
from spatial.v2.composition_execution_benchmark import development_jobs
from spatial.v2.composition_feasibility import stable_int, token_measurement
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_contracts import VERSION as INTERFACE
from spatial.v2.entailment_contracts import Contract, analyze, encode
from spatial.v2.entailment_dataset import (
    build_attempt,
    measure_depth,
    menus_for,
    uniqueness_failure_depth,
)
from spatial.v2.entailment_evidence import (
    REFERENCE_VERSION,
    build_evidence,
    render_pair,
)
from spatial.v2.entailment_external import SelectionScope, build_views, verify_view
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import (
    SCOPE,
    VERSION,
    all_scene_problem,
    render_prompt,
)
from spatial.v2.serialization import tagged_dataclass_to_dict


def selected_records(path):
    """First retained record per query/requested-depth/source, train plus extrapolation."""
    selected = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] not in ("train", "extrapolation-test"):
                continue
            key = (row["query_kind"], row["requested_depth"], row["source"])
            selected.setdefault(key, row)
    return list(selected.values())


def inspect_record(old, tokenizer):
    old_problem = load_evidence(old).semantics.problem
    problem = all_scene_problem(old_problem)
    solver = QuestionSolver()
    previous = analyze(old_problem, solver=solver)
    semantics = analyze(problem, solver=solver)
    kind = old["query_kind"]
    result = {
        "base_id": old["base_id"],
        "query_kind": kind,
        "source": old["source"],
        "requested_depth": old["requested_depth"],
        "old_domain_size": old["semantics"]["domain_size"],
        "new_domain_size": len(problem.query.candidates) + (1 if kind == "count" else 0)
        if kind != "direction"
        else 8,
        "old_entailed": sorted(previous.entailed),
        "new_entailed": sorted(semantics.entailed),
        "entailed_changed": previous.entailed != semantics.entailed,
        "old_menu_label_changes": sum(
            encode(semantics, tuple(row["menu"]), row["contract"]).letters
            != row["answer"]
            for row in old["rows"]
        ),
        "context_irrelevance": "not replayed: retained v5 records omit context indices",
    }
    try:
        depth = measure_depth(problem, solver=solver)
        result["new_depth"] = depth["depth"]
        result["depth_matches_old"] = depth["depth"] == old["measurement"]["depth"]
    except UnsupportedEvidence as exc:
        result.update(new_depth=None, depth_reason=str(exc), depth_matches_old=False)
    try:
        evidence = build_evidence(problem, solver=solver)
    except UnsupportedEvidence as exc:
        result["reference_status"] = "unsupported"
        result["reference_reason"] = str(exc)
        return result, None
    categories = solver.negative_categories(problem)
    result["negative_categories"] = dict(Counter(categories.values()))
    result["unique_failure_depth"] = uniqueness_failure_depth(evidence)
    rows, tokens = [], []
    for i, menu in enumerate(menus_for(semantics, stable_int(old["base_id"]))):
        for contract in Contract:
            decision = encode(semantics, menu, contract)
            row = {
                "id": f"{old['base_id']}-m{i}-{contract.value}",
                "split": "scope-transition-diagnostics",
                "interface": INTERFACE,
                "prompt_version": VERSION,
                "scope_policy": SCOPE,
                "reference_version": REFERENCE_VERSION,
                "bundle_id": f"{old['base_id']}-m{i}",
                "contract": contract.value,
                "menu": list(menu),
                "prompt": render_prompt(problem, menu, contract),
                "answer": decision.letters,
                "references": render_pair(
                    evidence, menu, contract, solver=solver, restate_premises=True
                ),
                "negative_menu_categories": {
                    letter: categories[v]
                    for letter, v in zip("ABCD", menu)
                    if v in categories
                },
            }
            rows.append(row)
            tokens.extend(
                token_measurement(tokenizer, row, style) for style in row["references"]
            )
    replay = {
        "interface": INTERFACE,
        "prompt_version": VERSION,
        "scope_policy": SCOPE,
        "problem": tagged_dataclass_to_dict(problem),
        "semantics": {"entailed": sorted(semantics.entailed)},
        "evidence": tagged_dataclass_to_dict(evidence),
        "rows": rows,
    }
    result["independent_claim_counts"] = audit_record(replay)
    result["reference_status"] = "independently-replayed"
    result["tokens_admitted"] = all(t["admitted"] for t in tokens)
    result["max_target_tokens"] = max(t["target_tokens"] for t in tokens)
    result["primary_answers"] = {r["contract"]: r["answer"] for r in rows[:3]}
    return result, rows[:3]


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh report path under an existing parent")
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("pinned local tokenizer required")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    inputs = [args.candidate / "attempts.jsonl", args.external / "references.jsonl"]
    hashes = {str(p): digest(p) for p in inputs}
    records, examples = [], {}
    for old in selected_records(inputs[0]):
        result, rows = inspect_record(old, tokenizer)
        records.append(result)
        if rows and result["query_kind"] not in examples:
            examples[result["query_kind"]] = rows
        print(
            json.dumps({"synthetic_checked": len(records), "base_id": old["base_id"]}),
            flush=True,
        )
    # Reuse only existing development jobs to examine original context indices.
    contexts = []
    for _, entry, problem in development_jobs():
        result = build_attempt(
            entry.story,
            all_scene_problem(problem),
            None,
            tokenizer,
            prompt_version=VERSION,
        )
        contexts.append(
            {
                "story_id": entry.story.story_id,
                "query_kind": type(problem.query).__name__,
                "status": result["status"],
                "stage": result["stage"],
                "reason": result.get("reason"),
            }
        )
    external, seen, counts = [], set(), Counter()
    with inputs[1].open() as stream:
        for line in stream:
            old = json.loads(line)
            kind = old["query_kind"]
            if old["base_id"] in seen or counts[kind] >= 8:
                continue
            seen.add(old["base_id"])
            counts[kind] += 1
            source = {"id": old["base_id"], "text": old["source_text"]}
            historical = build_views(source, SelectionScope.ALL)
            new = build_views(source, SelectionScope.ALL, prompt_version=VERSION)
            for row in new["rows"]:
                verify_view(row)
            plain_record = {
                "interface": INTERFACE,
                "prompt_version": VERSION,
                "scope_policy": SCOPE,
                "problem": new["rows"][0]["problem"],
                "semantics": new["semantics"],
                "evidence": new["evidence"],
                "rows": new["rows"],
            }
            claims = audit_record(plain_record) if new["evidence"] else None
            external.append(
                {
                    "base_id": old["base_id"],
                    "query_kind": kind,
                    "entailed_changed": historical["semantics"]["entailed"]
                    != new["semantics"]["entailed"],
                    "labels_changed": [r["answer"] for r in historical["rows"]]
                    != [r["answer"] for r in new["rows"]],
                    "independent_claim_counts": claims,
                    "visible_alias_leak": any(
                        "Entity aliases" in r["prompt"] for r in new["rows"]
                    ),
                    "tokens_admitted": all(
                        token_measurement(tokenizer, row, style)["admitted"]
                        for row in new["rows"]
                        if row["references"]
                        for style in ("natural", "symbolic")
                    ),
                }
            )
            if all(counts[k] >= 8 for k in ("direction", "which", "count")):
                break
    if any(digest(Path(p)) != sha for p, sha in hashes.items()):
        raise ValueError("preserved input changed during diagnostics")
    report = {
        "status": "bounded-diagnostics-not-a-release",
        "prompt_version": VERSION,
        "scope_policy": SCOPE,
        "sampling": "first train/extrapolation record per query/requested-depth/source; first eight external bases per query",
        "input_sha256": hashes,
        "script_sha256": digest(Path(__file__)),
        "synthetic": records,
        "development_context_checks": contexts,
        "external": external,
        "plain_examples": examples,
        "summary": {
            "synthetic_bases": len(records),
            "by_query": dict(Counter(r["query_kind"] for r in records)),
            "changed_entailments_by_query": dict(
                Counter(r["query_kind"] for r in records if r["entailed_changed"])
            ),
            "old_menu_label_changes": sum(r["old_menu_label_changes"] for r in records),
            "depth_matches": sum(r["depth_matches_old"] for r in records),
            "reference_status": dict(Counter(r["reference_status"] for r in records)),
            "count_without_invariant_total": sum(
                not r["new_entailed"] for r in records if r["query_kind"] == "count"
            ),
            "selection_sizes": dict(
                Counter(
                    len(r["new_entailed"])
                    for r in records
                    if r["query_kind"] == "which"
                )
            ),
            "external_bases": len(external),
            "external_label_changes": sum(r["labels_changed"] for r in external),
        },
        "not_covered": [
            "quota feasibility",
            "full manual prose review",
            "full external regeneration",
            "retained-v5 context-index replay",
        ],
        "replacement_generated": False,
        "sealed": False,
        "training_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
