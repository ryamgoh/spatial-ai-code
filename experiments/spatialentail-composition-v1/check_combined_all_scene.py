"""Fixed-reserve combined admission and diagnostic retention; no dataset export."""

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

from audit_entailment_candidate import audit_record
from check_repaired_fixed_reserve import mapping_preflight

from spatial.v2 import composition_production as historical
from spatial.v2 import entailment_production as production
from spatial.v2 import entailment_retention as retention
from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.entailment_scene_protocol import SCOPE, VERSION, all_scene_problem
from spatial.v2.entailment_scene_recipes import VERSION as RECIPE_VERSION
from spatial.v2.entailment_scene_recipes import map_fixed_slots

TOKENIZER = None
POLICY = None


def initialize(tokenizer_path, policy):
    global TOKENIZER, POLICY
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    POLICY = policy


def diagnostic_policy(frozen):
    policy = production.all_scene_diagnostic_policy(frozen)
    selection_total = sum(
        n
        for cell, n in historical.quotas(production.common_policy(policy)).items()
        if "/which/" in cell
    )
    if selection_total % 4:
        raise ValueError("25% diagnostic requires an integral Selection base target")
    policy["diagnostic_multi_selection_bases"] = selection_total // 4
    return policy


def admit_job(job):
    entry, kind = job
    problem = entry.story.problems()[historical.KINDS.index(kind)]
    try:
        sizes = {
            "objects": len(entry.story.objects),
            "premises": len(entry.story.premise.operands),
            "candidates": len(entry.story.candidates),
        }
        if any(n > POLICY["scene_limits"][key] for key, n in sizes.items()):
            raise ValueError("unchanged scene limits exceeded")
        # Plain scenes cannot silently lose declared but unmentioned objects.
        all_scene_problem(problem)
        record = production.admit(entry, problem, POLICY, TOKENIZER, QuestionSolver())
    except ValueError as exc:
        record = {
            "interface": POLICY["interface"],
            "base_id": f"{entry.story.story_id}-{kind}",
            "story_id": entry.story.story_id,
            "family_id": entry.story.source_family,
            "split": entry.split,
            "source": historical.source_for(entry),
            "requested_depth": entry.depth,
            "status": "pre-admission-rejected",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    record.update(
        stratum=historical.stratum(entry, kind),
        source_stratum=f"{entry.split}/{kind}/{historical.source_for(entry)}",
        query_kind=kind,
    )
    if record["status"] == "accepted":
        try:
            # Replay independently before discarding bulky prompts and evidence.
            audit_record(record)
            if kind == "count" and entry.depth is not None:
                values = record["semantics"]["entailed"]
                if len(values) != 1 or type(values[0]) is not int:
                    raise ValueError("controlled Count lacks one entailed integer")
                record["controlled_count_value"] = values[0]
            record["query_key"] = query_profile(all_scene_problem(problem))["key"]
            record["independent_audit"] = "passed"
        except (ValueError, RuntimeError) as exc:
            record.update(
                status="verification-failed",
                reason=f"independent replay: {type(exc).__name__}: {exc}",
            )
    compact = historical.summary_record(record)
    compact.pop("graph", None)
    return compact


def source_shortfalls(records, policy):
    """Optimistic bound before dedup: a shortage already disproves retention."""
    supply = Counter(r["source_stratum"] for r in records if r["status"] == "accepted")
    return [
        {
            "cell": cell,
            "required": n,
            "verified_raw_supply": supply[cell],
            "shortfall": n - supply[cell],
        }
        for cell, n in historical.source_quotas(
            production.common_policy(policy)
        ).items()
        if not cell.endswith("/controlled") and supply[cell] < n
    ]


def multi_selection_upper_bound(records, policy):
    """Optimistic necessary bound from unchanged Unique substantive minima."""
    accepted = [
        r for r in records if r["status"] == "accepted" and r["query_kind"] == "which"
    ]
    substantive = Counter(
        r["source_stratum"]
        for r in accepted
        if any(
            row.get("menu_role") == "primary"
            and row.get("contract") == "unique-entailed"
            and row.get("substantive_depth_support") is True
            for row in r["rows"]
        )
    )
    multi = Counter(
        r["source_stratum"] for r in accepted if retention.multi_selection(r)
    )
    # Verification errors are not semantic rejections. Include them in both
    # favorable categories so they cannot spuriously prove a shortage.
    unresolved = Counter(
        r["source_stratum"]
        for r in records
        if r["query_kind"] == "which" and r["status"] == "verification-failed"
    )
    quotas = historical.source_quotas(production.common_policy(policy))
    bounds = []
    for role in historical.ROLES:
        prefix = f"{role}/which/"
        controlled = quotas[prefix + "controlled"]
        source_cells = [
            (key, n)
            for key, n in quotas.items()
            if key.startswith(prefix) and not key.endswith("/controlled")
        ]
        positive_supply = sum(
            min(n, substantive[key] + unresolved[key]) for key, n in source_cells
        )
        multi_supply = sum(
            min(n, multi[key] + unresolved[key]) for key, n in source_cells
        )
        floor = max(
            (
                rule.get("minimum_bases", 0)
                for rule in policy["coverage_minima"]
                if rule["match"].get("split") == role
                and rule["match"].get("query_kind") == "which"
                and rule["match"].get("contract") == "unique-entailed"
                and rule["match"].get("menu_role") == "primary"
                and rule["match"].get("substantive_depth_support") is True
            ),
            default=0,
        )
        # Every controlled multi-answer base consumes a slot that cannot count
        # toward the substantive Unique minimum. Assume unlimited good recipes.
        controlled_multi = max(0, controlled - floor + positive_supply)
        bounds.append(
            {
                "split": role,
                "controlled_selection_quota": controlled,
                "unique_substantive_minimum": floor,
                "optimistic_noncontrolled_substantive_supply": positive_supply,
                "optimistic_noncontrolled_multi_supply": multi_supply,
                "optimistic_total_multi_upper_bound": min(
                    controlled + sum(n for _, n in source_cells),
                    controlled_multi + multi_supply,
                ),
            }
        )
    return {
        "by_split": bounds,
        "overall_upper_bound": sum(
            r["optimistic_total_multi_upper_bound"] for r in bounds
        ),
        "target": policy["diagnostic_multi_selection_bases"],
        "deduplication_or_other_coverage_rejections_applied": False,
        "verification_failures_counted_optimistically": sum(unresolved.values()),
    }


def load_verified_source_records(path, expected_hash, policy):
    """Resume a complete prior source phase; never accept a partial inventory."""
    if not expected_hash or digest(path) != expected_hash:
        raise ValueError("source metadata hash mismatch")
    expected = {}
    for i, batch in enumerate(policy["source_batches"]):
        if batch.get("depth") is not None:
            continue
        if batch["generator"] not in (
            "random-world",
            "random-atoms",
            "sampled-targeted",
        ):
            raise ValueError("unsupported source inventory for metadata resume")
        source = (
            batch["generator"]
            if batch["generator"] != "sampled-targeted"
            else "targeted"
        )
        for j in batch.get("indices", range(batch["count"])):
            for kind in batch["query_kinds"]:
                expected[f"production-b{i}-s{j}-{kind}"] = (
                    batch["split"],
                    kind,
                    f"{batch['split']}/{kind}/{source}",
                )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [r["base_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("source metadata does not cover the exact fixed inventory")
    for record in records:
        if (
            record["split"],
            record["query_kind"],
            record["source_stratum"],
        ) != expected[record["base_id"]]:
            raise ValueError("source metadata stratum mismatch")
        if record["status"] == "accepted" and (
            record.get("independent_audit") != "passed"
            or not record.get("rows")
            or not record.get("tokens")
            or not all(t["admitted"] for t in record["tokens"])
        ):
            raise ValueError(
                "source candidate lacks completed admission/replay metadata"
            )
    return records


def compact_summary(records):
    accepted = [r for r in records if r["status"] == "accepted"]
    return {
        "checked_slots": len(records),
        "accepted_slots_before_dedup": len(accepted),
        "statuses": dict(Counter(r["status"] for r in records)),
        "status_by_source_query": [
            {"cell": key, "statuses": dict(Counter(r["status"] for r in group))}
            for key, group in grouped(records, "source_stratum")
        ],
        "rejection_reasons": [
            {"reason": reason, "count": n}
            for reason, n in Counter(
                r.get("reason", "unspecified")
                for r in records
                if r["status"] != "accepted"
            ).most_common()
        ],
        "max_target_tokens": {
            style: max(
                (
                    t["target_tokens"]
                    for r in accepted
                    for t in r["tokens"]
                    if t["style"] == style
                ),
                default=0,
            )
            for style in ("natural", "symbolic")
        },
        "multi_selection_supply_before_dedup": sum(
            retention.multi_selection(r) for r in accepted
        ),
    }


def load_controlled_prefix(path, expected_hash, jobs):
    """Reuse explicitly trusted current-run metadata, never historical checkpoints."""
    if not expected_hash or digest(path) != expected_hash:
        raise ValueError("controlled metadata hash mismatch")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) > len(jobs):
        raise ValueError("controlled metadata exceeds the fixed inventory")
    for record, (entry, kind) in zip(records, jobs):
        problem = all_scene_problem(
            entry.story.problems()[historical.KINDS.index(kind)]
        )
        expected = {
            "base_id": f"{entry.story.story_id}-{kind}",
            "story_id": entry.story.story_id,
            "family_id": entry.story.source_family,
            "split": entry.split,
            "requested_depth": entry.depth,
            "query_kind": kind,
            "stratum": historical.stratum(entry, kind),
            "source_stratum": f"{entry.split}/{kind}/controlled",
            "query_key": query_profile(problem)["key"],
            "prompt_version": VERSION,
            "scope_policy": SCOPE,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("controlled metadata does not match the current job prefix")
        if (
            record.get("status") != "accepted"
            or record.get("independent_audit") != "passed"
            or record.get("recipe") != entry.story.recipe
            or not record.get("rows")
            or not record.get("tokens")
            or not all(t["admitted"] for t in record["tokens"])
            or not all(
                row.get("prompt_version") == VERSION
                and row.get("scope_policy") == SCOPE
                and "decision_depth_support" in row
                for row in record["rows"]
            )
        ):
            raise ValueError("controlled candidate lacks current admission/replay metadata")
    return records


def grouped(records, field):
    groups = defaultdict(list)
    for record in records:
        groups[record[field]].append(record)
    return sorted(groups.items())


def retention_supply_shortfalls(candidates, policy):
    targets = retention.quota_targets(policy)
    counts = Counter(
        key for record in candidates for key in retention.quota_keys(record)
    )
    quota_deficits = [
        {
            "kind": kind,
            "cell": cell,
            "required": n,
            "verified_supply": counts[(kind, cell)],
        }
        for (kind, cell), n in targets.items()
        if counts[(kind, cell)] < n
    ]
    coverage_deficits = []
    for rule in retention.requirements(policy):
        found = [r for r in candidates if retention.matches(r, rule)]
        families = {historical.source_lineage(r["family_id"]) for r in found}
        if len(found) < rule.get("minimum_bases", 0) or len(families) < rule.get(
            "minimum_families", 0
        ):
            coverage_deficits.append(
                {
                    "requirement": rule,
                    "verified_bases": len(found),
                    "verified_families": len(families),
                }
            )
    return {
        "quota_shortfalls": quota_deficits,
        "coverage_shortfalls": coverage_deficits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-records", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--controlled-records", type=Path)
    parser.add_argument("--controlled-sha256")
    parser.add_argument(
        "--fresh", action="store_true", help="Reject all saved-result reuse options."
    )
    args = parser.parse_args()
    if args.fresh and any(
        value is not None
        for value in (
            args.source_records,
            args.source_sha256,
            args.controlled_records,
            args.controlled_sha256,
        )
    ):
        parser.error("--fresh forbids saved-result reuse")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh receipt under an existing directory")
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("provide the pinned local tokenizer")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1-8")
    start = perf_counter()
    config_hash = digest(args.config)
    frozen = json.loads(args.config.read_text())
    policy = diagnostic_policy(frozen)
    exclusions_path = args.config.parent / frozen["development_exclusions"]["path"]
    exclusion_hash = digest(exclusions_path)
    if exclusion_hash != frozen["development_exclusions"]["sha256"]:
        raise ValueError("frozen exclusion inventory changed")
    exclusions = json.loads(exclusions_path.read_text())
    common = production.common_policy(policy)
    entries = ()
    if args.source_records is None:
        entries = historical.schedule(common)
        historical.preflight(entries, exclusions)
    report = {
        "status": "combined-feasibility-running",
        "config_sha256": config_hash,
        "exclusion_sha256": exclusion_hash,
        "base_target": sum(policy["base_targets"].values()),
        "diagnostic_multi_selection_bases": policy["diagnostic_multi_selection_bases"],
        "multi_target_denominator": "all 3600 Selection bases, including E/F cases",
        "per_split_or_depth_multi_quota": False,
        "g2_g3_g4_allocation_approved": False,
        "new_seeds": False,
        "numeric_quota_relaxations": False,
        "approved_unique_decision_depth_revision": True,
        "unique_decision_depth_revision_splits": list(historical.ROLES),
        "replacement_generated": False,
        "sealed": False,
        "training_started": False,
        "milp_run": False,
        "full_retention_passed": False,
        "scheduled_stories": len(entries) if entries else None,
        "source_phase_reused": args.source_records is not None,
        "fresh_run": args.fresh,
        "prompt_version": policy["prompt_version"],
        "scope_policy": policy["scope_policy"],
        "controlled_recipe_version": RECIPE_VERSION,
    }
    print(
        json.dumps(
            {
                "original_preflight": "passed"
                if entries
                else "not-rerun-for-source-resume",
                "stories": len(entries) if entries else None,
                "diagnostic_multi_selection_bases": policy[
                    "diagnostic_multi_selection_bases"
                ],
                "fresh_run": args.fresh,
                "source_phase_reused": args.source_records is not None,
                "controlled_phase_reused": args.controlled_records is not None,
                "prompt_version": policy["prompt_version"],
                "scope_policy": policy["scope_policy"],
                "controlled_recipe_version": RECIPE_VERSION,
                "unique_depth_rules": [
                    rule
                    for rule in policy["coverage_minima"]
                    if rule["match"].get("query_kind") == "which"
                    and rule["match"].get("contract") == "unique-entailed"
                    and rule["match"].get("menu_role") == "primary"
                    and rule["match"].get("decision_depth_support") is True
                ],
            }
        ),
        flush=True,
    )
    spool = Path(tempfile.mkdtemp(prefix="spatialentail-combined-"))
    report["intermediate_metadata_directory"] = str(spool)
    all_records = []

    def run(jobs, phase, completed=()):
        phase_records = list(completed)
        with (
            (spool / f"{phase}.jsonl").open("w") as stream,
            get_context("spawn").Pool(
                args.workers,
                initializer=initialize,
                initargs=(str(args.tokenizer), policy),
                maxtasksperchild=100,
            ) as pool,
        ):
            for record in completed:
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            for record in pool.imap(admit_job, jobs[len(completed) :], chunksize=4):
                phase_records.append(record)
                stream.write(json.dumps(record) + "\n")
                if len(phase_records) % 200 == 0:
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                "phase": phase,
                                "checked": len(phase_records),
                                "total": len(jobs),
                                "statuses": dict(
                                    Counter(r["status"] for r in phase_records)
                                ),
                            }
                        ),
                        flush=True,
                    )
        all_records.extend(phase_records)
        return phase_records

    noncontrolled_jobs = [
        (entry, kind)
        for entry in entries
        if entry.depth is None
        for kind in entry.query_kinds
    ]
    if args.source_records is not None:
        noncontrolled = load_verified_source_records(
            args.source_records, args.source_sha256, policy
        )
        all_records.extend(noncontrolled)
        report["source_metadata_path"] = str(args.source_records)
        report["source_metadata_sha256"] = args.source_sha256
    else:
        noncontrolled = run(noncontrolled_jobs, "noncontrolled")
    deficits = source_shortfalls(noncontrolled, policy)
    bound = multi_selection_upper_bound(noncontrolled, policy)
    report["noncontrolled"] = compact_summary(noncontrolled)
    report["mandatory_noncontrolled_source_shortfalls"] = deficits
    report["multi_selection_necessary_bound"] = bound
    if deficits:
        report.update(
            status="combined-infeasible-for-current-fixed-recipes",
            reason="mandatory noncontrolled source quotas exceed verified raw supply",
            controlled_full_admission_run=False,
            milp_reason="unnecessary: mandatory source supply gate already fails",
            multi_25_percent_retention_tested=False,
        )
    elif bound["overall_upper_bound"] < bound["target"]:
        report.update(
            status="combined-infeasible-under-current-unique-substantive-minima",
            reason="optimistic multi-Selection bound is below the overall 25% target",
            controlled_full_admission_run=False,
            milp_reason="unnecessary: a necessary combined capacity bound fails",
            multi_25_percent_retention_tested=False,
            multi_25_percent_necessary_conditions_tested=True,
        )
    else:
        if not entries:
            entries = historical.schedule(common)
            historical.preflight(entries, exclusions)
            report["scheduled_stories"] = len(entries)
        # One diagnostic recipe per existing Selection slot, never extra views.
        overrides = {
            e.story.story_id: 2 + stable_int(f"{e.story.story_id}/cardinality") % 3
            for e in entries
            if e.depth is not None
            and "which" in e.query_kinds
            and stable_int(f"{e.story.story_id}/multi-supply") % 2 == 0
        }
        mapped, manifest = map_fixed_slots(entries, policy, selection_matches=overrides)
        _, rejected, _ = mapping_preflight(mapped, exclusions)
        # Full scene isolation covers all sources, including previously admitted ones.
        controlled_jobs = [
            (entry, kind)
            for entry in mapped
            if entry.depth is not None and entry.story.story_id not in rejected
            for kind in entry.query_kinds
            if len(entry.story.objects) <= policy["scene_limits"]["objects"]
            and len(entry.story.premise.operands) <= policy["scene_limits"]["premises"]
            and len(entry.story.candidates) <= policy["scene_limits"]["candidates"]
        ]
        report["controlled_mapping_slots"] = len(manifest)
        report["overlap_rejected_story_slots"] = len(rejected)
        print(
            json.dumps(
                {
                    "phase": "controlled-preflight",
                    "jobs": len(controlled_jobs),
                    "overlap_rejected_story_slots": len(rejected),
                }
            ),
            flush=True,
        )
        completed = (
            load_controlled_prefix(
                args.controlled_records, args.controlled_sha256, controlled_jobs
            )
            if args.controlled_records is not None
            else []
        )
        report["controlled_phase_reused_slots"] = len(completed)
        if args.controlled_records is not None:
            report["controlled_metadata_path"] = str(args.controlled_records)
            report["controlled_metadata_sha256"] = args.controlled_sha256
        print(
            json.dumps(
                {
                    "phase": "controlled-resume",
                    "reused": len(completed),
                    "remaining": len(controlled_jobs) - len(completed),
                }
            ),
            flush=True,
        )
        controlled = run(controlled_jobs, "controlled", completed)
        report["controlled"] = compact_summary(controlled)
        seen, candidates = set(), []
        for record in all_records:
            if record["status"] != "accepted" or record["story_id"] in rejected:
                continue
            if record["query_key"] not in seen:
                seen.add(record["query_key"])
                candidates.append(record)
        report["distinct_verified_candidates"] = len(candidates)
        report["candidate_supply_checks"] = retention_supply_shortfalls(
            candidates, policy
        )
        report["multi_25_percent_retention_tested"] = True
        selected_ids = None
        if any(report["candidate_supply_checks"].values()):
            report["milp_reason"] = "unnecessary: verified candidate supply gate fails"
        else:
            report["milp_run"] = True
            try:
                selected_ids = retention.select(candidates, policy)
            except RuntimeError as exc:
                report["retention_solver_error"] = str(exc)
        report["full_retention_passed"] = selected_ids is not None
        report["status"] = (
            "combined-diagnostic-retention-passed-not-production"
            if selected_ids is not None
            else "combined-diagnostic-retention-inconclusive"
            if "retention_solver_error" in report
            else "combined-diagnostic-retention-infeasible"
        )
        if selected_ids is not None:
            selected = [r for r in candidates if r["base_id"] in selected_ids]
            retention.validate(candidates, selected, policy)
            report["retained_base_ids"] = sorted(selected_ids)
            report["retained_multi_selection_bases"] = sum(
                retention.multi_selection(r) for r in selected
            )
            report["retained_counts"] = dict(Counter(r["stratum"] for r in selected))
    if digest(args.config) != config_hash or digest(exclusions_path) != exclusion_hash:
        raise ValueError("fixed inputs changed during feasibility audit")
    report.update(
        seconds=perf_counter() - start,
        software_sha256={
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("spatial/v2/entailment_retention.py"),
                Path("spatial/v2/entailment_production.py"),
                Path("spatial/v2/entailment_scene_recipes.py"),
                Path("spatial/v2/entailment_dataset.py"),
                Path(__file__).with_name("audit_entailment_candidate.py"),
            )
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
