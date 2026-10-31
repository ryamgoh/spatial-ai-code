"""Build the all-controlled replacement, failing closed on fixed supply and audits."""

import argparse
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import run_all_controlled_prototype as tokenizer_loader
from audit_selected_explanation_depth import replay
from review_unique_pretraining import digest

from spatial.v2 import composition_production as historical
from spatial.v2 import entailment_production as production
from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_feasibility import stable_int, token_measurement
from spatial.v2.composition_pilot import story_key
from spatial.v2.composition_training import messages
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_dataset import build_attempt
from spatial.v2.entailment_evidence import CONCISE_VERSION, render_pair
from spatial.v2.entailment_generator import SceneRequest, build_scene
from spatial.v2.entailment_replacement_allocation import (
    Slot,
    policy_from,
    schedule,
    validate_records,
)
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.entailment_scene_protocol import SHORT_VERSION
from spatial.v2.entailment_scene_recipes import map_fixed_slots
from spatial.v2.entailment_training_selection import answer_category
from spatial.v2.entailment_unique_study import training_allocation_policy
from spatial.v2.solver import And, Direction, direction_constraint

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]


def request_for(slot, alternative):
    size = "contextual"
    if (
        slot.span == 1
        and not slot.contingent
        and alternative == 0
        and slot.index % 5 == 0
    ):
        size = "compact" if slot.index % 10 == 0 else "compact-extra"
    return SceneRequest(
        slot.span,
        tuple(Direction)[slot.index % 8],
        slot.kind,
        slot.case,
        slot.count_value,
        variant=(slot.index + alternative) % 2,
        size=size,
    )


def construct(slot, alternative):
    seed, identity = slot.seed(alternative), slot.identity(alternative)
    anchor_request = SceneRequest(10, Direction.NORTH, "which", "sole")
    anchor, _ = build_scene(
        anchor_request, identity=f"anchor-{seed}", family="anchor", structure_seed=seed
    )
    anchor_key = story_key(anchor.objects, anchor.premise)
    family = f"controlled-v2/{anchor_key}"
    story, problem = build_scene(
        request_for(slot, alternative),
        identity=identity,
        family=family,
        structure_seed=seed,
        contingent=slot.contingent,
    )
    return story, problem, anchor_key


def held_out_inventory(frozen, old_records, selected_ids):
    evaluation = [
        r for r in old_records if r["base_id"] in selected_ids and r["split"] != "train"
    ]
    if len(evaluation) != 4800:
        raise ValueError("fixed evaluation inventory must contain 4800 bases")
    story_ids = {r["story_id"] for r in evaluation}
    old_policy = training_allocation_policy(frozen)
    originals = tuple(
        e
        for e in historical.schedule(production.common_policy(old_policy))
        if e.story.story_id in story_ids
    )
    overrides = {
        e.story.story_id: 2 + stable_int(f"{e.story.story_id}/cardinality") % 3
        for e in originals
        if e.depth is not None
        and "which" in e.query_kinds
        and stable_int(f"{e.story.story_id}/multi-supply") % 2 == 0
    }
    mapped, lineage = map_fixed_slots(
        originals, old_policy, selection_matches=overrides
    )
    scenes = {
        story_key(e.story.objects, e.story.premise) for e in (*originals, *mapped)
    }
    anchors = {r["repair_anchor"].rsplit("/", 1)[-1] for r in lineage}
    anchors.update(r["family_id"].rsplit("/", 1)[-1] for r in evaluation)
    with (EXP / "spatialmap-unique-v1/evidence.jsonl").open() as stream:
        for line in stream:
            problem = json.loads(line)["semantics"]["problem"]
            premise = And(
                tuple(
                    direction_constraint(
                        a["subject"], a["reference"], Direction(a["allowed"][0])
                    )
                    for a in problem["premise"]["operands"]
                )
            )
            scenes.add(story_key(tuple(problem["objects"]), premise))
    return {
        "base_ids": sorted(r["base_id"] for r in evaluation),
        "query_keys": sorted(r["query_key"] for r in evaluation),
        "scene_keys": sorted(scenes),
        "anchor_keys": sorted(anchors),
    }


def preflight(slots, held_out):
    groups = defaultdict(list)
    for slot in slots:
        groups[
            (slot.stratum, slot.category, slot.span, slot.count_value, slot.contingent)
        ].append(slot)
    query_keys = set(held_out["query_keys"])
    excluded_scenes = set(held_out["scene_keys"])
    excluded_anchors = set(held_out["anchor_keys"])
    selected, shortages, reasons = [], [], Counter()
    for group, required in groups.items():
        found = []
        for alternative in range(3):
            for slot in required:
                if len(found) == len(required):
                    break
                story, problem, anchor = construct(slot, alternative)
                key = query_profile(problem)["key"]
                scene = story_key(story.objects, story.premise)
                reasons["attempts"] += 1
                if key in query_keys:
                    reasons["duplicate_or_evaluation_query"] += 1
                elif (
                    scene in excluded_scenes
                    or anchor in excluded_scenes
                    or anchor in excluded_anchors
                ):
                    reasons["held_out_scene_or_anchor"] += 1
                else:
                    query_keys.add(key)
                    found.append(
                        {
                            "slot": asdict(slot),
                            "alternative": alternative,
                            "query_key": key,
                            "scene_key": scene,
                            "anchor_key": anchor,
                        }
                    )
            if len(found) == len(required):
                break
        selected.extend(found)
        if len(found) != len(required):
            shortages.append(
                {"group": group, "required": len(required), "found": len(found)}
            )
        print(
            json.dumps(
                {
                    "preflight_selected": len(selected),
                    "attempts": reasons["attempts"],
                    "short_groups": len(shortages),
                }
            ),
            flush=True,
        )
    return selected, {"counts": dict(reasons), "shortages": shortages}


def admit(job):
    slot = Slot(**job["slot"])
    result = {"id": slot.identity(job["alternative"]), "job": job}
    try:
        story, problem, anchor = construct(slot, job["alternative"])
        if (
            query_profile(problem)["key"] != job["query_key"]
            or anchor != job["anchor_key"]
        ):
            raise ValueError("construction differs from preflight")
        record = build_attempt(
            story,
            problem,
            slot.depth,
            tokenizer_loader.TOKENIZER,
            omission_percent=100 if slot.category == "unique-unlisted" else 0,
            prompt_version=SHORT_VERSION,
        )
        result["record"] = record
        if record["stage"] != "complete":
            raise ValueError(f"{record['status']}: {record.get('reason')}")
        evidence = load_evidence(record)
        candidates = [r for r in record["rows"] if answer_category(r) == slot.category]
        if len(candidates) != 1:
            raise ValueError("intended semantic case did not yield one requested menu")
        row = candidates[0]
        if (
            slot.kind == "count"
            and slot.depth is not None
            and evidence.semantics.entailed != {slot.count_value}
        ):
            raise ValueError("constructed Count value changed")
        if slot.contingent and not row["sole_entailed_with_contingent"]:
            raise ValueError("required contingent alternative was not preserved")
        row["references"] = render_pair(
            evidence,
            tuple(row["menu"]),
            Contract.UNIQUE,
            restate_premises=True,
            concise=True,
        )
        row["reference_version"] = CONCISE_VERSION
        pattern = "homogeneous" if slot.span == 1 else "axis-split"
        direction = request_for(slot, job["alternative"]).direction.value
        metadata = {
            "split": "train",
            "source": "controlled",
            "proof_pattern": pattern,
            "direction": direction,
            "direction_group": "cardinal"
            if direction in historical.CARDINALS
            else "diagonal",
        }
        row.update(**metadata, nonhomogeneous_proof=slot.span > 1)
        record.update(
            **metadata,
            rows=[row],
            stratum=slot.stratum,
            source_stratum=f"train/{slot.kind}/controlled",
            query_key=job["query_key"],
            scene_key=job["scene_key"],
            anchor_key=anchor,
            tokens=[
                token_measurement(tokenizer_loader.TOKENIZER, row, style)
                for style in ("natural", "symbolic")
            ],
        )
        if slot.kind == "count" and slot.depth is not None:
            record["controlled_count_value"] = slot.count_value
        record["status"] = (
            "accepted"
            if all(t["admitted"] for t in record["tokens"])
            else "token-rejected"
        )
        record = replay(record)
        if (
            slot.depth is not None
            and not record["rows"][0]["selected_explanation_depth_support"]
        ):
            raise ValueError("selected explanation lacks the scheduled depth")
        result.update(record=record, status=record["status"])
    except (ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    return result


def export(records, output, report, manifest_path):
    staging = Path(tempfile.mkdtemp(prefix="controlled-training-export-"))
    handles = {
        name: (staging / name).open("x")
        for name in (
            "evidence.jsonl",
            "references.jsonl",
            "natural-training.jsonl",
            "symbolic-training.jsonl",
        )
    }
    try:
        for record in records:
            row = record["rows"][0]
            handles["evidence.jsonl"].write(json.dumps(record) + "\n")
            handles["references.jsonl"].write(json.dumps(row) + "\n")
            for style in ("natural", "symbolic"):
                example = {
                    "id": row["id"],
                    "base_id": record["base_id"],
                    "messages": [
                        *messages(row, style),
                        {"role": "assistant", "content": row["references"][style]},
                    ],
                }
                handles[f"{style}-training.jsonl"].write(json.dumps(example) + "\n")
    finally:
        for stream in handles.values():
            stream.close()
    # Read every exported line back and bind style messages to the checked row.
    for name in handles:
        with (staging / name).open() as stream:
            count = 0
            for record, line in zip(records, stream, strict=True):
                value, row = json.loads(line), record["rows"][0]
                if name == "evidence.jsonl":
                    if value != record:
                        raise ValueError("evidence export mismatch")
                elif name == "references.jsonl":
                    if value != row:
                        raise ValueError("reference export mismatch")
                else:
                    style = name.split("-")[0]
                    if value["messages"] != [
                        *messages(row, style),
                        {"role": "assistant", "content": row["references"][style]},
                    ]:
                        raise ValueError("style message export mismatch")
                    if (
                        value["id"] != row["id"]
                        or value["base_id"] != record["base_id"]
                    ):
                        raise ValueError("style identity mismatch")
                count += 1
            if count != 6000:
                raise ValueError("export line count differs")
    payloads = {name: digest(staging / name) for name in handles}
    manifest = {
        "status": "audited-unsealed-all-controlled-training",
        "bases": 6000,
        "examples_per_style": 6000,
        "sealed": False,
        "training_started": False,
        "preparation_manifest": str(manifest_path),
        "preparation_sha256": digest(manifest_path),
        "payload_sha256": payloads,
        "audit_summary": report,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (staging / "README.md").write_text(
        "# All-Controlled Training Candidate\n\n"
        "6,000 distinct questions, one selected menu each; matched Natural/Symbolic references.\n"
        "Per query: 1,200 A-D, 600 E, 200 F. Existing evaluation selections are unchanged.\n"
        "Evidence, selected depths, quotas, overlap exclusions, token limits and exported messages were checked.\n"
        "Unsealed; no model training has started. See manifest.json for fingerprints and audit details.\n"
    )
    if output.exists():
        raise ValueError("refusing to replace an existing training directory")
    shutil.copytree(staging, output)
    if any(digest(output / name) != sha for name, sha in payloads.items()):
        raise ValueError("published payload hash mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    manifest_path = args.receipt.with_suffix(".manifest.json")
    if args.output.exists() or args.receipt.exists() or manifest_path.exists():
        parser.error("existing run/output: no overwrite or automatic rerun")
    if (
        not args.output.parent.is_dir()
        or not args.receipt.parent.is_dir()
        or args.tokenizer.name != tokenizer_loader.REVISION
    ):
        parser.error("existing output parents and pinned tokenizer required")
    frozen_path = EXP / "entailment-config-frozen-v1.json"
    frozen = json.loads(frozen_path.read_text())
    policy = policy_from(frozen)
    planned = schedule(policy)
    old_receipt_path = EXP / "COMBINED-ALL-SCENE-FEASIBILITY-v4.json"
    old_receipt = json.loads(old_receipt_path.read_text())
    old_pool_path = (
        Path(old_receipt["intermediate_metadata_directory"]) / "candidates.jsonl"
    )
    if (
        digest(old_pool_path) != old_receipt["candidates_sha256"]
        or digest(frozen_path) != old_receipt["config_sha256"]
    ):
        raise ValueError("original frozen input fingerprint mismatch")
    selection_path = EXP / "unique-only-selection-v1/selection.json"
    selected_ids = set(json.loads(selection_path.read_text())["selected_base_ids"])
    with old_pool_path.open() as stream:
        old_records = [json.loads(line) for line in stream]
    protected = [
        frozen_path,
        old_receipt_path,
        selection_path,
        *[
            p
            for d in ("unique-training-candidate-v1", "spatialmap-unique-v1")
            for p in (EXP / d).iterdir()
            if p.suffix == ".jsonl" or p.name == "manifest.json"
        ],
    ]
    protected_hashes = {str(p): digest(p) for p in protected}
    sources = [
        Path(__file__),
        EXP / "audit_entailment_candidate.py",
        EXP / "audit_selected_explanation_depth.py",
        *[
            ROOT / "spatial/v2" / name
            for name in (
                "entailment_replacement_allocation.py",
                "entailment_generator.py",
                "entailment_dataset.py",
                "entailment_evidence.py",
                "entailment_scene_protocol.py",
                "entailment_contracts.py",
                "structure.py",
                "solver.py",
            )
        ],
    ]
    source_hashes = {str(p): digest(p) for p in sources}
    spool = Path(tempfile.mkdtemp(prefix="spatial-controlled-training-"))
    manifest = {
        "policy": policy,
        "slots": [asdict(s) for s in planned],
        "construction_cap": 18000,
        "solver_admission_cap": 6000,
        "selection_rule": "First nonoverlapping canonical questions in each fixed quota group; at most three predeclared candidates per slot; no refill.",
        "protected_sha256": protected_hashes,
        "source_sha256": source_hashes,
        "tokenizer": str(args.tokenizer),
        "workers": args.workers,
        "spool": str(spool),
        "output": str(args.output),
    }
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    args.receipt.write_text(
        json.dumps({"status": "preparing", "spool": str(spool)}) + "\n"
    )
    held_out = held_out_inventory(frozen, old_records, selected_ids)
    (spool / "held-out.json").write_text(json.dumps(held_out) + "\n")
    jobs, preflight_report = preflight(planned, held_out)
    (spool / "selected-constructions.json").write_text(json.dumps(jobs) + "\n")
    report = {
        "status": "preflight-complete",
        "preflight": preflight_report,
        "manifest_sha256": digest(manifest_path),
        "spool": str(spool),
        "evaluation_bases": len(held_out["base_ids"]),
        "held_out_sha256": digest(spool / "held-out.json"),
        "selected_constructions_sha256": digest(spool / "selected-constructions.json"),
    }
    if preflight_report["shortages"]:
        report["status"] = "insufficient-distinct-supply-no-refill"
        args.receipt.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    results = []
    with (
        get_context("spawn").Pool(
            args.workers,
            initializer=tokenizer_loader.initialize,
            initargs=(str(args.tokenizer),),
        ) as pool,
        (spool / "admissions.jsonl").open("x") as stream,
    ):
        for result in pool.imap(admit, jobs):
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            results.append(result)
            if len(results) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "admitted": len(results),
                            "statuses": dict(Counter(r["status"] for r in results)),
                        }
                    ),
                    flush=True,
                )
    report.update(
        statuses=dict(Counter(r["status"] for r in results)),
        admissions_sha256=digest(spool / "admissions.jsonl"),
        failures=[
            {k: v for k, v in r.items() if k != "record"}
            for r in results
            if r["status"] != "accepted"
        ],
    )
    try:
        if report["failures"]:
            raise ValueError(
                "one or more selected constructions failed admission; no refill"
            )
        records = [r["record"] for r in results]
        validate_records(records, policy)
        if {r["query_key"] for r in records} & set(held_out["query_keys"]):
            raise ValueError("evaluation query overlap")
        if {r["scene_key"] for r in records} & set(held_out["scene_keys"]):
            raise ValueError("evaluation scene overlap")
        for path, expected in {**protected_hashes, **source_hashes}.items():
            if digest(Path(path)) != expected:
                raise ValueError(f"input/source changed: {path}")
        size_counts = defaultdict(Counter)
        for record in records:
            n = len(record["problem"]["premise"]["operands"])
            size_counts[record["query_kind"], n][
                "E" if record["rows"][0]["answer"] == "E" else "non-E"
            ] += 1
        report.update(
            status="audited-unsealed-training-export",
            preserved_inputs_unchanged=True,
            independent_audit="passed",
            answer_counts={
                kind: dict(
                    Counter(
                        answer_category(r["rows"][0])
                        for r in records
                        if r["query_kind"] == kind
                    )
                )
                for kind in ("direction", "which", "count")
            },
            size_counts=[
                {"query": k, "premises": n, **counts}
                for (k, n), counts in sorted(size_counts.items())
            ],
            max_target_tokens={
                style: max(
                    t["target_tokens"]
                    for r in records
                    for t in r["tokens"]
                    if t["style"] == style
                )
                for style in ("natural", "symbolic")
            },
        )
        export(records, args.output, report, manifest_path)
        report["output"] = str(args.output)
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status="audit-blocked-no-export", reason=str(exc))
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("size_counts", "failures")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
