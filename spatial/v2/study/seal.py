"""Audit and seal a candidate study release from validated batch judge receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from spatial.v2.composition import HOLDOUT_COMPOSITIONS
from spatial.v2.context_budget import ContextBudget, validate_context_admission
from spatial.v2.grading import encode_menu_answer, resolve_answer
from spatial.v2.hitl.review import (
    FinalDisposition,
    HumanAppeal,
    ReviewVerdict,
    load_judge_run,
    load_review_packet,
    route_review_decisions,
)
from spatial.v2.solver import SpatialSolverV2
from spatial.v2.structure import canonical_structure_profile
from spatial.v2.study import role_path
from spatial.v2.study.build import _hash, _keys
from spatial.v2.symbolic_trace_codec import score_symbolic_training_trace
from spatial.v2.text import SpatialTextAdapter

BENCH_ROLES = {"bench-core", "bench-comp", "bench-composition", "bench-depth"}


def audit_release(release: Path) -> dict:
    manifest = json.loads((release / "manifest.json").read_text())
    if (
        manifest.get("schema") != "spatialentail-study-release-v1"
        or manifest.get("status") == "engineering-only"
    ):
        raise ValueError("production audit requires a study release")
    if not (BENCH_ROLES | {"train", "dev"}) <= set(manifest["quotas"]):
        raise ValueError("production release lacks required evaluation roles")
    oracle = SpatialSolverV2(backend="z3")
    adapter = SpatialTextAdapter()
    seen = [set(), set(), set()]
    summary = {}
    skeletons_by_role = {}
    for role, quotas in manifest["quotas"].items():
        path = role_path(release, role)
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["artifacts"][role]["sha256"]
        ):
            raise ValueError(f"{role} hash mismatch")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        groups = {}
        for row in rows:
            validate_context_admission(row)
            m = row["metadata"]
            if m.get("engineering_only") is not False or m["study_role"] != role:
                raise ValueError("row release role/engineering flag disagrees")
            if ContextBudget(**m["context_budget"]) != ContextBudget(
                **manifest["context_budget"]
            ):
                raise ValueError("row tokenizer envelope differs")
            groups.setdefault(m["base_id"], []).append(row)
        skeletons_by_role[role] = set()
        counts = Counter()
        query_counts = Counter()
        semantic_counts = Counter()
        positions = Counter()
        directions = Counter()
        for variants in groups.values():
            first = variants[0]
            m = first["metadata"]
            counts[m["benchmark_cell"]] += 1
            skeleton = m["skeleton_family"]
            skeletons_by_role[role].add(skeleton)
            if (
                role in {"train", "dev", "bench-comp"}
                and skeleton in HOLDOUT_COMPOSITIONS
            ):
                raise ValueError("held-out skeleton leaked into matched role")
            if role == "bench-composition" and skeleton not in HOLDOUT_COMPOSITIONS:
                raise ValueError("composition holdout lacks reserved skeleton")
            if m["skeleton_signature"] != _hash(["declared-skeleton-v1", skeleton]):
                raise ValueError("skeleton signature mismatch")
            depths = [
                value
                for key, value in m["difficulty"].items()
                if key
                in {
                    "x_depth",
                    "y_depth",
                    "max_membership_x_depth",
                    "max_membership_y_depth",
                }
                and isinstance(value, int)
            ]
            if role == "bench-depth":
                try:
                    expected_depth = int(m["benchmark_cell"].rsplit("-", 1)[1])
                except (ValueError, IndexError) as exc:
                    raise ValueError("invalid depth stratum") from exc
                if (
                    not 3 <= expected_depth <= 8
                    or max(depths, default=0) != expected_depth
                    or m["process_evaluation"] != "not-applicable"
                ):
                    raise ValueError(
                        "depth benchmark bounds or process contract disagrees"
                    )
            if role in {"train", "dev"} and max(depths, default=0) > 4:
                raise ValueError("deep training example leaked")
            keys = _keys(first)
            for previous, key in zip(seen, keys, strict=True):
                if key in previous:
                    raise ValueError("cross-role or within-role instance duplication")
                previous.add(key)
            packages = {
                (v["metadata"]["trace_format"], v["metadata"]["supervision_arm"])
                for v in variants
            }
            expected = (
                {("natural", "answer-only")}
                if role == "bench-depth"
                else {
                    ("natural", "answer-only"),
                    ("natural", "checked-trace"),
                    ("symbolic", "checked-trace"),
                }
            )
            if packages != expected or len(variants) != len(expected):
                raise ValueError("paired packages are incomplete")
            for v in variants:
                if (
                    _keys(v) != keys
                    or v["metadata"]["possible_values"] != m["possible_values"]
                    or v["metadata"]["oracle_letters"] != m["oracle_letters"]
                ):
                    raise ValueError("paired problem or gold changed")
            prompt = next(
                message["content"]
                for message in first["messages"]
                if message["role"] == "user"
            )
            parsed = adapter.parse(prompt)
            profile = canonical_structure_profile(parsed.problem)
            if (
                profile != m["structure_profile"]
                or _hash(profile) != m["structure_signature"]
            ):
                raise ValueError(
                    "persisted structural signature does not match visible problem"
                )
            analysis = oracle.analyze(
                parsed.problem,
                include_witnesses=not bool(m.get("oracle_only")),
            )
            if analysis.error or not analysis.consistent:
                raise ValueError("persisted prompt fails semantic audit")
            expected_answer = encode_menu_answer(
                resolve_answer(analysis, m["answer_mode"]), parsed.options
            )
            if sorted(expected_answer.letters) != m["oracle_letters"]:
                raise ValueError("persisted gold answer disagrees with semantic oracle")
            for variant in variants:
                completion = variant["messages"][-1]["content"]
                if variant["metadata"]["trace_format"] == "symbolic":
                    trace = completion.split("<think>\n", 1)[1].split(
                        "\n</think>\n", 1
                    )[0]
                    if not score_symbolic_training_trace(
                        parsed.problem, trace, expected_answer
                    ).fully_valid:
                        raise ValueError("persisted Symbolic trace fails replay")
                if completion.rsplit("Answer:", 1)[-1].strip() != ", ".join(
                    sorted(expected_answer.letters)
                ):
                    raise ValueError("persisted answer footer disagrees")
            query_counts[m["query_kind"]] += 1
            semantic_counts[m["resolution_status"]] += 1
            positions.update(m["oracle_letters"])
            directions.update(
                [m["target_direction"] or m["query_direction"] or "not-applicable"]
            )
        if dict(counts) != quotas:
            raise ValueError(f"{role} quotas disagree")
        if len(groups) != manifest["artifacts"][role]["bases"]:
            raise ValueError("base count mismatch")
        summary[role] = {
            "bases": len(groups),
            "cells": dict(counts),
            "query_counts": dict(query_counts),
            "semantic_counts": dict(semantic_counts),
            "answer_positions": dict(positions),
            "directions": dict(directions),
        }
    if skeletons_by_role["bench-composition"] & (
        skeletons_by_role["train"] | skeletons_by_role["dev"]
    ):
        raise ValueError("train/dev/heldout skeleton overlap")
    # Nested train pools must preserve complete paired source rows.
    full = {
        row["id"]: _hash(row)
        for row in (
            json.loads(x) for x in role_path(release, "train").read_text().splitlines()
        )
    }
    for role, record in manifest["artifacts"].items():
        if not role.startswith("train-"):
            continue
        path = role_path(release, role)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("nested hash mismatch")
        nested_rows = [json.loads(line) for line in path.read_text().splitlines()]
        ids = [row["id"] for row in nested_rows]
        if len(ids) != len(set(ids)) or len(ids) != record["rows"]:
            raise ValueError("nested row count or uniqueness mismatch")
        nested_groups = {}
        for row in nested_rows:
            if full.get(row["id"]) != _hash(row):
                raise ValueError("nested pool changed source row")
            nested_groups.setdefault(row["metadata"]["base_id"], []).append(row)
        if len(nested_groups) != record["bases"]:
            raise ValueError("nested base count mismatch")
        fraction = float(role.split("-", 1)[1])
        cell_counts = Counter()
        for variants in nested_groups.values():
            if len(variants) != 3 or {
                (r["metadata"]["trace_format"], r["metadata"]["supervision_arm"])
                for r in variants
            } != {
                ("natural", "checked-trace"),
                ("symbolic", "checked-trace"),
                ("natural", "answer-only"),
            }:
                raise ValueError("nested pool lacks complete paired groups")
            cell_counts[variants[0]["metadata"]["benchmark_cell"]] += 1
        expected = {
            cell: int(count * fraction)
            for cell, count in manifest["quotas"]["train"].items()
        }
        if (
            any(
                count * fraction != int(count * fraction)
                for count in manifest["quotas"]["train"].values()
            )
            or dict(cell_counts) != expected
        ):
            raise ValueError("nested cell allocation differs from frozen fraction")
    return {
        "schema": "spatialentail-release-audit-v1",
        "release_artifacts_sha256": {
            r: v["sha256"] for r, v in manifest["artifacts"].items()
        },
        "roles": summary,
        "overlap": 0,
    }


def seal_release(
    release: Path, review_index: Path, *, verify_only: bool = False
) -> dict:
    audit = audit_release(release)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != (
        "sealed" if verify_only else "candidate-awaiting-quality-review"
    ):
        raise ValueError("release must be an unsealed candidate")
    index = json.loads(review_index.read_text())
    if index.get("schema") != "spatialentail-release-review-v1":
        raise ValueError("invalid review index")
    if index["release_artifacts_sha256"] != audit["release_artifacts_sha256"]:
        raise ValueError("review covers different release artifacts")
    natural = {}
    for role in manifest["quotas"]:
        for line in role_path(release, role).read_text().splitlines():
            row = json.loads(line)
            if (
                row["metadata"]["trace_format"] == "natural"
                and row["metadata"]["supervision_arm"] == "checked-trace"
            ):
                natural[row["id"]] = row
    reviewed_role_counts = Counter()
    reviewed_counts = Counter()
    reviewed = set()
    covered = set()
    counts = Counter()
    receipt_hashes = {}
    for batch in index["batches"]:
        paths = {
            name: review_index.parent / batch[name]
            for name in ("packet", "crosswalk", "deepseek", "gpt")
        }
        packet = load_review_packet(paths["packet"])
        if (
            packet.registry_fingerprint != manifest["config_sha256"]
            or packet.rubric_version != "quality-judge-v1"
        ):
            raise ValueError("review packet uses a different release or rubric")
        crosswalk = json.loads(paths["crosswalk"].read_text())
        if crosswalk["packet_fingerprint"] != packet.fingerprint:
            raise ValueError("crosswalk fingerprint mismatch")
        first = load_judge_run(
            paths["deepseek"], packet, expected_model="DeepSeek-V4-Pro"
        )
        second = load_judge_run(paths["gpt"], packet, expected_model="GPT-5.5")
        reuse = batch.get("receipt_reuse")
        if reuse is not None:
            if not isinstance(reuse, dict) or set(reuse) != {
                "schema",
                "basis",
                "source_packet",
                "source_deepseek",
                "source_gpt",
            }:
                raise ValueError("invalid receipt reuse record")
            if (
                reuse["schema"] != "spatialentail-receipt-reuse-v1"
                or reuse["basis"] != "identical-review-items"
            ):
                raise ValueError("unsupported receipt reuse record")
            source_paths = {
                name: review_index.parent / reuse[name]
                for name in ("source_packet", "source_deepseek", "source_gpt")
            }
            prior_packet = load_review_packet(source_paths["source_packet"])
            if (
                prior_packet.packet_id != packet.packet_id
                or prior_packet.rubric_version != packet.rubric_version
                or prior_packet.items != packet.items
            ):
                raise ValueError("reused review items differ from current packet")
            prior_first = load_judge_run(
                source_paths["source_deepseek"],
                prior_packet,
                expected_model="DeepSeek-V4-Pro",
            )
            prior_second = load_judge_run(
                source_paths["source_gpt"],
                prior_packet,
                expected_model="GPT-5.5",
            )
            for prior, rebound in ((prior_first, first), (prior_second, second)):
                expected = prior.to_dict()
                expected["packet_fingerprint"] = packet.fingerprint
                if expected != rebound.to_dict():
                    raise ValueError("rebound judge receipt differs from source")
            paths.update(source_paths)
        routing = route_review_decisions(packet, first, second)
        appeals = {}
        if batch.get("appeals"):
            appeal_path = review_index.parent / batch["appeals"]
            paths["appeals"] = appeal_path
            for line in appeal_path.read_text().splitlines():
                value = json.loads(line)
                rebuilt = HumanAppeal.create(
                    packet,
                    value["review_item_id"],
                    value["verdict"],
                    reviewer=value["reviewer"],
                    reason_codes=tuple(value["reason_codes"]),
                    note=value["note"],
                )
                if (
                    value["packet_id"] != packet.packet_id
                    or value["packet_fingerprint"] != packet.fingerprint
                    or value["payload_hash"] != rebuilt.payload_hash
                    or not value.get("submitted_at")
                ):
                    raise ValueError("human appeal payload differs from frozen batch")
                if rebuilt.review_item_id in appeals:
                    raise ValueError("duplicate human appeal")
                appeals[rebuilt.review_item_id] = rebuilt
            eligible = {
                r.item.id
                for r in routing.items
                if r.disposition is FinalDisposition.HUMAN_APPEAL
            }
            if set(appeals) - eligible:
                raise ValueError("human appeal is not a double-rejection item")
        for routed in routing.items:
            source_id = crosswalk["items"][routed.item.id]["source_id"]
            if source_id in reviewed or source_id not in natural:
                raise ValueError("invalid or duplicate review source")
            row = natural[source_id]
            prompt = next(m["content"] for m in row["messages"] if m["role"] == "user")
            trace = (
                row["messages"][-1]["content"]
                .split("<think>\n", 1)[1]
                .split("\n</think>\n", 1)[0]
            )
            if (prompt, trace) != (routed.item.prompt, routed.item.natural_trace):
                raise ValueError("review payload does not match release row")
            reviewed_role_counts[row["metadata"]["study_role"]] += 1
            reviewed_counts[
                (row["metadata"]["study_role"], row["metadata"]["benchmark_cell"])
            ] += 1
            reviewed.add(source_id)
            covered.add(
                (row["metadata"]["study_role"], row["metadata"]["benchmark_cell"])
            )
            disposition = routed.disposition
            if (
                disposition is FinalDisposition.HUMAN_APPEAL
                and routed.item.id in appeals
            ):
                disposition = (
                    FinalDisposition.ACCEPT
                    if appeals[routed.item.id].verdict is ReviewVerdict.APPROVE
                    else FinalDisposition.HOLD_OUT
                )
            counts[disposition.value] += 1
        receipt_hashes.update(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths.values()
            }
        )
    required = {
        (role, cell)
        for role, quotas in manifest["quotas"].items()
        if role != "bench-depth"
        for cell in quotas
    }
    if any(
        reviewed_counts[key] < min(5, manifest["quotas"][key[0]][key[1]])
        for key in required
    ):
        raise ValueError("review sample lacks five-per-cell minimum")
    if not required <= covered:
        raise ValueError("quality review lacks role/cell coverage")
    if any(
        reviewed_role_counts[role]
        < max(
            math.ceil(0.02 * sum(quotas.values())),
            sum(min(5, n) for n in quotas.values()),
        )
        for role, quotas in manifest["quotas"].items()
        if role != "bench-depth"
    ):
        raise ValueError("review sample lacks per-role minimum")
    required_minimum = sum(
        max(
            math.ceil(0.02 * sum(quotas.values())),
            sum(min(5, count) for count in quotas.values()),
        )
        for role, quotas in manifest["quotas"].items()
        if role != "bench-depth"
    )
    if index["minimum_review_items"] < required_minimum:
        raise ValueError("review index weakens the declared sample minimum")
    if len(reviewed) < index["minimum_review_items"]:
        raise ValueError("review sample below frozen minimum")
    if counts.get("human-appeal", 0):
        raise ValueError("unresolved human appeals; seal refused")
    if counts.get("hold-out", 0):
        raise ValueError("held-out quality rows require replacement; seal refused")
    if counts.get("accept", 0) != len(reviewed):
        raise ValueError("quality review did not accept all included sampled rows")
    audit["quality"] = {
        "counts": dict(counts),
        "sample_items": len(reviewed),
        "receipt_sha256": receipt_hashes,
        "claim": "automated presentation screen; no population human-acceptability estimate",
    }
    if verify_only:
        return audit
    # Bundle reviewed inputs inside the release so verification survives transfer.
    import shutil

    quality_dir = release / "quality"
    quality_dir.mkdir()
    bundled = json.loads(review_index.read_text())
    for batch in bundled["batches"]:
        for key in ("packet", "crosswalk", "deepseek", "gpt", "appeals"):
            if key not in batch:
                continue
            source = review_index.parent / batch[key]
            name = source.name
            if (quality_dir / name).exists():
                raise ValueError("duplicate bundled review filename")
            shutil.copyfile(source, quality_dir / name)
            batch[key] = name
        if batch.get("receipt_reuse"):
            reuse = batch["receipt_reuse"]
            for key in ("source_packet", "source_deepseek", "source_gpt"):
                source = review_index.parent / reuse[key]
                name = source.name
                if (quality_dir / name).exists():
                    raise ValueError("duplicate bundled review filename")
                shutil.copyfile(source, quality_dir / name)
                reuse[key] = name
    (quality_dir / "index.json").write_text(
        json.dumps(bundled, indent=2, sort_keys=True) + "\n"
    )
    (release / "release-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        status="sealed",
        release_audit_sha256=hashlib.sha256(
            (release / "release-audit.json").read_bytes()
        ).hexdigest(),
        review_index_sha256=hashlib.sha256(
            (quality_dir / "index.json").read_bytes()
        ).hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--review-index", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            seal_release(args.release, args.review_index)
            if args.review_index
            else audit_release(args.release),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
