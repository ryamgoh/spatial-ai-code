"""Build exact role quotas, paired supervision, and disjoint corpus artifacts.

This command produces an audited candidate release. Model-judge screening and
operator sign-off are separate gates; generation never labels a release frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from spatial.v2.benchmark_registry import load_benchmark_registry
from spatial.v2.composition import HOLDOUT_COMPOSITIONS, TRAIN_COMPOSITIONS
from spatial.v2.context_budget import ContextBudget, admit_row_group, load_tokenizer
from spatial.v2.generation import GenerationPolicy, SpatialGeneratorV2
from spatial.v2.study import write_role_texts
from spatial.v2.supervision_controls import SupervisionArm, apply_supervision_arm
from spatial.v2.symbolic_trace_codec import score_symbolic_training_trace


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_rows(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _keys(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row["metadata"]
    prompt = next(
        message["content"] for message in row["messages"] if message["role"] == "user"
    )
    return metadata["base_id"], metadata["structure_signature"], _hash(prompt)


def build_release(
    config_path: Path, output: Path, *, tokenizer: Any = None
) -> dict[str, Any]:
    source = yaml.safe_load(config_path.read_text())
    expected = {
        "version",
        "name",
        "engineering_only",
        "seed",
        "context_budget",
        "registries",
        "contamination_files",
        "roles",
        "train_fractions",
        "attempts_per_base",
    }
    if (
        not isinstance(source, dict)
        or set(source) != expected
        or source["version"] != 1
    ):
        raise ValueError("invalid study release configuration")
    if type(source["engineering_only"]) is not bool:
        raise ValueError("engineering_only must be Boolean")
    if output.exists():
        raise FileExistsError(f"release output already exists: {output}")
    roles = source["roles"]
    allowed_roles = {
        "train",
        "dev",
        "bench-core",
        "bench-comp",
        "bench-composition",
        "bench-depth",
    }
    if (
        not isinstance(roles, dict)
        or set(roles) - allowed_roles
        or not {"train", "dev"} <= set(roles)
    ):
        raise ValueError("invalid release roles")
    fractions = source["train_fractions"]
    if (
        not fractions
        or any(type(f) not in (float, int) or not 0 < f <= 1 for f in fractions)
        or fractions != sorted(set(fractions))
    ):
        raise ValueError("train_fractions must be increasing unique fractions in (0,1]")
    if fractions[-1] != 1:
        raise ValueError("train_fractions must include the full pool")
    attempt_limit = source["attempts_per_base"]
    if type(attempt_limit) is not int or attempt_limit < 1:
        raise ValueError("attempts_per_base must be positive")
    base = config_path.parent
    budget = ContextBudget(**source["context_budget"])
    if not budget.tokenizer_revision:
        raise ValueError("release tokenizer revision must be pinned")
    if tokenizer is None:
        tokenizer = load_tokenizer(budget)
    cells = {}
    registry_hashes = {}
    for filename in source["registries"]:
        registry = load_benchmark_registry(base / filename)
        registry_hashes[filename] = registry.fingerprint
        for cell in registry.cells:
            if cell.id in cells:
                raise ValueError(f"duplicate registry cell: {cell.id}")
            cells[cell.id] = cell
    keys_seen = [set(), set(), set()]
    contamination_receipts = {}
    for filename in source["contamination_files"]:
        path = base / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing contamination ledger input: {path}")
        contamination_receipts[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in _read_rows(path):
            for known, key in zip(keys_seen, _keys(row), strict=True):
                known.add(key)
    # Work files are resumable proposals, never a completed release. Bind every
    # checkpoint to config, registries, contamination ledger and implementation.
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("spatial/v2").rglob("*.py"))
    }
    binding = _hash([source, registry_hashes, contamination_receipts, sources])
    work = output.with_name(output.name + "-work")
    work.mkdir(parents=True, exist_ok=True)
    artifact_rows = {}
    quotas = {}
    rejections = Counter()
    skeletons_by_role: dict[str, set[str]] = {}
    for role, requests in roles.items():
        groups = []
        quotas[role] = {}
        skeletons_by_role[role] = set()
        for request in requests:
            if not isinstance(request, dict) or set(request) - {
                "cell",
                "count",
                "policy",
                "stratum",
            }:
                raise ValueError("invalid role cell request")
            count = request.get("count")
            if type(count) is not int or count < 1:
                raise ValueError("role count must be a positive integer")
            if ("cell" in request) == ("policy" in request):
                raise ValueError("request needs exactly one cell or policy")
            if "cell" in request:
                cell = cells[request["cell"]]
                policy = cell.generation_policy()
                stratum = cell.id
                skeleton = policy.composition or "core:" + cell.id
            else:
                policy = GenerationPolicy(**request["policy"])
                stratum = request["stratum"]
                skeleton = policy.composition or "depth:" + policy.query_kind.value
            if policy.oracle_only and role != "bench-depth":
                raise ValueError("oracle-only generation is restricted to bench-depth")
            if stratum in quotas[role]:
                raise ValueError(f"duplicate role stratum: {stratum}")
            if (
                role in {"train", "dev", "bench-core", "bench-comp"}
                and policy.composition is not None
                and policy.composition not in TRAIN_COMPOSITIONS
            ):
                raise ValueError("held-out composition appears in matched role")
            if (
                role == "bench-composition"
                and policy.composition not in HOLDOUT_COMPOSITIONS
            ):
                raise ValueError(
                    "composition holdout role requires a reserved skeleton"
                )
            depth = max(policy.min_axis_depth, policy.min_membership_depth)
            maximum = max(
                depth,
                policy.max_axis_depth or depth,
                policy.max_membership_depth or depth,
            )
            if role in {"train", "dev"} and maximum > 4:
                raise ValueError("training and development depth must not exceed four")
            if role == "bench-depth" and (not 3 <= depth <= 8 or maximum != depth):
                raise ValueError(
                    "depth benchmark requires matched 3/4 anchors or extrapolative 5–8"
                )
            skeletons_by_role[role].add(skeleton)
            seed = int(_hash([source["seed"], role, stratum])[:8], 16)
            generator = SpatialGeneratorV2(seed)
            checkpoint = work / (role + "--" + stratum + ".json")
            accepted = []
            if checkpoint.exists():
                cached = json.loads(checkpoint.read_text())
                if cached.get("binding") != binding:
                    raise ValueError(
                        "stale release work checkpoint; use a new output version"
                    )
                accepted = [tuple(group) for group in cached["groups"]]
                if len(accepted) != count:
                    raise ValueError("incomplete cached stratum")
                for group in accepted:
                    for row in group:
                        from spatial.v2.context_budget import validate_context_admission

                        validate_context_admission(row)
                    keys = _keys(group[0])
                    if any(
                        key in known for known, key in zip(keys_seen, keys, strict=True)
                    ):
                        raise ValueError("checkpoint overlaps earlier release roles")
                    for known, key in zip(keys_seen, keys, strict=True):
                        known.add(key)
            for _ in range(count * attempt_limit):
                if len(accepted) == count:
                    break
                try:
                    sample = generator.generate(policy, max_attempts=1)
                except RuntimeError:
                    rejections[role + ":generation"] += 1
                    continue
                measured = max(
                    (
                        value
                        for name, value in sample.difficulty.items()
                        if name
                        in {
                            "x_depth",
                            "y_depth",
                            "max_membership_x_depth",
                            "max_membership_y_depth",
                        }
                        and isinstance(value, int)
                    ),
                    default=0,
                )
                if role in {"train", "dev"} and measured > 4:
                    rejections[role + ":achieved_depth_exceeded"] += 1
                    continue
                if role == "bench-depth" and measured != depth:
                    rejections[role + ":achieved_depth_mismatch"] += 1
                    continue
                natural = sample.with_trace("natural").as_sft_row()
                keys = _keys(natural)
                if (
                    keys[0] in keys_seen[0]
                    or keys[2] in keys_seen[2]
                    or (not source["engineering_only"] and keys[1] in keys_seen[1])
                ):
                    rejections[role + ":duplicate"] += 1
                    continue
                if role == "bench-depth":
                    rows = [
                        natural
                        if policy.oracle_only
                        else apply_supervision_arm(
                            natural, SupervisionArm.ANSWER_ONLY
                        )
                    ]
                else:
                    symbolic_sample = sample.with_trace("symbolic")
                    if not score_symbolic_training_trace(
                        sample.problem, symbolic_sample.trace, sample.menu_answer
                    ).fully_valid:
                        raise ValueError("generated Symbolic evidence failed replay")
                    rows = [
                        natural,
                        symbolic_sample.as_sft_row(),
                        apply_supervision_arm(natural, SupervisionArm.ANSWER_ONLY),
                    ]
                for row in rows:
                    row["metadata"].update(
                        benchmark_cell=stratum,
                        study_role=role,
                        skeleton_signature=_hash(["declared-skeleton-v1", skeleton]),
                        skeleton_family=skeleton,
                        engineering_only=source["engineering_only"],
                        process_evaluation="not-applicable"
                        if role == "bench-depth"
                        else "symbolic-replay",
                    )
                admission = admit_row_group(rows, tokenizer, budget)
                if not admission.accepted:
                    for reason in admission.rejection_reasons:
                        rejections[role + ":" + reason] += 1
                    continue
                for known, key in zip(keys_seen, keys, strict=True):
                    known.add(key)
                accepted.append(admission.rows)
            if len(accepted) != count:
                raise RuntimeError(
                    f"{role}/{stratum}: admitted {len(accepted)}/{count}; "
                    f"quota failed without output; rejections={dict(rejections)}"
                )
            # Keyed random ranks avoid ordering nested pools by generator accidents.
            accepted.sort(
                key=lambda group: _hash(
                    [source["seed"], stratum, group[0]["metadata"]["base_id"]]
                )
            )
            if not checkpoint.exists():
                checkpoint.write_text(
                    json.dumps({"binding": binding, "groups": accepted}, sort_keys=True)
                    + "\n"
                )
            groups.extend(accepted)
            quotas[role][stratum] = count
            print(f"admitted {role}/{stratum}: {count} bases", flush=True)
        artifact_rows[role] = [row for group in groups for row in group]
        if role == "train":
            by_cell = {}
            for group in groups:
                by_cell.setdefault(group[0]["metadata"]["benchmark_cell"], []).append(
                    group
                )
            for fraction in fractions[:-1]:
                subset = []
                for stratum, cell_groups in by_cell.items():
                    requested = len(cell_groups) * fraction
                    if requested != int(requested) or requested < 1:
                        raise ValueError(
                            "nested fractions must preserve complete positive cell allocations"
                        )
                    subset.extend(
                        row for group in cell_groups[: int(requested)] for row in group
                    )
                artifact_rows[f"train-{fraction:g}"] = subset
    holdout = skeletons_by_role.get("bench-composition", set())
    if holdout & (
        skeletons_by_role.get("train", set()) | skeletons_by_role.get("dev", set())
    ):
        raise ValueError("composition skeleton overlap")
    manifest = {
        "schema": "spatialentail-study-release-v1",
        "name": source["name"],
        "status": "engineering-only"
        if source["engineering_only"]
        else "candidate-awaiting-quality-review",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "registry_sha256": registry_hashes,
        "contamination_inputs_sha256": contamination_receipts,
        "context_budget": source["context_budget"],
        "quotas": quotas,
        "rejections": dict(rejections),
        "instance_signature": "visible-problem-structure-v1; direction-sensitive; conservative fallback",
        "skeleton_signature": "declared-skeleton-v1; construction-family reservation, not logical equivalence",
        "skeletons": {
            role: sorted(values) for role, values in skeletons_by_role.items()
        },
        "artifacts": {},
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest["software_revision"] = revision
    manifest["software_dirty"] = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    )
    tracked_changes = subprocess.run(
        ["git", "diff", "HEAD", "--", "spatial/v2", "experiments/15-v2-ablation"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    manifest["tracked_implementation_diff_sha256"] = hashlib.sha256(
        tracked_changes.encode()
    ).hexdigest()
    manifest["implementation_sources_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("spatial/v2").rglob("*.py"))
    }
    serialized = {}
    for role, rows in artifact_rows.items():
        serialized[role] = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
        manifest["artifacts"][role] = {
            "rows": len(rows),
            "bases": len({row["metadata"]["base_id"] for row in rows}),
            "unique_structures": len(
                {row["metadata"]["structure_signature"] for row in rows}
            ),
            "target_tokens": sum(
                row["metadata"]["target_model_tokens"] for row in rows
            ),
            "signature_fallback_rows": sum(
                row["metadata"]["structure_profile"]["canonicalization"] != "exact"
                for row in rows
            ),
            "query_base_counts": dict(
                Counter(
                    row["metadata"]["query_kind"]
                    for row in rows
                    if (
                        row["metadata"]["trace_format"] == "natural"
                        and row["metadata"]["supervision_arm"] == "checked-trace"
                    )
                    or role == "bench-depth"
                )
            ),
            "target_tokens_by_package": {
                package: sum(
                    row["metadata"]["target_model_tokens"]
                    for row in rows
                    if row["metadata"]["trace_format"]
                    + ":"
                    + row["metadata"]["supervision_arm"]
                    == package
                )
                for package in sorted(
                    {
                        row["metadata"]["trace_format"]
                        + ":"
                        + row["metadata"]["supervision_arm"]
                        for row in rows
                    }
                )
            },
            "answer_position_rows": dict(
                Counter(
                    letter
                    for row in rows
                    for letter in row["metadata"]["oracle_letters"]
                )
            ),
            "sha256": hashlib.sha256(serialized[role].encode()).hexdigest(),
        }
    output.mkdir(parents=True)
    write_role_texts(output, serialized)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_release(args.config, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
