"""Generate nested V13.1 8K SFT data from frozen V13 6K plus 2K rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SFT6 = load_module("v13_sft_8000_sft6", HERE / "make_sft_6000_data.py")
BREAKPOINT = load_module("v13_sft_8000_breakpoint", HERE / "make_breakpoint_data.py")
V1 = SFT6.V1
REPO = SFT6.REPO
SEED = 13800
VERSION = "v13.1-sft-8000-nested-v1"
EXPECTED_TRAIN = 8000
EXPECTED_ADDITIONS = 2000
EXPECTED_VAL = SFT6.EXPECTED_VAL


def paths(output: Path) -> tuple[Path, Path, Path]:
    return V1.probe_paths(output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def depth_cell(
    name: str,
    mode,
    x_depth: int,
    y_depth: int,
    policy,
    count: int,
):
    distractor_count = max(x_depth, y_depth)
    extra_entities = (
        distractor_count + 1
        if policy is V1.DistractorPolicy.DISCONNECTED
        else distractor_count
    )
    return V1.GenerationCell(
        name=name,
        spec=V1.GenerationSpec(
            semantic_subtype=V1.SemanticSubtype.DIR_1,
            relation_mode=mode,
            constraints=V1.StructuralConstraints(
                require_independent_axes=True,
                x_depth=V1.DepthRange.exact(x_depth),
                y_depth=V1.DepthRange.exact(y_depth),
                distractors=V1.DistractorSpec(policy=policy, count=distractor_count),
            ),
            num_entities=x_depth + y_depth + extra_entities,
            num_relations=x_depth + y_depth + distractor_count,
            max_attempts=2000,
            entity_pool=BREAKPOINT.CHALLENGE_ENTITY_NAMES,
        ),
        count=count,
    )


def interference_cells() -> list:
    cells = []
    symmetric = (
        ("d6-disconnected", 6, V1.DistractorPolicy.DISCONNECTED, 125),
        ("d6-query-branch", 6, V1.DistractorPolicy.QUERY_BRANCH, 150),
        ("d8-disconnected", 8, V1.DistractorPolicy.DISCONNECTED, 150),
        ("d8-query-branch", 8, V1.DistractorPolicy.QUERY_BRANCH, 200),
    )
    for label, depth, policy, count in symmetric:
        for mode in (V1.RelationMode.CARDINAL, V1.RelationMode.MIXED):
            cells.append(
                depth_cell(
                    f"sft8-{label}-{mode.value}",
                    mode,
                    depth,
                    depth,
                    policy,
                    count,
                )
            )

    unequal = ((2, 6), (6, 2), (4, 8), (8, 4), (6, 8), (8, 6))
    for x_depth, y_depth in unequal:
        for mode in (V1.RelationMode.CARDINAL, V1.RelationMode.MIXED):
            cells.append(
                depth_cell(
                    f"sft8-unequal-query-branch-{mode.value}-x{x_depth}-y{y_depth}",
                    mode,
                    x_depth,
                    y_depth,
                    V1.DistractorPolicy.QUERY_BRANCH,
                    25,
                )
            )
    assert sum(cell.count for cell in cells) == 1550
    return cells


def selected_loop_configurations() -> list[tuple]:
    candidates = []
    for family_index, (family, subtype) in enumerate(BREAKPOINT.LOOP_FAMILIES):
        for config_index, (mode, axes) in enumerate(BREAKPOINT.LOOP_CONFIGURATIONS):
            for placement_index, placement in enumerate(V1.CyclePlacement):
                for length_index, length in enumerate((6, 8)):
                    if (
                        family_index + config_index + placement_index + length_index
                    ) % 2 == 0:
                        candidates.append(
                            (family, subtype, mode, axes, placement, length)
                        )
    assert len(candidates) == 24
    candidates.append(
        (
            "which",
            V1.SemanticSubtype.WHICH_2,
            V1.RelationMode.MIXED,
            V1.CycleAxes.Y,
            V1.CyclePlacement.DISCONNECTED,
            8,
        )
    )
    assert len(candidates) == len(set(candidates)) == 25
    return candidates


def loop_cells() -> list:
    cells = []
    for index, (family, subtype, mode, axes, placement, length) in enumerate(
        selected_loop_configurations(), 1
    ):
        closed, opened = BREAKPOINT.loop_pair(
            family, subtype, mode, axes, placement, length
        )
        cells.extend(
            (
                V1.GenerationCell(
                    name=f"sft8-loop-{index}-{family}-l{length}-closed",
                    spec=closed.spec,
                    count=6,
                ),
                V1.GenerationCell(
                    name=f"sft8-loop-{index}-{family}-l{length}-open-control",
                    spec=opened.spec,
                    count=10,
                ),
            )
        )
    assert sum(cell.count for cell in cells) == 400
    return cells


def large_world_semantic_cells() -> list:
    configurations = (
        (V1.SemanticSubtype.WHICH_1, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_2, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_2, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_3, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_4, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_0, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.COUNT_1, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.COUNT_1, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.COUNT_OMIT, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.COUNT_OMIT, V1.RelationMode.MIXED),
    )
    cells = [
        V1.GenerationCell(
            name=f"sft8-large-{index}-{mode.value}-{subtype.value}",
            spec=V1.GenerationSpec(
                semantic_subtype=subtype,
                relation_mode=mode,
                num_entities=24,
                num_relations=32,
                max_attempts=2000,
                entity_pool=BREAKPOINT.CHALLENGE_ENTITY_NAMES,
            ),
            count=5,
        )
        for index, (subtype, mode) in enumerate(configurations, 1)
    ]
    assert sum(cell.count for cell in cells) == 50
    return cells


def build_addition_cells() -> list:
    cells = interference_cells() + loop_cells() + large_world_semantic_cells()
    assert len(cells) == 80
    assert len({cell.name for cell in cells}) == 80
    assert sum(cell.count for cell in cells) == EXPECTED_ADDITIONS
    return cells


def _hashes(rows: list[dict], function) -> set[str]:
    return {function(BREAKPOINT.user_text(row)) for row in rows}


def generate_additions(
    cells: list, forbidden: list[dict]
) -> tuple[list[dict], Counter]:
    accepted_prompts = _hashes(forbidden, BREAKPOINT.fingerprint)
    accepted_worlds = _hashes(forbidden, BREAKPOINT.world_fingerprint)
    engine = V1.SpatialGenerator()
    seed_rng = random.Random(SEED)
    rows = []
    rejections: Counter = Counter()
    for cell in cells:
        accepted = 0
        attempts = 0
        while accepted < cell.count:
            attempts += 1
            if attempts > cell.count * 2000:
                raise ValueError(
                    f"unable to fill {cell.name}: {accepted}/{cell.count}; "
                    f"rejections={dict(rejections)}"
                )
            example = engine.generate(
                cell.spec, random.Random(seed_rng.randrange(2**63))
            )
            row = example.to_row(generation_cell=cell.name)
            row["generator_version"] = VERSION
            text = BREAKPOINT.user_text(row)
            prompt_hash = BREAKPOINT.fingerprint(text)
            world_hash = BREAKPOINT.world_fingerprint(text)
            if prompt_hash in accepted_prompts:
                rejections["prompt_collision"] += 1
                continue
            if world_hash in accepted_worlds:
                rejections["world_collision"] += 1
                continue
            rows.append(row)
            accepted_prompts.add(prompt_hash)
            accepted_worlds.add(world_hash)
            accepted += 1
    seed_rng.shuffle(rows)
    return rows, rejections


def validate(
    train_path: Path,
    val_path: Path,
    manifest_path: Path,
    base_train_path: Path,
    base_val_path: Path,
    base_manifest_path: Path,
    base_1500_paths: tuple[Path, Path, Path],
    original_held_out: list[Path],
    breakpoint_path: Path,
    breakpoint_manifest_path: Path,
) -> None:
    SFT6.validate(
        base_train_path,
        base_val_path,
        base_manifest_path,
        *base_1500_paths,
        original_held_out,
    )
    BREAKPOINT.validate(
        breakpoint_path,
        breakpoint_manifest_path,
        [*original_held_out, base_train_path, base_val_path],
    )
    train = V1.load_jsonl(train_path)
    val = V1.load_jsonl(val_path)
    base_train = V1.load_jsonl(base_train_path)
    base_val = V1.load_jsonl(base_val_path)
    breakpoint = V1.load_jsonl(breakpoint_path)
    external = [
        row for path in original_held_out for row in V1.load_jsonl(path)
    ] + breakpoint
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("version") != VERSION or manifest.get("seed") != SEED:
        raise ValueError("8K manifest version or seed mismatch")
    if len(train) != EXPECTED_TRAIN or len(val) != EXPECTED_VAL:
        raise ValueError(f"unexpected sizes: train={len(train)} val={len(val)}")
    if train[: len(base_train)] != base_train:
        raise ValueError("8K does not contain the exact 6K rows as its prefix")
    if not train_path.read_bytes().startswith(base_train_path.read_bytes()):
        raise ValueError("8K serialized prefix differs from frozen 6K bytes")
    if val != base_val or val_path.read_bytes() != base_val_path.read_bytes():
        raise ValueError("8K validation differs from frozen 6K validation")
    for key, path in (
        ("base_train_sha256", base_train_path),
        ("base_validation_sha256", base_val_path),
        ("base_manifest_sha256", base_manifest_path),
        ("breakpoint_sha256", breakpoint_path),
        ("breakpoint_manifest_sha256", breakpoint_manifest_path),
    ):
        if manifest.get(key) != sha256(path):
            raise ValueError(f"8K manifest has stale {key}")

    train_prompts = _hashes(train, BREAKPOINT.fingerprint)
    train_worlds = _hashes(train, BREAKPOINT.world_fingerprint)
    val_prompts = _hashes(val, BREAKPOINT.fingerprint)
    val_worlds = _hashes(val, BREAKPOINT.world_fingerprint)
    external_prompts = _hashes(external, BREAKPOINT.fingerprint)
    external_worlds = _hashes(external, BREAKPOINT.world_fingerprint)
    if len(train_prompts) != EXPECTED_TRAIN or len(train_worlds) != EXPECTED_TRAIN:
        raise ValueError("duplicate prompt or world in 8K training data")
    if train_prompts & val_prompts or train_worlds & val_worlds:
        raise ValueError("8K train/validation leakage")
    if (train_prompts | val_prompts) & external_prompts:
        raise ValueError("8K prompt leakage into frozen evaluation data")
    if (train_worlds | val_worlds) & external_worlds:
        raise ValueError("8K world leakage into frozen evaluation data")

    additions = train[len(base_train) :]
    expected_cells = Counter({cell.name: cell.count for cell in build_addition_cells()})
    if Counter(row["generation_cell"] for row in additions) != expected_cells:
        raise ValueError("8K extension cell distribution mismatch")
    category_counts = Counter(
        "closed_loop_rows"
        if row["generation_cell"].endswith("-closed")
        else "open_chain_rows"
        if row["generation_cell"].endswith("-open-control")
        else "large_world_semantic_rows"
        if row["generation_cell"].startswith("sft8-large-")
        else "interference_rows"
        for row in additions
    )
    expected_categories = Counter(
        {
            "interference_rows": 1550,
            "closed_loop_rows": 150,
            "open_chain_rows": 250,
            "large_world_semantic_rows": 50,
        }
    )
    if EXPECTED_ADDITIONS == 2000 and category_counts != expected_categories:
        raise ValueError("8K extension category distribution mismatch")
    if EXPECTED_ADDITIONS == 2000 and any(
        manifest.get(key) != value for key, value in expected_categories.items()
    ):
        raise ValueError("8K manifest category distribution mismatch")
    if any(
        max(
            row["difficulty"].get("x_depth") or 0, row["difficulty"].get("y_depth") or 0
        )
        >= 10
        for row in additions
    ):
        raise ValueError("depth 10 must remain evaluation-only")
    if any(
        max(row["difficulty"].get("cycle_lengths", {}).values(), default=0) >= 10
        for row in additions
    ):
        raise ValueError("loop length 10 must remain evaluation-only")


def generate(
    output: Path,
    base_train_path: Path,
    base_val_path: Path,
    base_manifest_path: Path,
    base_1500_paths: tuple[Path, Path, Path],
    original_held_out: list[Path],
    breakpoint_path: Path,
    breakpoint_manifest_path: Path,
) -> tuple[Path, Path, Path]:
    SFT6.validate(
        base_train_path,
        base_val_path,
        base_manifest_path,
        *base_1500_paths,
        original_held_out,
    )
    BREAKPOINT.validate(
        breakpoint_path,
        breakpoint_manifest_path,
        [*original_held_out, base_train_path, base_val_path],
    )
    train_path, val_path, manifest_path = paths(output)
    base_train = V1.load_jsonl(base_train_path)
    base_val = V1.load_jsonl(base_val_path)
    external = [
        row for path in original_held_out for row in V1.load_jsonl(path)
    ] + V1.load_jsonl(breakpoint_path)
    additions, rejections = generate_additions(
        build_addition_cells(), external + base_train + base_val
    )

    with train_path.open("wb") as target:
        base_bytes = base_train_path.read_bytes()
        target.write(base_bytes)
        if base_bytes and not base_bytes.endswith(b"\n"):
            target.write(b"\n")
        for row in additions:
            target.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
    val_path.write_bytes(base_val_path.read_bytes())
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "train_rows": EXPECTED_TRAIN,
        "nested_base_rows": SFT6.EXPECTED_TRAIN,
        "new_rows": EXPECTED_ADDITIONS,
        "validation_rows": EXPECTED_VAL,
        "interference_rows": 1550,
        "closed_loop_rows": 150,
        "open_chain_rows": 250,
        "large_world_semantic_rows": 50,
        "addition_cells": len(build_addition_cells()),
        "base_train_sha256": sha256(base_train_path),
        "base_validation_sha256": sha256(base_val_path),
        "base_manifest_sha256": sha256(base_manifest_path),
        "breakpoint_sha256": sha256(breakpoint_path),
        "breakpoint_manifest_sha256": sha256(breakpoint_manifest_path),
        "prompt_overlap": 0,
        "world_overlap": 0,
        "addition_generation_rejections": dict(sorted(rejections.items())),
        "addition_cell_counts": dict(
            sorted(Counter(row["generation_cell"] for row in additions).items())
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate(
        train_path,
        val_path,
        manifest_path,
        base_train_path,
        base_val_path,
        base_manifest_path,
        base_1500_paths,
        original_held_out,
        breakpoint_path,
        breakpoint_manifest_path,
    )
    return train_path, val_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=REPO / "data" / "spatial_v13_sft_8000.jsonl"
    )
    parser.add_argument(
        "--base-train",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_6000_train.jsonl",
    )
    parser.add_argument(
        "--base-val",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_6000_val.jsonl",
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_6000_manifest.json",
    )
    parser.add_argument(
        "--base-1500-train",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_1500_train.jsonl",
    )
    parser.add_argument(
        "--base-1500-val",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_1500_val.jsonl",
    )
    parser.add_argument(
        "--base-1500-manifest",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_1500_manifest.json",
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=REPO / "data" / "spatial_v13_diagnostic_test.jsonl",
    )
    parser.add_argument(
        "--probe-v1-train",
        type=Path,
        default=REPO / "data" / "spatial_v13_probe_train.jsonl",
    )
    parser.add_argument(
        "--probe-v1-val",
        type=Path,
        default=REPO / "data" / "spatial_v13_probe_val.jsonl",
    )
    parser.add_argument(
        "--probe-v2-train",
        type=Path,
        default=REPO / "data" / "spatial_v13_probe_v2_train.jsonl",
    )
    parser.add_argument(
        "--probe-v2-val",
        type=Path,
        default=REPO / "data" / "spatial_v13_probe_v2_val.jsonl",
    )
    parser.add_argument(
        "--breakpoint",
        type=Path,
        default=REPO / "data" / "spatial_v13_breakpoint_test.jsonl",
    )
    parser.add_argument(
        "--breakpoint-manifest",
        type=Path,
        default=REPO / "data" / "spatial_v13_breakpoint_manifest.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    base_1500_paths = (
        args.base_1500_train,
        args.base_1500_val,
        args.base_1500_manifest,
    )
    original_held_out = [
        args.diagnostic,
        args.probe_v1_train,
        args.probe_v1_val,
        args.probe_v2_train,
        args.probe_v2_val,
    ]
    required = [
        args.base_train,
        args.base_val,
        args.base_manifest,
        *base_1500_paths,
        *original_held_out,
        args.breakpoint,
        args.breakpoint_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required input: " + ", ".join(missing))
    output_paths = paths(args.out)
    call_args = (
        args.base_train,
        args.base_val,
        args.base_manifest,
        base_1500_paths,
        original_held_out,
        args.breakpoint,
        args.breakpoint_manifest,
    )
    if args.validate_only:
        validate(*output_paths, *call_args)
        print("valid nested 8K data: " + ", ".join(map(str, output_paths)))
        return
    output_paths = generate(args.out, *call_args)
    print("wrote " + ", ".join(map(str, output_paths)))


if __name__ == "__main__":
    main()
