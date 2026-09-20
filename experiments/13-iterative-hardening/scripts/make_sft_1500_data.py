"""Generate the frozen native V13 1.5K train and per-cell validation set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "make_probe_data.py"
SPEC = importlib.util.spec_from_file_location("v13_1500_helpers", V1_PATH)
assert SPEC and SPEC.loader
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

REPO = V1.REPO
SEED = 13500
VERSION = "v13-sft-1500-v1"
EXPECTED_TRAIN = 1500
EXPECTED_VAL = 129
EXPECTED_CELLS = 129


def ordinary_cells(*, validation: bool = False) -> list:
    depth_count = 1 if validation else 20
    semantic_count = 1 if validation else 10
    cells = []

    # 240 rows: clean cardinal/mixed retention and unequal-axis composition.
    depth_specs = (
        (V1.RelationMode.CARDINAL, 1, 1),
        (V1.RelationMode.CARDINAL, 1, 3),
        (V1.RelationMode.CARDINAL, 2, 2),
        (V1.RelationMode.CARDINAL, 2, 4),
        (V1.RelationMode.CARDINAL, 3, 2),
        (V1.RelationMode.CARDINAL, 4, 4),
        (V1.RelationMode.MIXED, 2, 2),
        (V1.RelationMode.MIXED, 2, 4),
        (V1.RelationMode.MIXED, 3, 2),
        (V1.RelationMode.MIXED, 3, 4),
        (V1.RelationMode.MIXED, 4, 3),
        (V1.RelationMode.MIXED, 4, 4),
    )
    for index, (mode, x_depth, y_depth) in enumerate(depth_specs, 1):
        cells.append(
            V1.depth_cell(
                f"sft15-depth-{index}-{mode.value}-x{x_depth}-y{y_depth}",
                mode,
                x_depth,
                y_depth,
                count=depth_count,
            )
        )

    # 180 rows: hard-negative relation branches without depth-5 exposure.
    distractor_specs = (
        (V1.RelationMode.CARDINAL, 2, 3, V1.DistractorPolicy.DISCONNECTED, 2),
        (V1.RelationMode.CARDINAL, 3, 4, V1.DistractorPolicy.DISCONNECTED, 3),
        (V1.RelationMode.CARDINAL, 4, 3, V1.DistractorPolicy.QUERY_BRANCH, 3),
        (V1.RelationMode.CARDINAL, 4, 4, V1.DistractorPolicy.QUERY_BRANCH, 4),
        (V1.RelationMode.MIXED, 2, 3, V1.DistractorPolicy.DISCONNECTED, 2),
        (V1.RelationMode.MIXED, 3, 4, V1.DistractorPolicy.DISCONNECTED, 3),
        (V1.RelationMode.MIXED, 3, 2, V1.DistractorPolicy.QUERY_BRANCH, 2),
        (V1.RelationMode.MIXED, 4, 3, V1.DistractorPolicy.QUERY_BRANCH, 3),
        (V1.RelationMode.MIXED, 4, 4, V1.DistractorPolicy.QUERY_BRANCH, 4),
    )
    for index, (mode, xd, yd, policy, distractors) in enumerate(distractor_specs, 1):
        cells.append(
            V1.depth_cell(
                f"sft15-distractor-{index}-{mode.value}-x{xd}-y{yd}-{policy.value}",
                mode, xd, yd, policy, distractors, depth_count,
            )
        )

    # 360 rows: full 3 relation modes × 12 semantic subtype grid.
    for mode in V1.RelationMode:
        for subtype in V1.SemanticSubtype:
            cells.append(
                V1.semantic_cell(
                    f"sft15-semantic-{mode.value}-{subtype.value}",
                    subtype, mode, semantic_count,
                )
            )

    # 160 rows: targeted extra mass on the base model's weak answer policies.
    targeted = (
        (V1.SemanticSubtype.DIR_2, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.DIR_2, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.DIR_INCOMPLETE, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.DIR_UNDETERMINED, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_1, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_2, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_2, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_3, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_3, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_4, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_0, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_0, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.COUNT_1, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.COUNT_1, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.COUNT_OMIT, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.COUNT_OMIT, V1.RelationMode.MIXED),
    )
    for index, (subtype, mode) in enumerate(targeted, 1):
        cells.append(
            V1.semantic_cell(
                f"sft15-target-{index}-{mode.value}-{subtype.value}",
                subtype, mode, semantic_count,
            )
        )

    assert len(cells) == 73
    assert sum(cell.count for cell in cells) == (73 if validation else 940)
    return cells


def loop_configurations() -> tuple:
    return (
        (V1.RelationMode.CARDINAL, V1.CycleAxes.X, V1.CycleTopology.DIRECT, V1.CyclePlacement.QUERY_CONNECTED),
        (V1.RelationMode.CARDINAL, V1.CycleAxes.Y, V1.CycleTopology.DIRECT, V1.CyclePlacement.DISCONNECTED),
        (V1.RelationMode.CARDINAL, V1.CycleAxes.X, V1.CycleTopology.INDIRECT, V1.CyclePlacement.DISCONNECTED),
        (V1.RelationMode.CARDINAL, V1.CycleAxes.Y, V1.CycleTopology.INDIRECT, V1.CyclePlacement.QUERY_CONNECTED),
        (V1.RelationMode.CARDINAL, V1.CycleAxes.BOTH, V1.CycleTopology.INDIRECT, V1.CyclePlacement.QUERY_CONNECTED),
        (V1.RelationMode.MIXED, V1.CycleAxes.X, V1.CycleTopology.DIRECT, V1.CyclePlacement.DISCONNECTED),
        (V1.RelationMode.MIXED, V1.CycleAxes.Y, V1.CycleTopology.INDIRECT, V1.CyclePlacement.DISCONNECTED),
        (V1.RelationMode.MIXED, V1.CycleAxes.BOTH, V1.CycleTopology.INDIRECT, V1.CyclePlacement.QUERY_CONNECTED),
        (V1.RelationMode.DIAGONAL, V1.CycleAxes.BOTH, V1.CycleTopology.DIRECT, V1.CyclePlacement.QUERY_CONNECTED),
        (V1.RelationMode.DIAGONAL, V1.CycleAxes.BOTH, V1.CycleTopology.INDIRECT, V1.CyclePlacement.DISCONNECTED),
    )


def loop_cells(*, validation: bool = False) -> list:
    count = 1 if validation else 10
    cells = []
    families = (
        ("direction", V1.SemanticSubtype.DIR_1, 10),
        ("which", V1.SemanticSubtype.WHICH_2, 9),
        ("count", V1.SemanticSubtype.COUNT_1, 9),
    )
    configurations = loop_configurations()
    for family, subtype, number in families:
        for index, (mode, axes, topology, placement) in enumerate(configurations[:number], 1):
            cells.extend(
                V1.cycle_pair(
                    f"sft15-loop-{family}-{index}",
                    subtype, mode, axes, topology, placement, count,
                )
            )
    assert len(cells) == 56
    assert sum(cell.count for cell in cells) == (56 if validation else 560)
    return cells


def build_cells(*, validation: bool = False) -> list:
    cells = ordinary_cells(validation=validation) + loop_cells(validation=validation)
    assert len(cells) == EXPECTED_CELLS
    assert sum(cell.count for cell in cells) == (EXPECTED_VAL if validation else EXPECTED_TRAIN)
    return cells


def paths(output: Path) -> tuple[Path, Path, Path]:
    return V1.probe_paths(output)


def _hashes(rows: list[dict], function) -> set[str]:
    return {function(V1.user_text(row)) for row in rows}


def generate_unique_partition(
    cells: list,
    path: Path,
    seed: int,
    forbidden_prompts: set[str],
    forbidden_worlds: set[str],
) -> Counter:
    engine = V1.SpatialGenerator()
    seed_rng = random.Random(seed)
    accepted_prompts = set(forbidden_prompts)
    accepted_worlds = set(forbidden_worlds)
    rows = []
    rejections: Counter = Counter()
    for cell in cells:
        accepted = 0
        attempts = 0
        while accepted < cell.count:
            attempts += 1
            if attempts > max(1000, cell.count * 1000):
                raise ValueError(
                    f"unable to fill unique cell {cell.name}: "
                    f"accepted={accepted}/{cell.count}, rejections={dict(rejections)}"
                )
            example = engine.generate(
                cell.spec, random.Random(seed_rng.randrange(2**63))
            )
            row = example.to_row(generation_cell=cell.name)
            text = V1.user_text(row)
            prompt_hash = V1.fingerprint(text)
            world_hash = V1.world_fingerprint(text)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rejections


def validate(
    train_path: Path, val_path: Path, manifest_path: Path, held_out: list[Path]
) -> None:
    train, val = V1.load_jsonl(train_path), V1.load_jsonl(val_path)
    external = [row for path in held_out for row in V1.load_jsonl(path)]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("1.5K manifest version mismatch")
    if manifest.get("seed") != SEED or manifest.get("validation_seed") != SEED + 1:
        raise ValueError("1.5K seed mismatch")
    if len(train) != EXPECTED_TRAIN or len(val) != EXPECTED_VAL:
        raise ValueError(f"unexpected sizes: train={len(train)} val={len(val)}")

    train_prompts, val_prompts = _hashes(train, V1.fingerprint), _hashes(val, V1.fingerprint)
    external_prompts = _hashes(external, V1.fingerprint)
    train_worlds, val_worlds = _hashes(train, V1.world_fingerprint), _hashes(val, V1.world_fingerprint)
    external_worlds = _hashes(external, V1.world_fingerprint)
    if len(train_prompts) != EXPECTED_TRAIN or len(val_prompts) != EXPECTED_VAL:
        raise ValueError("duplicate prompt in 1.5K data")
    if train_prompts & val_prompts or (train_prompts | val_prompts) & external_prompts:
        raise ValueError("1.5K prompt leakage")
    if train_worlds & val_worlds or (train_worlds | val_worlds) & external_worlds:
        raise ValueError("1.5K world leakage")
    if any(
        max(row["difficulty"].get("x_depth") or 0, row["difficulty"].get("y_depth") or 0) >= 5
        for row in train + val
    ):
        raise ValueError("depth 5 must remain held out")
    if Counter(row["generation_cell"] for row in train) != Counter(
        {cell.name: cell.count for cell in build_cells()}
    ):
        raise ValueError("1.5K train cell distribution mismatch")
    if Counter(row["generation_cell"] for row in val) != Counter(
        {cell.name: cell.count for cell in build_cells(validation=True)}
    ):
        raise ValueError("1.5K validation cell distribution mismatch")


def generate(output: Path, held_out: list[Path]) -> tuple[Path, Path, Path]:
    train_path, val_path, manifest_path = paths(output)
    external = [row for path in held_out for row in V1.load_jsonl(path)]
    external_prompts = _hashes(external, V1.fingerprint)
    external_worlds = _hashes(external, V1.world_fingerprint)
    train_rejections = generate_unique_partition(
        build_cells(),
        train_path,
        SEED,
        external_prompts,
        external_worlds,
    )
    train_rows = V1.load_jsonl(train_path)
    val_rejections = generate_unique_partition(
        build_cells(validation=True),
        val_path,
        SEED + 1,
        external_prompts | _hashes(train_rows, V1.fingerprint),
        external_worlds | _hashes(train_rows, V1.world_fingerprint),
    )
    train, val = V1.load_jsonl(train_path), V1.load_jsonl(val_path)
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "validation_seed": SEED + 1,
        "train_rows": len(train),
        "validation_rows": len(val),
        "ordinary_consistent_rows": 940,
        "closed_loop_rows": 280,
        "open_chain_rows": 280,
        "generation_cells": EXPECTED_CELLS,
        "train_cell_counts": dict(sorted(Counter(row["generation_cell"] for row in train).items())),
        "validation_cell_counts": dict(sorted(Counter(row["generation_cell"] for row in val).items())),
        "held_out_sources": [str(path) for path in held_out],
        "prompt_overlap": 0,
        "world_overlap": 0,
        "train_generation_rejections": dict(sorted(train_rejections.items())),
        "validation_generation_rejections": dict(sorted(val_rejections.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate(train_path, val_path, manifest_path, held_out)
    return train_path, val_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "data" / "spatial_v13_sft_1500.jsonl")
    parser.add_argument("--diagnostic", type=Path, default=REPO / "data" / "spatial_v13_diagnostic_test.jsonl")
    parser.add_argument("--probe-v1-train", type=Path, default=REPO / "data" / "spatial_v13_probe_train.jsonl")
    parser.add_argument("--probe-v1-val", type=Path, default=REPO / "data" / "spatial_v13_probe_val.jsonl")
    parser.add_argument("--probe-v2-train", type=Path, default=REPO / "data" / "spatial_v13_probe_v2_train.jsonl")
    parser.add_argument("--probe-v2-val", type=Path, default=REPO / "data" / "spatial_v13_probe_v2_val.jsonl")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    held_out = [args.diagnostic, args.probe_v1_train, args.probe_v1_val, args.probe_v2_train, args.probe_v2_val]
    missing = [str(path) for path in held_out if not path.is_file()]
    if missing:
        raise SystemExit("missing held-out input: " + ", ".join(missing))
    output_paths = paths(args.out)
    if args.validate_only:
        validate(*output_paths, held_out)
        print("valid 1.5K data: " + ", ".join(str(path) for path in output_paths))
        return
    output_paths = generate(args.out, held_out)
    print("wrote " + ", ".join(str(path) for path in output_paths))


if __name__ == "__main__":
    main()
