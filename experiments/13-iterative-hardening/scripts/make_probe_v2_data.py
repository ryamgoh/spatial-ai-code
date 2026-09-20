"""Generate the consistency-focused 400-row native V13 Probe v2."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "make_probe_data.py"
SPEC = importlib.util.spec_from_file_location("v13_probe_v1_helpers", V1_PATH)
assert SPEC and SPEC.loader
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

REPO = V1.REPO
TRAIN_NONCYCLE = 10
TRAIN_CYCLE = 5
VAL_PER_CELL = 2
EXPECTED_CELLS = 55
SEED = 13231
PROBE_VERSION = "v13-probe-v2-400-v1"


def build_probe_v2_cells(*, validation: bool = False) -> list:
    noncycle_count = VAL_PER_CELL if validation else TRAIN_NONCYCLE
    cycle_count = VAL_PER_CELL if validation else TRAIN_CYCLE
    cells = []

    # 60 train rows: enough clean cardinal depth coverage to preserve the base
    # chain skill without dominating the consistency intervention.
    for index, (x_depth, y_depth) in enumerate(
        ((1, 1), (1, 3), (2, 2), (2, 4), (3, 3), (4, 4)), 1
    ):
        cells.append(
            V1.depth_cell(
                f"probe-v2-cardinal-{index}-x{x_depth}-y{y_depth}",
                V1.RelationMode.CARDINAL,
                x_depth,
                y_depth,
                count=noncycle_count,
            )
        )

    # 60 train rows: independent mixed composition, including both distractor
    # topologies implicated by the transfer audit.
    mixed_specs = (
        (2, 2, V1.DistractorPolicy.NONE, 0),
        (3, 4, V1.DistractorPolicy.NONE, 0),
        (4, 3, V1.DistractorPolicy.NONE, 0),
        (2, 3, V1.DistractorPolicy.DISCONNECTED, 2),
        (4, 4, V1.DistractorPolicy.DISCONNECTED, 3),
        (3, 3, V1.DistractorPolicy.QUERY_BRANCH, 3),
    )
    for index, (x_depth, y_depth, policy, distractors) in enumerate(mixed_specs, 1):
        cells.append(
            V1.depth_cell(
                f"probe-v2-mixed-{index}-x{x_depth}-y{y_depth}-{policy.value}",
                V1.RelationMode.MIXED,
                x_depth,
                y_depth,
                policy,
                distractors,
                noncycle_count,
            )
        )

    # 50 train rows: retain Probe v1's large gain on the partial-information
    # two-letter policy.
    dir2_modes = (
        V1.RelationMode.CARDINAL,
        V1.RelationMode.CARDINAL,
        V1.RelationMode.MIXED,
        V1.RelationMode.MIXED,
        V1.RelationMode.DIAGONAL,
    )
    for index, mode in enumerate(dir2_modes, 1):
        cells.append(
            V1.semantic_cell(
                f"probe-v2-dir-2-{mode.value}-{index}",
                V1.SemanticSubtype.DIR_2,
                mode,
                noncycle_count,
            )
        )

    # 50 train rows: retain enumeration improvements.
    which_specs = (
        (V1.SemanticSubtype.WHICH_1, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_2, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_3, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_4, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_0, V1.RelationMode.DIAGONAL),
    )
    for index, (subtype, mode) in enumerate(which_specs, 1):
        cells.append(
            V1.semantic_cell(
                f"probe-v2-{subtype.value}-{mode.value}-{index}",
                subtype,
                mode,
                noncycle_count,
            )
        )

    # 30 train rows: retain counting and option omission.
    for subtype, mode in (
        (V1.SemanticSubtype.COUNT_1, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.COUNT_1, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.COUNT_OMIT, V1.RelationMode.CARDINAL),
    ):
        cells.append(
            V1.semantic_cell(
                f"probe-v2-{subtype.value}-{mode.value}",
                subtype,
                mode,
                noncycle_count,
            )
        )

    # 75 closed-loop conditions + 75 open-chain controls. Five configurations
    # per family span axis,
    # direct/indirect topology, connected/disconnected placement and relation
    # mode. This is the only intervention changed from Probe v1.
    configurations = (
        (V1.RelationMode.CARDINAL, V1.CycleAxes.X, V1.CycleTopology.DIRECT, V1.CyclePlacement.QUERY_CONNECTED),
        (V1.RelationMode.CARDINAL, V1.CycleAxes.Y, V1.CycleTopology.INDIRECT, V1.CyclePlacement.DISCONNECTED),
        (V1.RelationMode.CARDINAL, V1.CycleAxes.BOTH, V1.CycleTopology.INDIRECT, V1.CyclePlacement.QUERY_CONNECTED),
        (V1.RelationMode.MIXED, V1.CycleAxes.X, V1.CycleTopology.INDIRECT, V1.CyclePlacement.DISCONNECTED),
        (V1.RelationMode.DIAGONAL, V1.CycleAxes.BOTH, V1.CycleTopology.DIRECT, V1.CyclePlacement.DISCONNECTED),
    )
    families = (
        ("direction", V1.SemanticSubtype.DIR_1),
        ("which", V1.SemanticSubtype.WHICH_2),
        ("count", V1.SemanticSubtype.COUNT_1),
    )
    for family, subtype in families:
        for index, (mode, axes, topology, placement) in enumerate(configurations, 1):
            cells.extend(
                V1.cycle_pair(
                    f"probe-v2-cycle-{family}-{index}",
                    subtype,
                    mode,
                    axes,
                    topology,
                    placement,
                    cycle_count,
                )
            )

    assert len(cells) == EXPECTED_CELLS
    assert sum(cell.count for cell in cells) == (110 if validation else 400)
    return cells


def paths(output: Path) -> tuple[Path, Path, Path]:
    return V1.probe_paths(output)


def _hashes(rows: list[dict], function) -> set[str]:
    return {function(V1.user_text(row)) for row in rows}


def validate(
    train_path: Path,
    val_path: Path,
    manifest_path: Path,
    diagnostic: Path,
    v1_train: Path,
    v1_val: Path,
) -> None:
    train, val = V1.load_jsonl(train_path), V1.load_jsonl(val_path)
    diagnostic_rows = V1.load_jsonl(diagnostic)
    v1_rows = V1.load_jsonl(v1_train) + V1.load_jsonl(v1_val)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("probe_version") != PROBE_VERSION:
        raise ValueError("Probe v2 manifest version mismatch")
    if manifest.get("seed") != SEED or manifest.get("validation_seed") != SEED + 1:
        raise ValueError("Probe v2 seed mismatch")
    if len(train) != 400 or len(val) != 110:
        raise ValueError(f"unexpected Probe v2 sizes: train={len(train)} val={len(val)}")

    train_prompts, val_prompts = _hashes(train, V1.fingerprint), _hashes(val, V1.fingerprint)
    external_prompts = _hashes(diagnostic_rows + v1_rows, V1.fingerprint)
    train_worlds, val_worlds = _hashes(train, V1.world_fingerprint), _hashes(val, V1.world_fingerprint)
    external_worlds = _hashes(diagnostic_rows + v1_rows, V1.world_fingerprint)
    if len(train_prompts) != 400 or len(val_prompts) != 110:
        raise ValueError("duplicate prompt within Probe v2")
    if train_prompts & val_prompts or (train_prompts | val_prompts) & external_prompts:
        raise ValueError("Probe v2 prompt leakage")
    if train_worlds & val_worlds or (train_worlds | val_worlds) & external_worlds:
        raise ValueError("Probe v2 world leakage")
    if any(
        max(row["difficulty"].get("x_depth") or 0, row["difficulty"].get("y_depth") or 0) >= 5
        for row in train + val
    ):
        raise ValueError("depth 5 must remain held out from Probe v2")
    if Counter(row["generation_cell"] for row in train) != Counter(
        {cell.name: cell.count for cell in build_probe_v2_cells()}
    ):
        raise ValueError("Probe v2 train cell distribution mismatch")
    if Counter(row["generation_cell"] for row in val) != Counter(
        {cell.name: cell.count for cell in build_probe_v2_cells(validation=True)}
    ):
        raise ValueError("Probe v2 validation cell distribution mismatch")


def generate(
    output: Path, diagnostic: Path, v1_train: Path, v1_val: Path
) -> tuple[Path, Path, Path]:
    train_path, val_path, manifest_path = paths(output)
    V1.generate_partition(build_probe_v2_cells(), train_path, SEED)
    V1.generate_partition(build_probe_v2_cells(validation=True), val_path, SEED + 1)
    train, val = V1.load_jsonl(train_path), V1.load_jsonl(val_path)
    manifest = {
        "probe_version": PROBE_VERSION,
        "seed": SEED,
        "validation_seed": SEED + 1,
        "train_rows": len(train),
        "validation_rows": len(val),
        "generation_cells": EXPECTED_CELLS,
        "train_cell_counts": dict(sorted(Counter(row["generation_cell"] for row in train).items())),
        "validation_cell_counts": dict(sorted(Counter(row["generation_cell"] for row in val).items())),
        "held_out_sources": [str(diagnostic), str(v1_train), str(v1_val)],
        "prompt_overlap": 0,
        "world_overlap": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate(train_path, val_path, manifest_path, diagnostic, v1_train, v1_val)
    return train_path, val_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "data" / "spatial_v13_probe_v2.jsonl")
    parser.add_argument("--diagnostic", type=Path, default=REPO / "data" / "spatial_v13_diagnostic_test.jsonl")
    parser.add_argument("--v1-train", type=Path, default=REPO / "data" / "spatial_v13_probe_train.jsonl")
    parser.add_argument("--v1-val", type=Path, default=REPO / "data" / "spatial_v13_probe_val.jsonl")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    required = (args.diagnostic, args.v1_train, args.v1_val)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing held-out input: " + ", ".join(missing))
    output_paths = paths(args.out)
    if args.validate_only:
        validate(*output_paths, args.diagnostic, args.v1_train, args.v1_val)
        print("valid Probe v2 data: " + ", ".join(str(path) for path in output_paths))
        return
    output_paths = generate(args.out, args.diagnostic, args.v1_train, args.v1_val)
    print("wrote " + ", ".join(str(path) for path in output_paths))


if __name__ == "__main__":
    main()
