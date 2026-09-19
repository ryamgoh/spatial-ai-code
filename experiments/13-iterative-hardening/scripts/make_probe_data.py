"""Generate the 400-train/120-validation native V13 SFT probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "spatial"))

from spatial_generation_v13 import (  # noqa: E402
    CycleAxes,
    CyclePlacement,
    CycleSpec,
    CycleTopology,
    DepthRange,
    DistractorPolicy,
    DistractorSpec,
    GenerationCell,
    GenerationSpec,
    RelationMode,
    SemanticSubtype,
    SpatialGenerator,
    StructuralConstraints,
    WorldConsistency,
)


TRAIN_PER_CELL = 10
VAL_PER_CELL = 3
ROWS_PER_CELL = TRAIN_PER_CELL + VAL_PER_CELL
EXPECTED_CELLS = 40
SEED = 13131
PROBE_VERSION = "v13-probe-400-v1"
RELATION_RE = re.compile(
    r"The (.+?) is to the "
    r"(Northeast|Northwest|Southeast|Southwest|North|South|East|West) "
    r"of the (.+?)\.",
    re.I,
)


def depth_cell(
    name: str,
    mode: RelationMode,
    x_depth: int,
    y_depth: int,
    policy: DistractorPolicy = DistractorPolicy.NONE,
    distractor_count: int = 0,
    count: int = ROWS_PER_CELL,
) -> GenerationCell:
    constraints = StructuralConstraints(
        require_independent_axes=True,
        x_depth=DepthRange.exact(x_depth),
        y_depth=DepthRange.exact(y_depth),
        distractors=DistractorSpec(policy=policy, count=distractor_count),
    )
    extra_entities = (
        distractor_count + 1
        if policy is DistractorPolicy.DISCONNECTED
        else distractor_count
    )
    if policy is DistractorPolicy.NONE:
        extra_entities = 2 if mode is RelationMode.MIXED and x_depth == 1 else 0
    num_entities = max(8, x_depth + y_depth + extra_entities)
    num_relations = x_depth + y_depth + distractor_count
    if mode is RelationMode.MIXED and policy is DistractorPolicy.NONE and x_depth == 1:
        num_relations += 1
    return GenerationCell(
        name=name,
        spec=GenerationSpec(
            semantic_subtype=SemanticSubtype.DIR_1,
            relation_mode=mode,
            constraints=constraints,
            num_entities=num_entities,
            num_relations=num_relations,
        ),
        count=count,
    )


def semantic_cell(
    name: str, subtype: SemanticSubtype, mode: RelationMode, count: int = ROWS_PER_CELL
) -> GenerationCell:
    return GenerationCell(
        name=name,
        spec=GenerationSpec(semantic_subtype=subtype, relation_mode=mode),
        count=count,
    )


def cycle_pair(
    name: str,
    subtype: SemanticSubtype,
    mode: RelationMode,
    axes: CycleAxes,
    topology: CycleTopology,
    placement: CyclePlacement,
    count: int = ROWS_PER_CELL,
) -> tuple[GenerationCell, GenerationCell]:
    length = 2 if topology is CycleTopology.DIRECT else 4
    relation_count = length * 2 if axes is CycleAxes.BOTH and mode is RelationMode.CARDINAL else length
    fresh_entities = length + 1 - (1 if placement is CyclePlacement.QUERY_CONNECTED else 0)

    def make(consistency: WorldConsistency, suffix: str) -> GenerationCell:
        return GenerationCell(
            name=f"{name}-{suffix}",
            spec=GenerationSpec(
                semantic_subtype=subtype,
                relation_mode=mode,
                cycle=CycleSpec(
                    axes=axes,
                    topology=topology,
                    placement=placement,
                    length=length,
                    world_consistency=consistency,
                ),
                num_entities=8 + fresh_entities,
                num_relations=10 + relation_count,
            ),
            count=count,
        )

    return (
        make(WorldConsistency.INCONSISTENT, "closed"),
        make(WorldConsistency.CONSISTENT, "open-control"),
    )


def build_probe_cells(rows_per_cell: int = ROWS_PER_CELL) -> list[GenerationCell]:
    cells: list[GenerationCell] = []

    # 100 train rows: cardinal retention anchors with exact, unequal and equal
    # X/Y depths. Depth 5 remains evaluation-only.
    for index, (x_depth, y_depth) in enumerate(
        ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3),
         (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)),
        1,
    ):
        cells.append(depth_cell(f"probe-cardinal-retain-{index}-x{x_depth}-y{y_depth}", RelationMode.CARDINAL, x_depth, y_depth, count=rows_per_cell))

    # 80 train rows: mixed independent-axis composition with clean and
    # structured-distractor variants.
    mixed_specs = (
        (2, 2, DistractorPolicy.NONE, 0),
        (2, 4, DistractorPolicy.NONE, 0),
        (3, 3, DistractorPolicy.NONE, 0),
        (4, 4, DistractorPolicy.NONE, 0),
        (2, 3, DistractorPolicy.DISCONNECTED, 2),
        (4, 3, DistractorPolicy.DISCONNECTED, 3),
        (3, 2, DistractorPolicy.QUERY_BRANCH, 2),
        (3, 4, DistractorPolicy.QUERY_BRANCH, 3),
    )
    for index, (x_depth, y_depth, policy, count) in enumerate(mixed_specs, 1):
        cells.append(depth_cell(f"probe-mixed-{index}-x{x_depth}-y{y_depth}-{policy.value}", RelationMode.MIXED, x_depth, y_depth, policy, count, rows_per_cell))

    # 60 train rows: the base model's weakest direction answer-set policy.
    for mode in RelationMode:
        for replica in (1, 2):
            cells.append(semantic_cell(f"probe-dir-2-{mode.value}-{replica}", SemanticSubtype.DIR_2, mode, rows_per_cell))

    # 60 train rows: complete-set enumeration, covering every which subtype.
    which_specs = (
        (SemanticSubtype.WHICH_1, RelationMode.CARDINAL),
        (SemanticSubtype.WHICH_2, RelationMode.MIXED),
        (SemanticSubtype.WHICH_3, RelationMode.CARDINAL),
        (SemanticSubtype.WHICH_4, RelationMode.MIXED),
        (SemanticSubtype.WHICH_0, RelationMode.CARDINAL),
        (SemanticSubtype.WHICH_2, RelationMode.DIAGONAL),
    )
    for index, (subtype, mode) in enumerate(which_specs, 1):
        cells.append(semantic_cell(f"probe-{subtype.value}-{mode.value}-{index}", subtype, mode, rows_per_cell))

    # 40 train rows: counting and omitted-count option semantics.
    for subtype, mode in (
        (SemanticSubtype.COUNT_1, RelationMode.CARDINAL),
        (SemanticSubtype.COUNT_1, RelationMode.MIXED),
        (SemanticSubtype.COUNT_OMIT, RelationMode.CARDINAL),
        (SemanticSubtype.COUNT_OMIT, RelationMode.MIXED),
    ):
        cells.append(semantic_cell(f"probe-{subtype.value}-{mode.value}", subtype, mode, rows_per_cell))

    # 30 closed + 30 open controls across all three question families.
    for pair in (
        cycle_pair("probe-cycle-direction", SemanticSubtype.DIR_1, RelationMode.CARDINAL, CycleAxes.X, CycleTopology.DIRECT, CyclePlacement.QUERY_CONNECTED, rows_per_cell),
        cycle_pair("probe-cycle-which", SemanticSubtype.WHICH_2, RelationMode.MIXED, CycleAxes.Y, CycleTopology.INDIRECT, CyclePlacement.DISCONNECTED, rows_per_cell),
        cycle_pair("probe-cycle-count", SemanticSubtype.COUNT_1, RelationMode.CARDINAL, CycleAxes.BOTH, CycleTopology.INDIRECT, CyclePlacement.QUERY_CONNECTED, rows_per_cell),
    ):
        cells.extend(pair)

    assert len(cells) == EXPECTED_CELLS
    return cells


def user_text(row: dict) -> str:
    return next(str(msg.get("content") or "") for msg in row.get("messages") or [] if msg.get("role") == "user")


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def world_fingerprint(text: str) -> str:
    premises = text.split("Question:", 1)[0]
    relations = sorted(
        (
            match.group(1).strip().lower(),
            match.group(2).strip().lower(),
            match.group(3).strip().lower(),
        )
        for match in RELATION_RE.finditer(premises)
    )
    normalized = json.dumps(relations, separators=(",", ":"))
    return fingerprint(normalized)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def generate_partition(cells: list[GenerationCell], path: Path, seed: int) -> None:
    engine = SpatialGenerator()
    seed_rng = random.Random(seed)
    rows = [
        engine.generate(cell.spec, random.Random(seed_rng.randrange(2**63))).to_row(
            generation_cell=cell.name
        )
        for cell in cells
        for _ in range(cell.count)
    ]
    seed_rng.shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def probe_paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output.with_name(output.stem + "_train.jsonl"),
        output.with_name(output.stem + "_val.jsonl"),
        output.with_name(output.stem + "_manifest.json"),
    )


def validate_probe(
    train_path: Path, val_path: Path, manifest_path: Path, diagnostic: Path
) -> None:
    train_rows, val_rows = load_jsonl(train_path), load_jsonl(val_path)
    diagnostic_rows = load_jsonl(diagnostic)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("probe_version") != PROBE_VERSION:
        raise ValueError("probe manifest version mismatch")
    if manifest.get("seed") != SEED or manifest.get("validation_seed") != SEED + 1:
        raise ValueError("probe seed mismatch")
    train_hashes = {fingerprint(user_text(row)) for row in train_rows}
    val_hashes = {fingerprint(user_text(row)) for row in val_rows}
    diagnostic_hashes = {fingerprint(user_text(row)) for row in diagnostic_rows}
    train_worlds = {world_fingerprint(user_text(row)) for row in train_rows}
    val_worlds = {world_fingerprint(user_text(row)) for row in val_rows}
    diagnostic_worlds = {world_fingerprint(user_text(row)) for row in diagnostic_rows}
    if len(train_rows) != 400 or len(val_rows) != 120:
        raise ValueError(f"unexpected split sizes: train={len(train_rows)} val={len(val_rows)}")
    if len(train_hashes) != len(train_rows) or len(val_hashes) != len(val_rows):
        raise ValueError("duplicate prompt within probe train or validation")
    if train_hashes & val_hashes:
        raise ValueError("probe train/validation prompt leakage")
    if (train_hashes | val_hashes) & diagnostic_hashes:
        raise ValueError("probe data overlaps the frozen V13 diagnostic")
    if train_worlds & val_worlds:
        raise ValueError("probe train/validation world leakage")
    if (train_worlds | val_worlds) & diagnostic_worlds:
        raise ValueError("probe worlds overlap the frozen V13 diagnostic")
    if any(
        max(row["difficulty"].get("x_depth") or 0, row["difficulty"].get("y_depth") or 0) >= 5
        for row in train_rows + val_rows
    ):
        raise ValueError("depth 5 must remain held out from probe data")
    expected_train_cells = {cell.name: TRAIN_PER_CELL for cell in build_probe_cells(TRAIN_PER_CELL)}
    expected_val_cells = {cell.name: VAL_PER_CELL for cell in build_probe_cells(VAL_PER_CELL)}
    if Counter(row["generation_cell"] for row in train_rows) != Counter(expected_train_cells):
        raise ValueError("probe train cell distribution mismatch")
    if Counter(row["generation_cell"] for row in val_rows) != Counter(expected_val_cells):
        raise ValueError("probe validation cell distribution mismatch")


def generate_probe(output: Path, diagnostic: Path, seed: int = SEED) -> tuple[Path, Path, Path]:
    if seed != SEED:
        raise ValueError(f"the frozen probe requires seed {SEED}")
    train_path, val_path, manifest_path = probe_paths(output)
    generate_partition(build_probe_cells(TRAIN_PER_CELL), train_path, seed)
    generate_partition(build_probe_cells(VAL_PER_CELL), val_path, seed + 1)
    train_rows, val_rows = load_jsonl(train_path), load_jsonl(val_path)
    manifest = {
        "probe_version": PROBE_VERSION,
        "seed": seed,
        "validation_seed": seed + 1,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "generation_cells": len(build_probe_cells(TRAIN_PER_CELL)),
        "train_cell_counts": dict(sorted(Counter(row["generation_cell"] for row in train_rows).items())),
        "validation_cell_counts": dict(sorted(Counter(row["generation_cell"] for row in val_rows).items())),
        "diagnostic_path": str(diagnostic),
        "train_validation_overlap": 0,
        "train_diagnostic_overlap": 0,
        "validation_diagnostic_overlap": 0,
        "train_validation_world_overlap": 0,
        "train_diagnostic_world_overlap": 0,
        "validation_diagnostic_world_overlap": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate_probe(train_path, val_path, manifest_path, diagnostic)
    return train_path, val_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "data" / "spatial_v13_probe.jsonl")
    parser.add_argument("--diagnostic", type=Path, default=REPO / "data" / "spatial_v13_diagnostic_test.jsonl")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.diagnostic.is_file():
        raise SystemExit(f"missing frozen diagnostic: {args.diagnostic}")
    paths = probe_paths(args.out)
    if args.validate_only:
        validate_probe(*paths, args.diagnostic)
        print("valid probe data: " + ", ".join(str(path) for path in paths))
        return
    paths = generate_probe(args.out, args.diagnostic, args.seed)
    print("wrote " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
