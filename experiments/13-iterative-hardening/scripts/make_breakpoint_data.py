"""Generate the frozen evaluation-only V13.1 structural breakpoint suite."""

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

from spatial_generation_v13 import (
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
from spatial_graph import ENTITY_NAMES

SEED = 13700
VERSION = "v13.1-structural-breakpoint-v1"
DEPTHS = (6, 8, 10)
UNEQUAL_DEPTHS = ((2, 6), (6, 2), (4, 8), (8, 4), (6, 10), (10, 6))
DEPTH_PER_CELL = 12
LOOP_PER_CELL = 5
EXPECTED_CELLS = 186
EXPECTED_ROWS = 1224
EXPECTED_DEPTH_ROWS = 504
EXPECTED_CLOSED_ROWS = 360
EXPECTED_OPEN_ROWS = 360

# Frozen V13 uses the first 20 names. V13.1 deliberately permits larger
# worlds, while GenerationSpec's default pool preserves old seeded outputs.
CHALLENGE_ENTITY_NAMES = (
    *ENTITY_NAMES,
    "Train Station",
    "Airport",
    "Restaurant",
    "Hotel",
    "Courthouse",
    "Community Center",
    "Sports Stadium",
    "Art Gallery",
    "Bus Terminal",
    "Harbor",
    "Factory",
    "Warehouse",
    "Observatory",
    "Botanical Garden",
    "Theater",
    "Bookstore",
    "Clinic",
    "Playground",
    "Farmers Market",
    "Elementary School",
    "Recreation Center",
    "Aquarium",
    "Convention Center",
    "Research Lab",
    "Music Hall",
    "Water Tower",
    "Power Station",
    "Town Square",
)

RELATION_RE = re.compile(
    r"The (.+?) is to the "
    r"(Northeast|Northwest|Southeast|Southwest|North|South|East|West) "
    r"of the (.+?)\.",
    re.IGNORECASE,
)


def depth_cell(
    bucket: str,
    mode: RelationMode,
    x_depth: int,
    y_depth: int,
    policy: DistractorPolicy,
) -> GenerationCell:
    distractor_count = 0 if policy is DistractorPolicy.NONE else max(x_depth, y_depth)
    extra_entities = (
        distractor_count + 1
        if policy is DistractorPolicy.DISCONNECTED
        else distractor_count
    )
    return GenerationCell(
        name=(f"break-{bucket}-{mode.value}-x{x_depth}-y{y_depth}-{policy.value}"),
        spec=GenerationSpec(
            semantic_subtype=SemanticSubtype.DIR_1,
            relation_mode=mode,
            constraints=StructuralConstraints(
                require_independent_axes=True,
                x_depth=DepthRange.exact(x_depth),
                y_depth=DepthRange.exact(y_depth),
                distractors=DistractorSpec(policy=policy, count=distractor_count),
            ),
            num_entities=x_depth + y_depth + extra_entities,
            num_relations=x_depth + y_depth + distractor_count,
            max_attempts=2000,
            entity_pool=CHALLENGE_ENTITY_NAMES,
        ),
        count=DEPTH_PER_CELL,
    )


def depth_cells() -> list[GenerationCell]:
    cells = []
    symmetric = (
        ("long-clean", DistractorPolicy.NONE),
        ("long-disconnected", DistractorPolicy.DISCONNECTED),
        ("long-query-branch", DistractorPolicy.QUERY_BRANCH),
    )
    for bucket, policy in symmetric:
        for depth in DEPTHS:
            for mode in (RelationMode.CARDINAL, RelationMode.MIXED):
                cells.append(depth_cell(bucket, mode, depth, depth, policy))

    for x_depth, y_depth in UNEQUAL_DEPTHS:
        for mode in (RelationMode.CARDINAL, RelationMode.MIXED):
            cells.append(
                depth_cell(
                    "unequal-clean",
                    mode,
                    x_depth,
                    y_depth,
                    DistractorPolicy.NONE,
                )
            )
            cells.append(
                depth_cell(
                    "unequal-query-branch",
                    mode,
                    x_depth,
                    y_depth,
                    DistractorPolicy.QUERY_BRANCH,
                )
            )
    assert len(cells) == 42
    assert sum(cell.count for cell in cells) == EXPECTED_DEPTH_ROWS
    return cells


LOOP_CONFIGURATIONS = (
    (RelationMode.CARDINAL, CycleAxes.X),
    (RelationMode.CARDINAL, CycleAxes.BOTH),
    (RelationMode.MIXED, CycleAxes.Y),
    (RelationMode.DIAGONAL, CycleAxes.BOTH),
)
LOOP_FAMILIES = (
    ("direction", SemanticSubtype.DIR_1),
    ("which", SemanticSubtype.WHICH_2),
    ("count", SemanticSubtype.COUNT_1),
)


def loop_pair(
    family: str,
    subtype: SemanticSubtype,
    mode: RelationMode,
    axes: CycleAxes,
    placement: CyclePlacement,
    length: int,
) -> tuple[GenerationCell, GenerationCell]:
    relation_count = (
        length * 2
        if (axes is CycleAxes.BOTH and mode is RelationMode.CARDINAL)
        else length
    )
    fresh_entities = (
        length + 1 - (1 if placement is CyclePlacement.QUERY_CONNECTED else 0)
    )
    prefix = (
        f"break-long-loop-{family}-{mode.value}-{axes.value}-"
        f"{placement.value}-l{length}"
    )

    def make(consistency: WorldConsistency, suffix: str) -> GenerationCell:
        return GenerationCell(
            name=f"{prefix}-{suffix}",
            spec=GenerationSpec(
                semantic_subtype=subtype,
                relation_mode=mode,
                cycle=CycleSpec(
                    axes=axes,
                    topology=CycleTopology.INDIRECT,
                    placement=placement,
                    length=length,
                    world_consistency=consistency,
                ),
                num_entities=8 + fresh_entities,
                num_relations=10 + relation_count,
                max_attempts=2000,
                entity_pool=CHALLENGE_ENTITY_NAMES,
            ),
            count=LOOP_PER_CELL,
        )

    return (
        make(WorldConsistency.INCONSISTENT, "closed"),
        make(WorldConsistency.CONSISTENT, "open-control"),
    )


def loop_cells() -> list[GenerationCell]:
    cells = []
    for family, subtype in LOOP_FAMILIES:
        for mode, axes in LOOP_CONFIGURATIONS:
            for placement in CyclePlacement:
                for length in DEPTHS:
                    cells.extend(
                        loop_pair(family, subtype, mode, axes, placement, length)
                    )
    assert len(cells) == 144
    assert sum(cell.count for cell in cells) == (
        EXPECTED_CLOSED_ROWS + EXPECTED_OPEN_ROWS
    )
    return cells


def build_cells() -> list[GenerationCell]:
    cells = depth_cells() + loop_cells()
    assert len(cells) == EXPECTED_CELLS
    assert len({cell.name for cell in cells}) == EXPECTED_CELLS
    assert sum(cell.count for cell in cells) == EXPECTED_ROWS
    return cells


def paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output.with_name(output.stem + "_train.jsonl"),
        output.with_name(output.stem + "_test.jsonl"),
        output.with_name(output.stem + "_manifest.json"),
    )


def user_text(row: dict) -> str:
    return next(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if message.get("role") == "user"
    )


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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
    return fingerprint(json.dumps(relations, separators=(",", ":")))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hashes(rows: list[dict], function) -> set[str]:
    return {function(user_text(row)) for row in rows}


def generate(output: Path, held_out: list[Path]) -> tuple[Path, Path, Path]:
    train_path, test_path, manifest_path = paths(output)
    external = [row for path in held_out for row in load_jsonl(path)]
    accepted_prompts = _hashes(external, fingerprint)
    accepted_worlds = _hashes(external, world_fingerprint)
    engine = SpatialGenerator()
    seed_rng = random.Random(SEED)
    rows = []
    rejections: Counter = Counter()
    for cell in build_cells():
        accepted = 0
        attempts = 0
        while accepted < cell.count:
            attempts += 1
            if attempts > cell.count * 2000:
                raise ValueError(
                    f"unable to fill unique cell {cell.name}: "
                    f"{accepted}/{cell.count}, rejections={dict(rejections)}"
                )
            example = engine.generate(
                cell.spec, random.Random(seed_rng.randrange(2**63))
            )
            row = example.to_row(generation_cell=cell.name)
            row["generator_version"] = VERSION
            text = user_text(row)
            prompt_hash = fingerprint(text)
            world_hash = world_fingerprint(text)
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
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("")
    test_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "test_rows": len(rows),
        "generation_cells": EXPECTED_CELLS,
        "long_proof_rows": EXPECTED_DEPTH_ROWS,
        "closed_loop_rows": EXPECTED_CLOSED_ROWS,
        "open_chain_rows": EXPECTED_OPEN_ROWS,
        "held_out_sources": [str(path) for path in held_out],
        "prompt_overlap": 0,
        "world_overlap": 0,
        "generation_rejections": dict(sorted(rejections.items())),
        "cell_counts": dict(
            sorted(Counter(row["generation_cell"] for row in rows).items())
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate(test_path, manifest_path, held_out)
    return train_path, test_path, manifest_path


def validate(test_path: Path, manifest_path: Path, held_out: list[Path]) -> None:
    rows = load_jsonl(test_path)
    external = [row for path in held_out for row in load_jsonl(path)]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION or manifest.get("seed") != SEED:
        raise ValueError("breakpoint manifest version or seed mismatch")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    expected_cells = Counter({cell.name: cell.count for cell in build_cells()})
    if Counter(row.get("generation_cell") for row in rows) != expected_cells:
        raise ValueError("breakpoint cell distribution mismatch")
    if {row.get("difficulty_schema_version") for row in rows} != {4}:
        raise ValueError("breakpoint difficulty schema mismatch")
    if {row.get("generator_version") for row in rows} != {VERSION}:
        raise ValueError("breakpoint generator version mismatch")

    prompts = _hashes(rows, fingerprint)
    worlds = _hashes(rows, world_fingerprint)
    external_prompts = _hashes(external, fingerprint)
    external_worlds = _hashes(external, world_fingerprint)
    if len(prompts) != EXPECTED_ROWS or len(worlds) != EXPECTED_ROWS:
        raise ValueError("duplicate breakpoint prompt or world")
    if prompts & external_prompts:
        raise ValueError("breakpoint prompt leakage")
    if worlds & external_worlds:
        raise ValueError("breakpoint world leakage")

    for row in rows:
        cell = row["generation_cell"]
        difficulty = row["difficulty"]
        if cell.startswith("break-long-loop-"):
            expected_consistency = (
                "consistent" if cell.endswith("-open-control") else "inconsistent"
            )
            if difficulty["world_consistency"] != expected_consistency:
                raise ValueError(f"wrong loop consistency in {cell}")
            length = int(re.search(r"-l(\d+)-", cell).group(1))
            if expected_consistency == "inconsistent" and any(
                value != length for value in difficulty["cycle_lengths"].values()
            ):
                raise ValueError(f"wrong loop length in {cell}")
        else:
            match = re.search(r"-x(\d+)-y(\d+)-", cell)
            x_depth, y_depth = map(int, match.groups())
            if (difficulty["x_depth"], difficulty["y_depth"]) != (x_depth, y_depth):
                raise ValueError(f"wrong controlled depth in {cell}")
            if max(x_depth, y_depth) < 6:
                raise ValueError(f"non-challenge depth in {cell}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "spatial_v13_breakpoint.jsonl",
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
        "--sft-6k-train",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_6000_train.jsonl",
    )
    parser.add_argument(
        "--sft-6k-val",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_6000_val.jsonl",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    held_out = [
        args.diagnostic,
        args.probe_v1_train,
        args.probe_v1_val,
        args.probe_v2_train,
        args.probe_v2_val,
        args.sft_6k_train,
        args.sft_6k_val,
    ]
    missing = [str(path) for path in held_out if not path.is_file()]
    if missing:
        raise SystemExit("missing held-out input: " + ", ".join(missing))
    _train, test, manifest = paths(args.out)
    if args.validate_only:
        validate(test, manifest, held_out)
        print(f"valid V13.1 breakpoint suite: {test}")
        return
    output_paths = generate(args.out, held_out)
    print("wrote " + ", ".join(str(path) for path in output_paths))


if __name__ == "__main__":
    main()
