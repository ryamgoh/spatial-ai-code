"""Build the frozen v13 diagnostic evaluation suite.

This is intentionally an evaluation-only preset over the public v13
generation API.  It does not create an SFT training mixture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spatial_generation_v13 import (
    SEMANTIC_SUBTYPES,
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
    StructuralConstraints,
    WorldConsistency,
    generate_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "spatial_v13_diagnostic.jsonl"
QUESTION_FAMILY_SUBTYPES = (
    SemanticSubtype.DIR_1,
    SemanticSubtype.WHICH_2,
    SemanticSubtype.COUNT_1,
)
DEPTHS = (1, 2, 3, 4, 5)
DISTRACTOR_POLICIES = (
    DistractorPolicy.NONE,
    DistractorPolicy.DISCONNECTED,
    DistractorPolicy.QUERY_BRANCH,
)


def _depth_cell(
    mode: RelationMode,
    depth: int,
    policy: DistractorPolicy,
    count: int,
) -> GenerationCell:
    distractor_count = 0 if policy is DistractorPolicy.NONE else 3
    num_entities = depth * 2 + 4
    num_relations = depth * 2 + distractor_count
    return GenerationCell(
        name=f"depth-d{depth}-{mode.value}-{policy.value}",
        spec=GenerationSpec(
            semantic_subtype=SemanticSubtype.DIR_1,
            relation_mode=mode,
            constraints=StructuralConstraints(
                require_independent_axes=True,
                x_depth=DepthRange.exact(depth),
                y_depth=DepthRange.exact(depth),
                distractors=DistractorSpec(
                    policy=policy, count=distractor_count
                ),
            ),
            num_entities=num_entities,
            num_relations=num_relations,
        ),
        count=count,
    )


def _cycle_axes_for_mode(mode: RelationMode) -> tuple[CycleAxes, ...]:
    # A diagonal edge necessarily updates both axes, so an X-only or Y-only
    # diagonal cycle cannot exist while remaining diagonal-only.
    if mode is RelationMode.DIAGONAL:
        return (CycleAxes.BOTH,)
    return tuple(CycleAxes)


def _cycle_cells(
    subtype: SemanticSubtype,
    mode: RelationMode,
    axes: CycleAxes,
    topology: CycleTopology,
    placement: CyclePlacement,
    count: int,
) -> tuple[GenerationCell, GenerationCell]:
    length = 2 if topology is CycleTopology.DIRECT else 4
    cycle_relations = (
        length * 2
        if axes is CycleAxes.BOTH and mode is RelationMode.CARDINAL
        else length
    )
    # The closed cycle and open-chain control have exactly the same overall
    # entity and relation budgets.  The control spends its extra entity on
    # opening the final edge instead of closing the loop.
    fresh_entities = length + 1 - (
        1 if placement is CyclePlacement.QUERY_CONNECTED else 0
    )
    num_entities = 8 + fresh_entities
    num_relations = 10 + cycle_relations
    prefix = (
        f"cycle-{subtype.value}-{mode.value}-{axes.value}-"
        f"{topology.value}-{placement.value}-l{length}"
    )

    def cell(consistency: WorldConsistency, suffix: str) -> GenerationCell:
        return GenerationCell(
            name=prefix + suffix,
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
                num_entities=num_entities,
                num_relations=num_relations,
            ),
            count=count,
        )

    return (
        cell(WorldConsistency.INCONSISTENT, ""),
        cell(WorldConsistency.CONSISTENT, "-control"),
    )


def build_diagnostic_cells(
    *,
    semantic_per_cell: int = 20,
    structural_per_cell: int = 24,
    cycle_per_cell: int = 5,
) -> list[GenerationCell]:
    """Return the explicit, reproducible v13 diagnostic matrix.

    Default total: 2,256 rows across 233 generation cells.
    """
    if min(semantic_per_cell, structural_per_cell, cycle_per_cell) < 0:
        raise ValueError("per-cell counts must be non-negative")

    cells = [
        GenerationCell(
            name=f"semantic-{mode.value}-{subtype}",
            spec=GenerationSpec(
                semantic_subtype=SemanticSubtype(subtype),
                relation_mode=mode,
            ),
            count=semantic_per_cell,
        )
        for mode in RelationMode
        for subtype in SEMANTIC_SUBTYPES
    ]

    cells.extend(
        _depth_cell(mode, depth, policy, structural_per_cell)
        for mode in (RelationMode.CARDINAL, RelationMode.MIXED)
        for depth in DEPTHS
        for policy in DISTRACTOR_POLICIES
        # A direct (depth-1), independent X/Y proof cannot contain a relevant
        # diagonal statement. Mixed depth-1 therefore requires a distractor.
        if not (
            mode is RelationMode.MIXED
            and depth == 1
            and policy is DistractorPolicy.NONE
        )
    )

    for subtype in QUESTION_FAMILY_SUBTYPES:
        for mode in RelationMode:
            for axes in _cycle_axes_for_mode(mode):
                for topology in CycleTopology:
                    for placement in CyclePlacement:
                        cells.extend(
                            _cycle_cells(
                                subtype,
                                mode,
                                axes,
                                topology,
                                placement,
                                cycle_per_cell,
                            )
                        )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the evaluation-only v13 diagnostic suite"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1313)
    parser.add_argument("--semantic-per-cell", type=int, default=20)
    parser.add_argument("--structural-per-cell", type=int, default=24)
    parser.add_argument("--cycle-per-cell", type=int, default=5)
    args = parser.parse_args()

    cells = build_diagnostic_cells(
        semantic_per_cell=args.semantic_per_cell,
        structural_per_cell=args.structural_per_cell,
        cycle_per_cell=args.cycle_per_cell,
    )
    train_path, test_path = generate_dataset(
        cells,
        output_file=args.out,
        test_fraction=1.0,
        seed=args.seed,
    )
    total = sum(cell.count for cell in cells)
    print(f"wrote {total} diagnostic rows across {len(cells)} cells to {test_path}")
    print(f"wrote empty training sentinel to {train_path}")


if __name__ == "__main__":
    main()
