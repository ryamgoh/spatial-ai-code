"""Public-interface tests for the deep v13 generation module."""

from __future__ import annotations

import json
import random

import pytest

from spatial_generation_v13 import (
    DepthRange,
    GenerationCell,
    GenerationError,
    GenerationSpec,
    RelationMode,
    SemanticSubtype,
    SpatialGenerator,
    StructuralConstraints,
    generate_dataset,
)
from spatial_solver_v13 import SpatialSolverV13


def user_text(example) -> str:
    return next(
        message["content"]
        for message in example.messages
        if message["role"] == "user"
    )


def test_generate_accepts_one_coherent_spec() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.MIXED,
        constraints=StructuralConstraints(require_independent_axes=True),
        num_entities=10,
        num_relations=15,
    )

    row = SpatialGenerator().generate(spec, random.Random(1321))
    solved = SpatialSolverV13().solve_and_analyze(user_text(row))

    assert row.oracle_option == solved.grade.raw
    assert row.difficulty == solved.structure
    assert row.difficulty["semantic_subtype"] == "dir-1"
    assert row.difficulty["relation_mix"] == "mixed"
    assert row.difficulty["axes_independent"] is True
    assert row.to_row()["difficulty_schema_version"] == 1


@pytest.mark.parametrize("relation_mode", tuple(RelationMode))
@pytest.mark.parametrize("semantic_subtype", tuple(SemanticSubtype))
def test_every_semantic_subtype_and_relation_mode_round_trips(
    semantic_subtype: SemanticSubtype, relation_mode: RelationMode
) -> None:
    spec = GenerationSpec(
        semantic_subtype=semantic_subtype,
        relation_mode=relation_mode,
        num_entities=10,
        num_relations=15,
    )

    example = SpatialGenerator().generate(spec, random.Random(1313))
    solved = SpatialSolverV13().solve_and_analyze(user_text(example))

    assert example.oracle_option == solved.grade.raw
    assert example.difficulty == solved.structure
    assert example.difficulty["semantic_subtype"] == semantic_subtype.value
    assert example.difficulty["relation_mix"] == {
        RelationMode.DIAGONAL: "diagonal-only",
        RelationMode.CARDINAL: "cardinal-only",
        RelationMode.MIXED: "mixed",
    }[relation_mode]


def test_invalid_spec_fails_before_generation() -> None:
    with pytest.raises(ValueError, match="independent axes.*dir-1"):
        GenerationSpec(
            semantic_subtype=SemanticSubtype.COUNT_1,
            relation_mode=RelationMode.MIXED,
            constraints=StructuralConstraints(require_independent_axes=True),
        )


def test_generation_error_reports_rejection_reasons() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.MIXED,
        constraints=StructuralConstraints(require_independent_axes=True),
        num_entities=2,
        num_relations=1,
        max_attempts=3,
    )

    with pytest.raises(GenerationError) as caught:
        SpatialGenerator().generate(spec, random.Random(7))

    assert caught.value.attempts == 3
    assert caught.value.rejections
    assert sum(caught.value.rejections.values()) == 3
    assert set(caught.value.rejections) <= {
        "no_qualifying_query",
        "solver_rejected",
        "wrong_semantic_subtype",
        "wrong_relation_mode",
        "axes_not_independent",
    }


def test_generate_dataset_accepts_explicit_cells(tmp_path) -> None:
    cells = [
        GenerationCell(
            name="cardinal-dir-2",
            spec=GenerationSpec(
                semantic_subtype=SemanticSubtype.DIR_2,
                relation_mode=RelationMode.CARDINAL,
            ),
            count=5,
        ),
        GenerationCell(
            name="mixed-count-omit",
            spec=GenerationSpec(
                semantic_subtype=SemanticSubtype.COUNT_OMIT,
                relation_mode=RelationMode.MIXED,
            ),
            count=5,
        ),
    ]

    train_path, test_path = generate_dataset(
        cells,
        output_file=tmp_path / "planned.jsonl",
        test_fraction=0.2,
        seed=13,
    )

    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    test = [json.loads(line) for line in test_path.read_text().splitlines()]
    assert len(train) == 8
    assert len(test) == 2
    assert {row["generation_cell"] for row in train + test} == {
        "cardinal-dir-2",
        "mixed-count-omit",
    }
    assert {row["generation_cell"] for row in test} == {
        "cardinal-dir-2",
        "mixed-count-omit",
    }


def test_depth_range_validates_bounds() -> None:
    assert DepthRange.exact(3).contains(3)
    assert not DepthRange(3, 5).contains(2)
    assert DepthRange(3, 5).contains(5)
    with pytest.raises(ValueError, match="minimum.*positive"):
        DepthRange(0, 2)
    with pytest.raises(ValueError, match="maximum.*minimum"):
        DepthRange(4, 3)


def test_depth_constraints_require_dir1() -> None:
    with pytest.raises(ValueError, match="proof depth.*dir-1"):
        GenerationSpec(
            semantic_subtype=SemanticSubtype.DIR_2,
            relation_mode=RelationMode.CARDINAL,
            constraints=StructuralConstraints(x_depth=DepthRange.exact(2)),
        )
    with pytest.raises(ValueError, match="proof depth.*independent axes"):
        GenerationSpec(
            semantic_subtype=SemanticSubtype.DIR_1,
            relation_mode=RelationMode.CARDINAL,
            constraints=StructuralConstraints(
                x_depth=DepthRange.exact(2),
                y_depth=DepthRange.exact(2),
            ),
        )


@pytest.mark.parametrize(
    "relation_mode,x_depth,y_depth",
    [
        (RelationMode.CARDINAL, 3, 4),
        (RelationMode.MIXED, 4, 3),
    ],
)
def test_generate_enforces_independent_axis_depths(
    relation_mode: RelationMode, x_depth: int, y_depth: int
) -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=relation_mode,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange.exact(x_depth),
            y_depth=DepthRange.exact(y_depth),
        ),
        num_entities=10,
        num_relations=12,
    )

    example = SpatialGenerator().generate(spec, random.Random(1401))

    assert example.difficulty["x_depth"] == x_depth
    assert example.difficulty["y_depth"] == y_depth
    assert example.difficulty["axes_independent"] is True
    assert example.difficulty["shared_supporting_statements"] == []


def test_generate_accepts_depth_ranges_not_only_exact_values() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.CARDINAL,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange(2, 3),
            y_depth=DepthRange(3, 4),
        ),
        num_entities=9,
        num_relations=10,
    )

    example = SpatialGenerator().generate(spec, random.Random(1403))

    assert 2 <= example.difficulty["x_depth"] <= 3
    assert 3 <= example.difficulty["y_depth"] <= 4
