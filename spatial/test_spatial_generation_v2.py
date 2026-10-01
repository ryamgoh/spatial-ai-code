"""Contracts for the solver-validated V2 SpatialMap generator."""

from __future__ import annotations

import json

import pytest
from generate_all_v2 import app, generate_workload
from spatial_explanation_renderers_v2 import StateMode, TraceFormat
from spatial_generation_v2 import (
    GenerationPolicy,
    MenuCoverage,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
)
from spatial_grading_v2 import AnswerMode, encode_menu_answer, resolve_answer
from spatial_solver_v2 import (
    CountQuery,
    Direction,
    DirectionQuery,
    SpatialSolverV2,
    WhichQuery,
)
from spatial_text_v2 import SpatialTextAdapter
from typer.testing import CliRunner


@pytest.mark.parametrize(
    ("query_kind", "query_type"),
    [
        (QueryKind.DIRECTION, DirectionQuery),
        (QueryKind.WHICH, WhichQuery),
        (QueryKind.COUNT, CountQuery),
    ],
)
@pytest.mark.parametrize("answer_mode", tuple(AnswerMode))
def test_generated_rows_round_trip_for_every_query_and_answer_mode(
    query_kind: QueryKind,
    query_type: type,
    answer_mode: AnswerMode,
) -> None:
    policy = GenerationPolicy(
        query_kind=query_kind,
        answer_mode=answer_mode,
        semantic_shape=SemanticShape.UNIQUE,
        menu_coverage=MenuCoverage.FULL,
        num_entities=5,
        num_premises=6,
        query_direction=(
            None if query_kind is QueryKind.DIRECTION else Direction.NORTH
        ),
        trace_format=TraceFormat.SYMBOLIC,
        state_mode=StateMode.DELTA,
    )

    sample = SpatialGeneratorV2(seed=1701).generate(policy)
    row = sample.as_sft_row()
    parsed = SpatialTextAdapter().parse(row["messages"][1]["content"])

    assert isinstance(parsed.problem.query, query_type)
    assert parsed.problem == sample.problem
    assert parsed.options == sample.options
    assert row["metadata"]["answer_mode"] == answer_mode.value
    assert row["metadata"]["round_trip_verified"] is True
    assert row["metadata"]["oracle_letters"] == sorted(sample.menu_answer.letters)
    assert row["messages"][2]["content"].endswith(
        "Answer: " + ", ".join(sorted(sample.menu_answer.letters))
    )
    assert "Answer-Mode=" in sample.trace


def test_generator_supports_every_compass_direction_for_which_and_count() -> None:
    generator = SpatialGeneratorV2(seed=1703)

    for query_kind in (QueryKind.WHICH, QueryKind.COUNT):
        for direction in Direction:
            sample = generator.generate(
                GenerationPolicy(
                    query_kind=query_kind,
                    query_direction=direction,
                    semantic_shape=SemanticShape.UNIQUE,
                    num_entities=4,
                    num_premises=4,
                )
            )
            assert sample.problem.query.directions == frozenset({direction})


@pytest.mark.parametrize("trace_format", tuple(TraceFormat))
@pytest.mark.parametrize("state_mode", tuple(StateMode))
def test_training_rows_never_contain_coordinate_witnesses(
    trace_format: TraceFormat,
    state_mode: StateMode,
) -> None:
    sample = SpatialGeneratorV2(seed=1705).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            trace_format=trace_format,
            state_mode=state_mode,
        )
    )

    serialized = json.dumps(sample.as_sft_row())
    audit = sample.audit_metadata

    assert "coordinates" not in serialized.lower()
    assert "generation_witness" in audit
    assert set(audit["generation_witness"]) == set(sample.problem.objects)


def test_partial_all_possible_menu_maps_to_cannot_be_determined() -> None:
    sample = SpatialGeneratorV2(seed=1707).generate(
        GenerationPolicy(
            query_kind=QueryKind.COUNT,
            answer_mode=AnswerMode.ALL_POSSIBLE,
            semantic_shape=SemanticShape.AMBIGUOUS,
            menu_coverage=MenuCoverage.PARTIAL,
            num_entities=5,
            num_premises=4,
        )
    )

    assert sample.menu_answer.status == "incomplete-menu"
    assert "Cannot be determined" in sample.options.values()
    assert sample.menu_answer.letters


def test_zero_visible_possible_menu_maps_to_none_of_options() -> None:
    sample = SpatialGeneratorV2(seed=1709).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            answer_mode=AnswerMode.VISIBLE_POSSIBLE,
            semantic_shape=SemanticShape.UNIQUE,
            menu_coverage=MenuCoverage.ZERO,
        )
    )

    assert sample.menu_answer.status == "none-of-options"
    assert "None of the Options" in sample.options.values()
    assert "The menu result is none-of-options" in sample.trace


def test_map_scoped_which_keeps_hidden_objects_in_candidate_universe() -> None:
    prompt = (
        "Consider a map with multiple locations:\n\n"
        "A is to the Northeast of R. B is to the Northwest of R. "
        "C is to the Southwest of R.\n\n"
        "Question: Select the complete set of possible answers. "
        "Which object in the map is in the Northeast of R? "
        "Available options: A. B, B. C, C. None of the Options"
    )

    parsed = SpatialTextAdapter().parse(prompt)
    analysis = SpatialSolverV2().analyze(parsed.problem)
    answer = encode_menu_answer(
        resolve_answer(analysis, AnswerMode.ALL_POSSIBLE),
        parsed.options,
    )

    assert isinstance(parsed.problem.query, WhichQuery)
    assert parsed.problem.query.candidates == ("A", "B", "C")
    assert analysis.possible_entities == ("A",)
    assert answer.status == "none-of-options"
    assert answer.raw == "C"


def test_generator_is_deterministic_per_seed_and_policy() -> None:
    policy = GenerationPolicy(
        query_kind=QueryKind.WHICH,
        answer_mode=AnswerMode.SINGLE,
        semantic_shape=SemanticShape.UNIQUE,
        query_direction=Direction.SOUTHWEST,
    )

    first = SpatialGeneratorV2(seed=1711).generate(policy).as_sft_row()
    second = SpatialGeneratorV2(seed=1711).generate(policy).as_sft_row()

    assert first == second


def test_policy_rejects_impossible_partial_singleton_contract() -> None:
    with pytest.raises(ValueError, match="partial menu coverage"):
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            answer_mode=AnswerMode.SINGLE,
            menu_coverage=MenuCoverage.PARTIAL,
        )


def test_balanced_workload_writes_unique_verified_rows_without_audit(tmp_path) -> None:
    train_path, test_path = generate_workload(
        tmp_path / "workload.jsonl",
        samples_per_cell=2,
        query_kinds=(QueryKind.DIRECTION, QueryKind.COUNT),
        answer_modes=(AnswerMode.SINGLE,),
        semantic_shapes=(SemanticShape.UNIQUE,),
        menu_coverages=(MenuCoverage.FULL,),
        trace_formats=(TraceFormat.NATURAL, TraceFormat.SYMBOLIC),
        state_modes=(StateMode.DELTA,),
        test_split=0.25,
        seed=1713,
    )

    rows = [
        json.loads(line)
        for path in (train_path, test_path)
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 8
    assert len({row["id"] for row in rows}) == 8
    assert {row["metadata"]["query_kind"] for row in rows} == {
        "direction",
        "count",
    }
    assert {row["metadata"]["trace_format"] for row in rows} == {
        "natural",
        "symbolic",
    }
    assert all("audit" not in row for row in rows)


def test_cli_rejects_unknown_policy_dimensions(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(tmp_path / "bad.jsonl"),
            "--query-kinds",
            "direction,teleport",
        ],
    )

    assert result.exit_code != 0
    assert "unknown query kinds: teleport" in result.output


def test_workload_can_balance_which_queries_across_directions(tmp_path) -> None:
    train_path, _ = generate_workload(
        tmp_path / "directions.jsonl",
        samples_per_cell=1,
        query_kinds=(QueryKind.WHICH,),
        answer_modes=(AnswerMode.SINGLE,),
        semantic_shapes=(SemanticShape.UNIQUE,),
        menu_coverages=(MenuCoverage.FULL,),
        trace_formats=(TraceFormat.NATURAL,),
        state_modes=(StateMode.DELTA,),
        query_directions=(Direction.NORTH, Direction.SOUTHWEST),
        test_split=0,
        seed=1715,
    )

    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert {row["metadata"]["query_direction"] for row in rows} == {
        "North",
        "Southwest",
    }
