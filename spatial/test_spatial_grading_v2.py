"""Contracts for semantic resolution, menu encoding, and response scoring."""

from __future__ import annotations

from spatial_grading_v2 import (
    AnswerSemantics,
    ResolutionStatus,
    SelectionMode,
    encode_menu_answer,
    resolve_answer,
    score_response,
)
from spatial_solver_v2 import CountAnalysis, Direction, DirectionAnalysis, WhichAnalysis


def test_direction_exact_and_all_possible_are_distinct() -> None:
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTHEAST, Direction.NORTHWEST),
    )

    exact = resolve_answer(analysis, AnswerSemantics.EXACT)
    possible = resolve_answer(analysis, AnswerSemantics.ALL_POSSIBLE)

    assert exact.status is ResolutionStatus.AMBIGUOUS
    assert exact.values == ()
    assert possible.status is ResolutionStatus.POSSIBILITIES
    assert possible.values == (Direction.NORTHEAST, Direction.NORTHWEST)


def test_all_possible_never_falls_back_for_an_incomplete_menu() -> None:
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTHEAST, Direction.NORTHWEST),
    )
    resolution = resolve_answer(analysis, AnswerSemantics.ALL_POSSIBLE)

    answer = encode_menu_answer(
        resolution,
        {"A": "Northeast", "B": "Cannot be determined"},
        SelectionMode.MULTI_SELECT,
    )

    assert answer.status == "incomplete-menu"
    assert answer.letters == frozenset()
    assert not answer.is_resolved
    assert answer.error == "menu is missing possible answer: Northwest"


def test_exact_which_set_can_require_multiple_selections() -> None:
    analysis = WhichAnalysis(
        consistent=True,
        reference="R",
        directions=frozenset({Direction.NORTHEAST}),
        candidates=("A", "B", "C"),
        possible_entities=("A", "B"),
        entailed_entities=("A", "B"),
    )
    resolution = resolve_answer(analysis, AnswerSemantics.EXACT)

    multiple = encode_menu_answer(
        resolution,
        {"A": "A", "B": "B", "C": "C", "D": "Cannot be determined"},
        SelectionMode.MULTI_SELECT,
    )
    single = encode_menu_answer(
        resolution,
        {"A": "A", "B": "B", "C": "C", "D": "Cannot be determined"},
        SelectionMode.SINGLE_SELECT,
    )

    assert resolution.status is ResolutionStatus.EXACT
    assert resolution.values == ("A", "B")
    assert multiple.raw == "A,B"
    assert multiple.is_resolved
    assert single.status == "selection-mismatch"
    assert not single.is_resolved


def test_exact_which_with_a_contingent_member_is_undetermined() -> None:
    analysis = WhichAnalysis(
        consistent=True,
        reference="R",
        directions=frozenset({Direction.NORTHEAST}),
        candidates=("A", "B"),
        possible_entities=("A", "B"),
        entailed_entities=("A",),
    )
    resolution = resolve_answer(analysis, AnswerSemantics.EXACT)

    answer = encode_menu_answer(
        resolution,
        {"A": "A", "B": "B", "C": "Cannot be determined"},
        SelectionMode.SINGLE_SELECT,
    )

    assert resolution.status is ResolutionStatus.AMBIGUOUS
    assert answer.raw == "C"
    assert answer.status == "undetermined"


def test_count_resolution_preserves_non_contiguous_possibilities() -> None:
    analysis = CountAnalysis(
        consistent=True,
        reference="R",
        directions=frozenset({Direction.NORTHEAST}),
        candidates=("A", "B", "C"),
        possible_counts=(1, 3),
    )

    exact = resolve_answer(analysis, AnswerSemantics.EXACT)
    possible = resolve_answer(analysis, AnswerSemantics.ALL_POSSIBLE)
    answer = encode_menu_answer(
        possible,
        {"A": "1", "B": "2", "C": "3"},
        SelectionMode.MULTI_SELECT,
    )

    assert exact.status is ResolutionStatus.AMBIGUOUS
    assert possible.values == (1, 3)
    assert answer.raw == "A,C"


def test_response_scoring_uses_exact_set_equality_and_reports_partial_metrics() -> None:
    analysis = WhichAnalysis(
        consistent=True,
        reference="R",
        directions=frozenset({Direction.NORTHEAST}),
        candidates=("A", "B"),
        possible_entities=("A", "B"),
        entailed_entities=("A", "B"),
    )
    expected = encode_menu_answer(
        resolve_answer(analysis, AnswerSemantics.EXACT),
        {"A": "A", "B": "B"},
        SelectionMode.MULTI_SELECT,
    )

    partial = score_response("Answer: A", expected)
    exact = score_response("Answer: B, A", expected)

    assert not partial.exact_match
    assert partial.precision == 1.0
    assert partial.recall == 0.5
    assert partial.f1 == 2 / 3
    assert exact.exact_match
