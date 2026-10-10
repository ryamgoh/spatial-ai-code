"""Contracts for semantic resolution, menu encoding, and response scoring."""

from __future__ import annotations

import json

import pytest

from spatial.v2.grading import (
    AnswerDecisionCheckError,
    AnswerMode,
    ResolutionStatus,
    check_symbolic_answer_decision,
    encode_menu_answer,
    parse_symbolic_answer_decision,
    render_symbolic_answer_decision,
    resolve_answer,
    score_response,
)
from spatial.v2.solver import CountAnalysis, Direction, DirectionAnalysis, WhichAnalysis


def test_direction_single_and_multi_complete_are_distinct() -> None:
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTHEAST, Direction.NORTHWEST),
    )

    exact = resolve_answer(analysis, AnswerMode.SINGLE)
    possible = resolve_answer(analysis, AnswerMode.ALL_POSSIBLE)

    assert exact.status is ResolutionStatus.AMBIGUOUS
    assert exact.values == ()
    assert possible.status is ResolutionStatus.POSSIBILITIES
    assert possible.values == (Direction.NORTHEAST, Direction.NORTHWEST)


def test_multi_complete_uses_undetermined_for_partially_complete_menu() -> None:
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTHEAST, Direction.NORTHWEST),
    )
    resolution = resolve_answer(analysis, AnswerMode.ALL_POSSIBLE)

    answer = encode_menu_answer(
        resolution,
        {"A": "Northeast", "B": "Cannot be determined"},
    )

    assert answer.status == "incomplete-menu"
    assert answer.raw == "B"
    assert answer.is_resolved


def test_multi_complete_partial_menu_fails_without_undetermined_option() -> None:
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTHEAST, Direction.NORTHWEST),
    )
    resolution = resolve_answer(analysis, AnswerMode.ALL_POSSIBLE)

    answer = encode_menu_answer(
        resolution,
        {"A": "Northeast", "B": "Southeast"},
    )

    assert answer.status == "incomplete-menu"
    assert answer.raw == "Error: menu is missing possible answer: Northwest"


def test_multi_complete_uses_none_when_no_possible_option_is_shown() -> None:
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTHWEST,),
    )
    resolution = resolve_answer(analysis, AnswerMode.ALL_POSSIBLE)

    answer = encode_menu_answer(
        resolution,
        {"A": "Northeast", "B": "Southeast", "C": "None of the options"},
    )

    assert answer.status == "none-of-options"
    assert answer.raw == "C"


def test_multi_visible_selects_only_possible_options_that_are_shown() -> None:
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTHEAST, Direction.NORTHWEST),
    )
    resolution = resolve_answer(analysis, AnswerMode.VISIBLE_POSSIBLE)

    answer = encode_menu_answer(
        resolution,
        {"A": "Northeast", "B": "Southeast"},
    )

    assert answer.status == "possibilities"
    assert answer.letters == frozenset({"A"})
    assert answer.is_resolved


def test_multi_visible_uses_none_when_no_possible_option_is_shown() -> None:
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTHWEST,),
    )
    resolution = resolve_answer(analysis, AnswerMode.VISIBLE_POSSIBLE)

    answer = encode_menu_answer(
        resolution,
        {"A": "Northeast", "B": "Southeast", "C": "None of the options"},
    )

    assert answer.status == "none-of-options"
    assert answer.raw == "C"


def test_single_which_rejects_multiple_answers_while_multi_selects_them() -> None:
    analysis = WhichAnalysis(
        consistent=True,
        reference="R",
        directions=frozenset({Direction.NORTHEAST}),
        candidates=("A", "B", "C"),
        possible_entities=("A", "B"),
        entailed_entities=("A", "B"),
    )
    multiple = encode_menu_answer(
        resolve_answer(analysis, AnswerMode.ALL_POSSIBLE),
        {"A": "A", "B": "B", "C": "C", "D": "Cannot be determined"},
    )
    single = encode_menu_answer(
        resolve_answer(analysis, AnswerMode.SINGLE),
        {"A": "A", "B": "B", "C": "C", "D": "Cannot be determined"},
    )

    assert multiple.raw == "A,B"
    assert multiple.is_resolved
    assert single.status == "undetermined"
    assert single.raw == "D"


def test_single_which_with_a_contingent_member_is_undetermined() -> None:
    analysis = WhichAnalysis(
        consistent=True,
        reference="R",
        directions=frozenset({Direction.NORTHEAST}),
        candidates=("A", "B"),
        possible_entities=("A", "B"),
        entailed_entities=("A",),
    )
    resolution = resolve_answer(analysis, AnswerMode.SINGLE)

    answer = encode_menu_answer(
        resolution,
        {"A": "A", "B": "B", "C": "Cannot be determined"},
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

    exact = resolve_answer(analysis, AnswerMode.SINGLE)
    possible = resolve_answer(analysis, AnswerMode.ALL_POSSIBLE)
    answer = encode_menu_answer(
        possible,
        {"A": "1", "B": "2", "C": "3"},
    )

    assert exact.status is ResolutionStatus.AMBIGUOUS
    assert possible.values == (1, 3)
    assert answer.raw == "A,C"


def test_single_missing_scalar_uses_explicit_none_of_options() -> None:
    analysis = CountAnalysis(
        consistent=True,
        reference="R",
        directions=frozenset({Direction.NORTHEAST}),
        candidates=("A",),
        possible_counts=(1,),
    )
    resolution = resolve_answer(analysis, AnswerMode.SINGLE)

    answer = encode_menu_answer(
        resolution,
        {"A": "0", "B": "None of the options"},
    )

    assert answer.status == "none-of-options"
    assert answer.raw == "B"


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
        resolve_answer(analysis, AnswerMode.ALL_POSSIBLE),
        {"A": "A", "B": "B"},
    )

    partial = score_response("Answer: A", expected)
    exact = score_response("Answer: B, A", expected)

    assert not partial.exact_match
    assert partial.precision == 1.0
    assert partial.recall == 0.5
    assert partial.f1 == 2 / 3
    assert exact.exact_match


def test_symbolic_answer_decision_round_trips_and_rejects_tampering() -> None:
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTHEAST,),
    )
    resolution = resolve_answer(analysis, AnswerMode.SINGLE)
    expected = encode_menu_answer(
        resolution,
        {"A": "North", "B": "Northeast"},
    )
    rendered = render_symbolic_answer_decision(resolution, expected)
    parsed = parse_symbolic_answer_decision(rendered)

    check_symbolic_answer_decision(parsed, expected)
    payload = json.loads(rendered)
    payload["select"] = ["A"]
    tampered = parse_symbolic_answer_decision(json.dumps(payload))
    with pytest.raises(AnswerDecisionCheckError, match="does not match"):
        check_symbolic_answer_decision(tampered, expected)


@pytest.mark.parametrize("mode", tuple(AnswerMode))
def test_menu_permutation_preserves_semantic_selection_for_every_answer_mode(mode):
    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(Direction.NORTH, Direction.SOUTH),
    )
    resolution = resolve_answer(analysis, mode)
    labels = ("North", "Cannot be determined", "East", "South", "None of the Options")
    expected_values = (
        {"Cannot be determined"} if mode is AnswerMode.SINGLE else {"North", "South"}
    )
    for offset in range(len(labels)):
        rotated = labels[offset:] + labels[:offset]
        options = dict(zip("ABCDE", rotated))
        answer = encode_menu_answer(resolution, options)
        assert {options[letter] for letter in answer.letters} == expected_values
        assert score_response(answer.letters, answer).exact_match
