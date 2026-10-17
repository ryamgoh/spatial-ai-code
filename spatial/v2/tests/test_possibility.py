"""Contracts for set-valued possibility skeletons."""

from __future__ import annotations

import pytest

from spatial.v2.generation import GenerationPolicy, SpatialGeneratorV2
from spatial.v2.grading import AnswerMode
from spatial.v2.solver import (
    And,
    CountQuery,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    WhichQuery,
)


def _policy(**overrides) -> GenerationPolicy:
    return GenerationPolicy(
        answer_mode=AnswerMode.ALL_POSSIBLE,
        semantic_shape="ambiguous",
        ordinary_option_target=5,
        **overrides,
    )


@pytest.mark.parametrize("query_kind", ["direction", "which", "count"])
@pytest.mark.parametrize("skeleton", ["mp", "iff", "dje", "csr"])
@pytest.mark.parametrize("gold_size", [2, 4])
def test_boolean_skeletons_plant_exactly_gold_size_answers(
    query_kind, skeleton, gold_size
) -> None:
    sample = SpatialGeneratorV2(seed=9200).generate(
        _policy(
            query_kind=query_kind,
            possibility_skeleton=skeleton,
            possibility_gold_size=gold_size,
            num_entities=10,
            num_premises=11,
        ),
        max_attempts=60,
    )

    assert sample.generation_provenance.value == "proof-template-first"
    assert sample.menu_answer.status == "possibilities"
    assert len(sample.menu_answer.letters) == gold_size
    assert len(sample.options) == 5
    assert set(sample.menu_answer.letters) <= set(sample.options)


@pytest.mark.parametrize("query_kind", ["direction", "which"])
@pytest.mark.parametrize("gold_size", [2, 4])
def test_exclusion_skeleton_defines_open_world_gold_without_enumerating_it(
    query_kind, gold_size
) -> None:
    sample = SpatialGeneratorV2(seed=9250).generate(
        _policy(
            query_kind=query_kind,
            possibility_skeleton="exclude",
            possibility_gold_size=gold_size,
            possibility_candidate_count=5,
            num_entities=10,
            num_premises=9,
        )
    )

    assert len(sample.menu_answer.letters) == gold_size
    assert len(sample.options) == 5
    assert all(isinstance(premise, (Not, Or)) for premise in sample.problem.premise.operands)


@pytest.mark.parametrize("query_kind", ["direction", "which", "count"])
@pytest.mark.parametrize("depth", [3, 5])
def test_depth_skeletons_leave_multiple_answers_open(query_kind, depth) -> None:
    sample = SpatialGeneratorV2(seed=9100).generate(
        _policy(
            query_kind=query_kind,
            possibility_skeleton="depth",
            possibility_depth=depth,
            num_entities=2 * depth + 4,
            num_premises=2 * depth + 5,
        ),
        max_attempts=60,
    )

    assert len(sample.menu_answer.letters) >= 2
    assert len(sample.options) == 5
    assert set(sample.menu_answer.letters) <= set(sample.options)


@pytest.mark.parametrize("gold_size", [2, 3])
def test_joint_count_skeleton_skips_intermediate_counts(gold_size) -> None:
    sample = SpatialGeneratorV2(seed=9200).generate(
        _policy(
            query_kind="count",
            possibility_skeleton="jcr",
            possibility_gold_size=gold_size,
            num_entities=9,
            num_premises=10,
        ),
        max_attempts=60,
    )

    assert sample.analysis.possible_counts == (0, gold_size)
    assert sample.difficulty["joint_count_dependency"]


def _children(formula):
    if isinstance(formula, (And, Or)):
        return formula.operands
    if isinstance(formula, (Implies, Iff)):
        return (
            (formula.antecedent, formula.consequent)
            if isinstance(formula, Implies)
            else (formula.left, formula.right)
        )
    if isinstance(formula, Not):
        return (formula.operand,)
    return ()


def _query_atom(atom, query) -> bool:
    if not isinstance(atom, RelationConstraint):
        return False
    if isinstance(query, DirectionQuery):
        return atom.subject == query.target and atom.reference == query.reference
    if isinstance(query, (WhichQuery, CountQuery)):
        return (
            atom.subject in query.candidates
            and atom.reference == query.reference
            and bool(atom.allowed & query.directions)
        )
    raise AssertionError(type(query))


def _assert_no_disjunction_enumerates_query_answers(formula, query) -> None:
    if isinstance(formula, Or):
        assert not any(_query_atom(operand, query) for operand in formula.operands)
    for child in _children(formula):
        _assert_no_disjunction_enumerates_query_answers(child, query)


@pytest.mark.parametrize("gold_size", [2, 4])
def test_case_chain_hides_answers_behind_branches_and_multihop_proofs(
    gold_size,
) -> None:
    sample = SpatialGeneratorV2(seed=9300 + gold_size).generate(
        _policy(
            query_kind="which",
            possibility_skeleton="case-chain",
            possibility_gold_size=gold_size,
            possibility_depth=2,
            num_entities=13,
            num_premises=15,
        ),
        max_attempts=60,
    )

    assert len(sample.menu_answer.letters) == gold_size
    assert sample.difficulty["direct_query_relation"] is False
    _assert_no_disjunction_enumerates_query_answers(
        sample.problem.premise, sample.problem.query
    )


@pytest.mark.parametrize("gold_size", [2, 4])
def test_case_count_requires_complete_branch_membership_assignments(gold_size) -> None:
    sample = SpatialGeneratorV2(seed=9400 + gold_size).generate(
        _policy(
            query_kind="count",
            possibility_skeleton="case-count",
            possibility_gold_size=gold_size,
            num_entities=10,
            num_premises=10,
        ),
        max_attempts=60,
    )

    assert len(sample.analysis.possible_counts) == gold_size
    assert len(sample.menu_answer.letters) == gold_size
    assert sample.difficulty["joint_count_dependency"]
    _assert_no_disjunction_enumerates_query_answers(
        sample.problem.premise, sample.problem.query
    )


@pytest.mark.parametrize(
    ("gold_size", "expected_counts"),
    ((2, (0, 5)), (3, (0, 2, 4)), (4, (0, 2, 3, 5))),
)
def test_factor_count_combines_independent_correlated_groups(
    gold_size, expected_counts
) -> None:
    sample = SpatialGeneratorV2(seed=9500 + gold_size).generate(
        _policy(
            query_kind="count",
            possibility_skeleton="factor-count",
            possibility_gold_size=gold_size,
            num_entities=10,
            num_premises=10,
        ),
        max_attempts=100,
    )

    assert sample.analysis.possible_counts == expected_counts
    assert len(sample.menu_answer.letters) == gold_size
    assert sample.difficulty["direct_query_relation"] is False
    assert sample.difficulty["joint_count_dependency"]
    _assert_no_disjunction_enumerates_query_answers(
        sample.problem.premise, sample.problem.query
    )


def test_gold_size_above_four_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 2 and 4"):
        GenerationPolicy(
            answer_mode=AnswerMode.ALL_POSSIBLE,
            query_kind="which",
            semantic_shape="ambiguous",
            possibility_skeleton="mp",
            possibility_gold_size=5,
            num_entities=10,
            num_premises=11,
        )


def test_skeletons_reject_unique_semantics() -> None:
    with pytest.raises(ValueError, match="ambiguous semantics"):
        GenerationPolicy(
            query_kind="direction",
            semantic_shape="unique",
            possibility_skeleton="mp",
            num_entities=7,
            num_premises=8,
        )
