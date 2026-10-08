"""Contract tests for structured V2 explanations."""

from __future__ import annotations

import json

import pytest
from spatial_audit_rendering_v2 import render_audit_explanation
from spatial_explanations_v2 import (
    Axis,
    AxisRelation,
    ClaimStatus,
    CountExplanation,
    DirectionExplanation,
    SpatialExplainerV2,
    WhichExplanation,
    explanation_to_dict,
)
from spatial_solver_v2 import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    Iff,
    Or,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
)


def atom(subject: str, direction: Direction, reference: str) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


@pytest.mark.parametrize("backend", ["reference", "z3"])
def test_claim_assessment_retains_positive_and_negative_models(backend: str) -> None:
    if backend == "z3":
        pytest.importorskip("z3")
    solver = SpatialSolverV2(backend=backend)
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=Or((northeast, northwest)),
        query=DirectionQuery("A", "B"),
    )

    assessment = solver.assess(problem, northeast)

    assert assessment.consistent
    assert assessment.possible
    assert not assessment.entailed
    assert assessment.witness is not None
    assert assessment.counterexample is not None


def test_direction_explanation_extracts_axis_transitivity_proof() -> None:
    problem = SpatialProblem(
        objects=("A", "B", "C"),
        premise=And(
            (
                atom("A", Direction.NORTHEAST, "B"),
                atom("B", Direction.NORTH, "C"),
            )
        ),
        query=DirectionQuery("A", "C"),
    )

    explanation = SpatialExplainerV2().explain(problem)

    assert isinstance(explanation, DirectionExplanation)
    assert explanation.possible_directions == (Direction.NORTHEAST,)
    northeast = next(
        case for case in explanation.cases if case.direction is Direction.NORTHEAST
    )
    assert northeast.evidence.status is ClaimStatus.ENTAILED
    assert northeast.evidence.negation_unsatisfiable
    assert northeast.evidence.axis_proof is not None
    assert northeast.evidence.axis_proof.x.axis is Axis.X
    assert northeast.evidence.axis_proof.x.relation is AxisRelation.GREATER
    assert northeast.evidence.axis_proof.x.path == ("A", "B", "C")
    assert northeast.evidence.axis_proof.x.premise_indices == (0, 1)
    assert northeast.evidence.axis_proof.y.path == ("A", "B", "C")


def test_propositional_entailment_uses_solver_certificate_fallback() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=Or((northeast, northeast)),
        query=DirectionQuery("A", "B"),
    )

    explanation = SpatialExplainerV2().explain(problem)

    assert isinstance(explanation, DirectionExplanation)
    case = next(
        item for item in explanation.cases if item.direction is Direction.NORTHEAST
    )
    assert case.evidence.status is ClaimStatus.ENTAILED
    assert case.evidence.axis_proof is None
    assert case.evidence.negation_unsatisfiable
    assert (
        next(
            item for item in explanation.cases if item.direction is Direction.NORTHWEST
        ).evidence.status
        is ClaimStatus.IMPOSSIBLE
    )


def test_ambiguous_direction_explanation_has_two_sided_evidence() -> None:
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=RelationConstraint(
            "A",
            "B",
            frozenset({Direction.NORTHEAST, Direction.NORTHWEST}),
        ),
        query=DirectionQuery("A", "B"),
    )

    explanation = SpatialExplainerV2().explain(problem)

    assert isinstance(explanation, DirectionExplanation)
    assert explanation.possible_directions == (
        Direction.NORTHEAST,
        Direction.NORTHWEST,
    )
    for case in explanation.cases:
        if case.direction in explanation.possible_directions:
            assert case.evidence.status is ClaimStatus.CONTINGENT
            assert case.evidence.witness is not None
            assert case.evidence.counterexample is not None


def test_which_explanation_classifies_each_candidate() -> None:
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And(
            (
                atom("A", Direction.NORTHEAST, "B"),
                RelationConstraint(
                    "C",
                    "B",
                    frozenset({Direction.NORTHEAST, Direction.NORTHWEST}),
                ),
                atom("D", Direction.SOUTHWEST, "B"),
            )
        ),
        query=WhichQuery(
            frozenset({Direction.NORTHEAST}),
            "B",
            ("A", "C", "D"),
        ),
    )

    explanation = SpatialExplainerV2().explain(problem)

    assert isinstance(explanation, WhichExplanation)
    statuses = {
        membership.candidate: membership.evidence.status
        for membership in explanation.memberships
    }
    assert statuses == {
        "A": ClaimStatus.ENTAILED,
        "C": ClaimStatus.CONTINGENT,
        "D": ClaimStatus.IMPOSSIBLE,
    }
    contingent = next(
        membership
        for membership in explanation.memberships
        if membership.candidate == "C"
    )
    assert contingent.possible_directions == (
        Direction.NORTHEAST,
        Direction.NORTHWEST,
    )
    assert contingent.evidence.witness is not None
    assert contingent.evidence.counterexample is not None


def test_count_explanation_preserves_correlated_non_contiguous_counts() -> None:
    a_member = atom("A", Direction.NORTHEAST, "R")
    b_member = atom("B", Direction.NORTHEAST, "R")
    problem = SpatialProblem(
        objects=("A", "B", "R"),
        premise=Iff(a_member, b_member),
        query=CountQuery(
            frozenset({Direction.NORTHEAST}),
            "R",
            ("A", "B"),
        ),
    )

    explanation = SpatialExplainerV2().explain(problem)

    assert isinstance(explanation, CountExplanation)
    assert explanation.possible_counts == (0, 2)
    assert [case.status for case in explanation.counts] == [
        ClaimStatus.CONTINGENT,
        ClaimStatus.IMPOSSIBLE,
        ClaimStatus.CONTINGENT,
    ]
    assert [case.members for case in explanation.counts] == [(), (), ("A", "B")]
    assert all(
        membership.evidence.status is ClaimStatus.CONTINGENT
        for membership in explanation.memberships
    )


def test_renderer_uses_optional_display_labels_without_dataset_knowledge() -> None:
    problem = SpatialProblem(
        objects=("obj_0", "obj_1"),
        premise=atom("obj_0", Direction.NORTH, "obj_1"),
        query=DirectionQuery("obj_0", "obj_1"),
    )

    explanation = SpatialExplainerV2().explain(problem)
    rendered = render_audit_explanation(
        explanation,
        {"obj_0": "Bakery", "obj_1": "Library"},
    )

    assert "Bakery is North of Library: entailed." in rendered
    assert "Combining both axes gives North." in rendered
    assert "obj_0" not in rendered

    serialized = explanation_to_dict(explanation)
    assert serialized["type"] == "DirectionExplanation"
    assert serialized["cases"][0]["direction"] == "North"
    assert serialized["cases"][0]["evidence"]["negation_unsatisfiable"] is True
    json.dumps(serialized)
