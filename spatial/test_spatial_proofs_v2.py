"""Contracts for proof-first SpatialEntail Direction certificates."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest
from spatial_explanation_renderers_v2 import TraceFormat
from spatial_proof_renderers_v2 import render_direction_proof
from spatial_proofs_v2 import (
    AxisFact,
    DirectionClaim,
    OrderRelation,
    ProofCheckError,
    ProofConstructionError,
    ProofRule,
    build_direction_proof,
    check_direction_proof,
    proof_to_dict,
)
from spatial_solver_v2 import (
    And,
    Direction,
    DirectionQuery,
    Or,
    RelationConstraint,
    SpatialProblem,
)


def atom(subject: str, direction: Direction, reference: str) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


@pytest.mark.parametrize("direction", list(Direction))
def test_direct_certificate_supports_all_eight_directions(direction: Direction) -> None:
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=atom("A", direction, "B"),
        query=DirectionQuery("A", "B"),
    )

    proof = build_direction_proof(problem)

    check_direction_proof(proof)
    assert proof.conclusion == DirectionClaim("A", direction, "B")
    assert proof.support_premise_indices == (0,)
    assert proof.steps[-1].rule is ProofRule.DIRECTION_RECOMPOSITION
    json.dumps(proof_to_dict(proof))


def test_transitive_certificate_renders_one_checked_proof_in_two_forms() -> None:
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

    proof = build_direction_proof(problem)
    natural = render_direction_proof(proof, TraceFormat.NATURAL)
    symbolic = render_direction_proof(proof, TraceFormat.SYMBOLIC)

    assert proof.conclusion.direction is Direction.NORTHEAST
    assert proof.support_premise_indices == (0, 1)
    assert "By X-axis transitivity" in natural
    assert "By Y-axis transitivity" in natural
    assert "Combining the X and Y conclusions gives A is Northeast of C" in natural
    assert "DIR_NORTHEAST(A,C)" in symbolic
    assert "[direction-recomposition" in symbolic
    assert re.search(r"=\(-?\d+,\s*-?\d+\)", natural + symbolic) is None


def test_certificate_can_use_independent_x_and_y_support() -> None:
    problem = SpatialProblem(
        objects=("A", "B", "C", "R"),
        premise=And(
            (
                atom("B", Direction.NORTHEAST, "R"),
                atom("A", Direction.SOUTHEAST, "B"),
                atom("C", Direction.NORTHEAST, "R"),
                atom("A", Direction.NORTHWEST, "C"),
            )
        ),
        query=DirectionQuery("A", "R"),
    )

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.NORTHEAST
    assert proof.support_premise_indices == (0, 1, 2, 3)
    reachable = {
        step.id for step in proof.steps if step.rule is ProofRule.AXIS_TRANSITIVITY
    }
    assert any(step_id.startswith("D-X-") for step_id in reachable)
    assert any(step_id.startswith("D-Y-") for step_id in reachable)


def test_certificate_excludes_distractor_premises_and_decompositions() -> None:
    problem = SpatialProblem(
        objects=("A", "B", "C"),
        premise=And(
            (
                atom("A", Direction.NORTHEAST, "B"),
                atom("C", Direction.SOUTHWEST, "B"),
            )
        ),
        query=DirectionQuery("A", "B"),
    )

    proof = build_direction_proof(problem)

    premise_steps = [step for step in proof.steps if step.rule is ProofRule.PREMISE]
    assert proof.support_premise_indices == (0,)
    assert [step.premise_index for step in premise_steps] == [0]
    assert all(not step.id.startswith("P2") for step in proof.steps)


def test_proof_builder_rejects_boolean_premises_in_initial_fragment() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=Or((northeast, northeast)),
        query=DirectionQuery("A", "B"),
    )

    with pytest.raises(ProofConstructionError, match="positive conjunctions"):
        build_direction_proof(problem)


def test_checker_rejects_a_tampered_direction_conclusion() -> None:
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=atom("A", Direction.NORTHEAST, "B"),
        query=DirectionQuery("A", "B"),
    )
    proof = build_direction_proof(problem)
    conclusion = proof.steps[-1]
    tampered = replace(
        proof,
        steps=(
            *proof.steps[:-1],
            replace(
                conclusion,
                conclusion=DirectionClaim("A", Direction.NORTHWEST, "B"),
            ),
        ),
    )

    with pytest.raises(ProofCheckError, match="wrong direction"):
        check_direction_proof(tampered)


def test_checker_rejects_a_tampered_transitivity_step() -> None:
    problem = SpatialProblem(
        objects=("A", "B", "C"),
        premise=And(
            (
                atom("A", Direction.NORTHEAST, "B"),
                atom("B", Direction.NORTHEAST, "C"),
            )
        ),
        query=DirectionQuery("A", "C"),
    )
    proof = build_direction_proof(problem)
    index = next(
        index
        for index, step in enumerate(proof.steps)
        if step.rule is ProofRule.AXIS_TRANSITIVITY
    )
    step = proof.steps[index]
    assert isinstance(step.conclusion, AxisFact)
    tampered_steps = list(proof.steps)
    tampered_steps[index] = replace(
        step,
        conclusion=replace(step.conclusion, relation=OrderRelation.EQUAL),
    )
    tampered = replace(proof, steps=tuple(tampered_steps))

    with pytest.raises(ProofCheckError, match="invalid transitivity"):
        check_direction_proof(tampered)


def test_renderer_uses_display_labels_without_changing_certificate() -> None:
    problem = SpatialProblem(
        objects=("obj_0", "obj_1"),
        premise=atom("obj_0", Direction.SOUTHWEST, "obj_1"),
        query=DirectionQuery("obj_0", "obj_1"),
    )
    proof = build_direction_proof(problem)

    rendered = render_direction_proof(
        proof,
        TraceFormat.NATURAL,
        {"obj_0": "Bakery", "obj_1": "Library"},
    )

    assert "Bakery is Southwest of Library" in rendered
    assert "obj_0" not in rendered
