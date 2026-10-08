"""Contracts for proof-first SpatialEntail Direction certificates."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from spatial.v2.proof_renderers import (
    render_direction_proof,
    render_direction_refutation,
)
from spatial.v2.proofs import (
    AxisFact,
    Contradiction,
    DirectionClaim,
    DirectionProofCertificate,
    OrderRelation,
    ProofAxis,
    ProofCheckError,
    ProofConstructionError,
    ProofRule,
    ProofStep,
    build_direction_proof,
    build_direction_refutation,
    build_formula_refutation,
    check_direction_proof,
    check_direction_refutation,
    check_formula_refutation,
    proof_to_dict,
    refutation_to_dict,
)
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialFormula,
    SpatialProblem,
    direction_signs,
)
from spatial.v2.text import SpatialTextAdapter
from spatial.v2.trace import TraceFormat


def atom(subject: str, direction: Direction, reference: str) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


def premise_steps(*premises: SpatialFormula) -> tuple[ProofStep, ...]:
    return tuple(
        ProofStep(f"P{index + 1}", ProofRule.PREMISE, premise, premise_index=index)
        for index, premise in enumerate(premises)
    )


def certificate_from_atom_step(
    problem: SpatialProblem,
    steps: tuple[ProofStep, ...],
    atom_step: str,
    conclusion: RelationConstraint,
) -> DirectionProofCertificate:
    direction = next(iter(conclusion.allowed))
    x_sign, y_sign = direction_signs(direction)
    relation = {
        -1: OrderRelation.LESS,
        0: OrderRelation.EQUAL,
        1: OrderRelation.GREATER,
    }
    x_step = ProofStep(
        "Q-X",
        ProofRule.DIRECTION_DECOMPOSITION,
        AxisFact(
            ProofAxis.X,
            conclusion.subject,
            relation[x_sign],
            conclusion.reference,
        ),
        (atom_step,),
    )
    y_step = ProofStep(
        "Q-Y",
        ProofRule.DIRECTION_DECOMPOSITION,
        AxisFact(
            ProofAxis.Y,
            conclusion.subject,
            relation[y_sign],
            conclusion.reference,
        ),
        (atom_step,),
    )
    final = ProofStep(
        "Q-DIR",
        ProofRule.DIRECTION_RECOMPOSITION,
        DirectionClaim(conclusion.subject, direction, conclusion.reference),
        (x_step.id, y_step.id),
    )
    return DirectionProofCertificate(
        problem,
        (*steps, x_step, y_step, final),
        final.id,
    )


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


@pytest.mark.parametrize("actual", list(Direction))
def test_refutation_certificate_excludes_every_other_direction(
    actual: Direction,
) -> None:
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=atom("A", actual, "B"),
        query=DirectionQuery("A", "B"),
    )

    for candidate in Direction:
        if candidate is actual:
            continue
        refutation = build_direction_refutation(problem, candidate)
        check_direction_refutation(refutation)
        assert next(iter(refutation.claim.allowed)) is candidate
        assert refutation.support_premise_indices == (0,)
        assert "Impossible:" in render_direction_refutation(
            refutation,
            TraceFormat.SYMBOLIC,
        )
        json.dumps(refutation_to_dict(refutation))


def test_refutation_builder_rejects_a_possible_direction() -> None:
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=atom("A", Direction.NORTHEAST, "B"),
        query=DirectionQuery("A", "B"),
    )

    with pytest.raises(ProofConstructionError, match="do not refute Northeast"):
        build_direction_refutation(problem, Direction.NORTHEAST)


def test_automatic_builder_applies_modus_ponens() -> None:
    antecedent = atom("A", Direction.NORTH, "B")
    consequent = atom("C", Direction.SOUTHWEST, "D")
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And((antecedent, Implies(antecedent, consequent))),
        query=DirectionQuery("C", "D"),
    )

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.SOUTHWEST
    assert any(step.rule is ProofRule.MODUS_PONENS for step in proof.steps)


def test_automatic_builder_accepts_parsed_controlled_boolean_text() -> None:
    prompt = (
        "Consider a map with multiple locations:\n\n"
        "A is to the North of B. "
        "IF (A is to the North of B) THEN (C is to the East of D).\n\n"
        "Question: In which direction is C relative to D?"
    )
    problem = SpatialTextAdapter().parse(prompt).problem

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.EAST
    assert any(step.rule is ProofRule.MODUS_PONENS for step in proof.steps)


def test_automatic_builder_constructs_conjunctive_antecedent() -> None:
    north = atom("A", Direction.NORTH, "B")
    east = atom("C", Direction.EAST, "D")
    consequent = atom("E", Direction.SOUTHWEST, "F")
    problem = SpatialProblem(
        objects=("A", "B", "C", "D", "E", "F"),
        premise=And((north, east, Implies(And((north, east)), consequent))),
        query=DirectionQuery("E", "F"),
    )

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.SOUTHWEST
    assert any(step.rule is ProofRule.AND_INTRODUCTION for step in proof.steps)
    assert any(step.rule is ProofRule.MODUS_PONENS for step in proof.steps)


def test_boolean_derived_atom_feeds_axis_transitivity() -> None:
    antecedent = atom("A", Direction.NORTHEAST, "B")
    derived = atom("B", Direction.NORTH, "C")
    problem = SpatialProblem(
        objects=("A", "B", "C"),
        premise=And((antecedent, Implies(antecedent, derived))),
        query=DirectionQuery("A", "C"),
    )

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.NORTHEAST
    assert any(step.rule is ProofRule.MODUS_PONENS for step in proof.steps)
    assert any(step.rule is ProofRule.AXIS_TRANSITIVITY for step in proof.steps)


def test_automatic_builder_applies_iff_and_double_negation() -> None:
    left = atom("A", Direction.EAST, "B")
    right = atom("C", Direction.SOUTH, "D")
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And((Iff(left, right), Not(Not(left)))),
        query=DirectionQuery("C", "D"),
    )

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.SOUTH
    assert any(step.rule is ProofRule.DOUBLE_NEGATION for step in proof.steps)
    assert any(step.rule is ProofRule.IFF_ELIMINATION for step in proof.steps)


def test_automatic_builder_applies_disjunctive_syllogism() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=And((Or((northeast, northwest)), Not(northwest))),
        query=DirectionQuery("A", "B"),
    )

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.NORTHEAST
    assert any(step.rule is ProofRule.DISJUNCTIVE_SYLLOGISM for step in proof.steps)


def test_automatic_builder_performs_complete_case_split() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    east = atom("C", Direction.EAST, "D")
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And(
            (
                Or((northeast, northwest)),
                Implies(northeast, east),
                Implies(northwest, east),
            )
        ),
        query=DirectionQuery("C", "D"),
    )

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.EAST
    assert any(step.rule is ProofRule.CASE_SPLIT for step in proof.steps)
    assert sum(step.rule is ProofRule.ASSUMPTION for step in proof.steps) == 2


def test_automatic_case_split_closes_contradictory_branch() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    southwest = atom("A", Direction.SOUTHWEST, "B")
    east = atom("C", Direction.EAST, "D")
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And(
            (
                Or((northeast, northwest, southwest)),
                Implies(northwest, Not(northwest)),
                Implies(northeast, east),
                Implies(southwest, east),
            )
        ),
        query=DirectionQuery("C", "D"),
    )

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.EAST
    assert any(step.rule is ProofRule.CONTRADICTION for step in proof.steps)
    assert any(step.rule is ProofRule.EXPLOSION for step in proof.steps)
    assert any(step.rule is ProofRule.CASE_SPLIT for step in proof.steps)


def test_automatic_direction_refutation_uses_boolean_contradiction() -> None:
    south = atom("A", Direction.SOUTH, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=Implies(south, Not(south)),
        query=DirectionQuery("A", "B"),
    )

    refutation = build_direction_refutation(problem, Direction.SOUTH)

    assert any(step.rule is ProofRule.MODUS_PONENS for step in refutation.steps)
    assert refutation.steps[-1].rule is ProofRule.CONTRADICTION


def test_automatic_formula_refutation_preserves_boolean_correlation() -> None:
    a_member = atom("A", Direction.NORTHEAST, "R")
    b_member = atom("B", Direction.NORTHEAST, "R")
    claim = And((Not(a_member), Not(b_member)))
    problem = SpatialProblem(
        objects=("A", "B", "R"),
        premise=Iff(a_member, Not(b_member)),
        query=DirectionQuery("A", "R"),
    )

    refutation = build_formula_refutation(problem, claim)

    check_formula_refutation(refutation)
    assert refutation.claim == claim
    assert any(step.rule is ProofRule.IFF_ELIMINATION for step in refutation.steps)
    assert refutation.steps[-1].rule is ProofRule.CONTRADICTION

    invalid_claim = replace(
        refutation,
        claim=atom("missing", Direction.NORTH, "R"),
    )
    with pytest.raises(ProofCheckError, match="invalid refutation claim"):
        check_formula_refutation(invalid_claim)


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


def test_disjunctive_syllogism_derives_a_spatial_atom_and_direction() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=And((Or((northeast, northwest)), Not(northwest))),
        query=DirectionQuery("A", "B"),
    )
    proof = certificate_from_atom_step(
        problem,
        (
            *premise_steps(*problem.premise.operands),
            ProofStep(
                "S1",
                ProofRule.DISJUNCTIVE_SYLLOGISM,
                northeast,
                ("P1", "P2"),
            ),
        ),
        "S1",
        northeast,
    )

    check_direction_proof(proof)
    natural = render_direction_proof(proof, TraceFormat.NATURAL)
    symbolic = render_direction_proof(proof, TraceFormat.SYMBOLIC)
    assert "by disjunctive syllogism" in natural
    assert "[disjunctive-syllogism P1,P2]" in symbolic
    assert proof.conclusion.direction is Direction.NORTHEAST


def test_modus_ponens_derives_a_spatial_atom() -> None:
    antecedent = atom("A", Direction.NORTH, "B")
    consequent = atom("C", Direction.SOUTHWEST, "D")
    implication = Implies(antecedent, consequent)
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And((antecedent, implication)),
        query=DirectionQuery("C", "D"),
    )
    proof = certificate_from_atom_step(
        problem,
        (
            *premise_steps(antecedent, implication),
            ProofStep("S1", ProofRule.MODUS_PONENS, consequent, ("P1", "P2")),
        ),
        "S1",
        consequent,
    )

    check_direction_proof(proof)
    assert proof.conclusion.direction is Direction.SOUTHWEST


def test_iff_elimination_derives_the_other_spatial_atom() -> None:
    left = atom("A", Direction.EAST, "B")
    right = atom("C", Direction.SOUTH, "D")
    equivalence = Iff(left, right)
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And((equivalence, left)),
        query=DirectionQuery("C", "D"),
    )
    proof = certificate_from_atom_step(
        problem,
        (
            *premise_steps(equivalence, left),
            ProofStep("S1", ProofRule.IFF_ELIMINATION, right, ("P1", "P2")),
        ),
        "S1",
        right,
    )

    check_direction_proof(proof)
    assert proof.conclusion.direction is Direction.SOUTH


def test_conjunction_rules_and_double_negation_are_replayable() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    west = atom("C", Direction.WEST, "D")
    conjunction = And((northeast, west))
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And((northeast, west, Not(Not(northeast)))),
        query=DirectionQuery("A", "B"),
    )
    proof = certificate_from_atom_step(
        problem,
        (
            *premise_steps(northeast, west, problem.premise.operands[2]),
            ProofStep("S1", ProofRule.AND_INTRODUCTION, conjunction, ("P1", "P2")),
            ProofStep("S2", ProofRule.AND_ELIMINATION, northeast, ("S1",)),
            ProofStep("S3", ProofRule.DOUBLE_NEGATION, northeast, ("P3",)),
        ),
        "S2",
        northeast,
    )

    check_direction_proof(proof)
    assert proof.conclusion.direction is Direction.NORTHEAST


def test_explicit_contradiction_step_is_checked() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    negated = Not(northeast)
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=And((northeast, negated)),
        query=DirectionQuery("A", "B"),
    )
    proof = certificate_from_atom_step(
        problem,
        (
            *premise_steps(northeast, negated),
            ProofStep(
                "C1",
                ProofRule.CONTRADICTION,
                Contradiction(),
                ("P1", "P2"),
            ),
        ),
        "P1",
        northeast,
    )

    check_direction_proof(proof)
    assert "CONTRADICTION" in render_direction_proof(proof, TraceFormat.SYMBOLIC)


def test_checker_rejects_invalid_boolean_rule_applications() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    southwest = atom("C", Direction.SOUTHWEST, "D")
    premises = (
        northeast,
        northwest,
        Or((northeast, northwest)),
        Not(northwest),
        Implies(northeast, southwest),
        Iff(northeast, southwest),
        Not(Not(northeast)),
    )
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And(premises),
        query=DirectionQuery("A", "B"),
    )
    indexed_premises = premise_steps(*premises)
    invalid_steps = (
        (
            ProofStep("BAD", ProofRule.AND_ELIMINATION, northeast, ("P1",)),
            "eliminate one conjunction",
        ),
        (
            ProofStep(
                "BAD",
                ProofRule.AND_INTRODUCTION,
                And((northwest, northeast)),
                ("P1", "P2"),
            ),
            "wrong conjunction",
        ),
        (
            ProofStep("BAD", ProofRule.MODUS_PONENS, northwest, ("P1", "P5")),
            "not a valid modus ponens",
        ),
        (
            ProofStep(
                "BAD",
                ProofRule.DISJUNCTIVE_SYLLOGISM,
                northwest,
                ("P3", "P4"),
            ),
            "wrong remaining disjunct",
        ),
        (
            ProofStep("BAD", ProofRule.IFF_ELIMINATION, northwest, ("P6", "P1")),
            "wrong equivalent formula",
        ),
        (
            ProofStep("BAD", ProofRule.DOUBLE_NEGATION, northwest, ("P7",)),
            "not a valid double-negation",
        ),
        (
            ProofStep("BAD", ProofRule.CONTRADICTION, Contradiction(), ("P1", "P2")),
            "not an explicit contradiction",
        ),
    )

    for invalid_step, message in invalid_steps:
        proof = certificate_from_atom_step(
            problem,
            (*indexed_premises, invalid_step),
            "P1",
            northeast,
        )
        with pytest.raises(ProofCheckError, match=message):
            check_direction_proof(proof)


def test_case_split_recombines_two_spatial_proof_branches() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    east = atom("C", Direction.EAST, "D")
    premises = (
        Or((northeast, northwest)),
        Implies(northeast, east),
        Implies(northwest, east),
    )
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And(premises),
        query=DirectionQuery("C", "D"),
    )
    proof = certificate_from_atom_step(
        problem,
        (
            *premise_steps(*premises),
            ProofStep(
                "B1-A",
                ProofRule.ASSUMPTION,
                northeast,
                ("P1",),
                branch="northeast-case",
            ),
            ProofStep(
                "B1-S",
                ProofRule.MODUS_PONENS,
                east,
                ("B1-A", "P2"),
                branch="northeast-case",
            ),
            ProofStep(
                "B2-A",
                ProofRule.ASSUMPTION,
                northwest,
                ("P1",),
                branch="northwest-case",
            ),
            ProofStep(
                "B2-S",
                ProofRule.MODUS_PONENS,
                east,
                ("B2-A", "P3"),
                branch="northwest-case",
            ),
            ProofStep(
                "S1",
                ProofRule.CASE_SPLIT,
                east,
                ("P1", "B1-S", "B2-S"),
            ),
        ),
        "S1",
        east,
    )

    check_direction_proof(proof)
    natural = render_direction_proof(proof, TraceFormat.NATURAL)
    symbolic = render_direction_proof(proof, TraceFormat.SYMBOLIC)
    assert "[northeast-case] B1-A: Assume" in natural
    assert "Every case from P1 concludes" in natural
    assert "branch=northwest-case" in symbolic
    assert proof.conclusion.direction is Direction.EAST


def test_case_split_can_close_one_branch_by_contradiction() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    east = atom("C", Direction.EAST, "D")
    premises = (
        Or((northeast, northwest)),
        Not(northwest),
        Implies(northeast, east),
    )
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And(premises),
        query=DirectionQuery("C", "D"),
    )
    proof = certificate_from_atom_step(
        problem,
        (
            *premise_steps(*premises),
            ProofStep(
                "B1-A",
                ProofRule.ASSUMPTION,
                northeast,
                ("P1",),
                branch="open-case",
            ),
            ProofStep(
                "B1-S",
                ProofRule.MODUS_PONENS,
                east,
                ("B1-A", "P3"),
                branch="open-case",
            ),
            ProofStep(
                "B2-A",
                ProofRule.ASSUMPTION,
                northwest,
                ("P1",),
                branch="closed-case",
            ),
            ProofStep(
                "B2-C",
                ProofRule.CONTRADICTION,
                Contradiction(),
                ("B2-A", "P2"),
                branch="closed-case",
            ),
            ProofStep(
                "B2-S",
                ProofRule.EXPLOSION,
                east,
                ("B2-C",),
                branch="closed-case",
            ),
            ProofStep(
                "S1",
                ProofRule.CASE_SPLIT,
                east,
                ("P1", "B1-S", "B2-S"),
            ),
        ),
        "S1",
        east,
    )

    check_direction_proof(proof)
    rendered = render_direction_proof(proof, TraceFormat.NATURAL)
    assert "Branch closed-case is closed" in rendered


def test_checker_rejects_cross_branch_dependencies_and_incomplete_cases() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    east = atom("C", Direction.EAST, "D")
    premises = (
        Or((northeast, northwest)),
        Implies(northeast, east),
        Implies(northwest, east),
    )
    problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And(premises),
        query=DirectionQuery("C", "D"),
    )
    common = (
        *premise_steps(*premises),
        ProofStep(
            "B1-A",
            ProofRule.ASSUMPTION,
            northeast,
            ("P1",),
            branch="case-1",
        ),
        ProofStep(
            "B1-S",
            ProofRule.MODUS_PONENS,
            east,
            ("B1-A", "P2"),
            branch="case-1",
        ),
    )
    cross_branch = certificate_from_atom_step(
        problem,
        (
            *common,
            ProofStep(
                "B2-A",
                ProofRule.ASSUMPTION,
                northwest,
                ("P1",),
                branch="case-2",
            ),
            ProofStep(
                "B2-S",
                ProofRule.MODUS_PONENS,
                east,
                ("B1-A", "P3"),
                branch="case-2",
            ),
        ),
        "B1-S",
        east,
    )
    with pytest.raises(ProofCheckError, match="another proof branch"):
        check_direction_proof(cross_branch)

    incomplete = certificate_from_atom_step(
        problem,
        (
            *common,
            ProofStep(
                "B2-A",
                ProofRule.ASSUMPTION,
                northeast,
                ("P1",),
                branch="case-2",
            ),
            ProofStep(
                "B2-S",
                ProofRule.MODUS_PONENS,
                east,
                ("B2-A", "P2"),
                branch="case-2",
            ),
            ProofStep(
                "S1",
                ProofRule.CASE_SPLIT,
                east,
                ("P1", "B1-S", "B2-S"),
            ),
        ),
        "S1",
        east,
    )
    with pytest.raises(ProofCheckError, match="cover every disjunct"):
        check_direction_proof(incomplete)

    duplicate_premises = (
        Or((northeast, northwest)),
        Implies(northeast, east),
        And((east, northeast)),
    )
    duplicate_problem = SpatialProblem(
        objects=("A", "B", "C", "D"),
        premise=And(duplicate_premises),
        query=DirectionQuery("C", "D"),
    )
    duplicate_assumptions = certificate_from_atom_step(
        duplicate_problem,
        (
            *premise_steps(*duplicate_premises),
            ProofStep(
                "B1-A1",
                ProofRule.ASSUMPTION,
                northeast,
                ("P1",),
                branch="case-1",
            ),
            ProofStep(
                "B1-A2",
                ProofRule.ASSUMPTION,
                northwest,
                ("P1",),
                branch="case-1",
            ),
            ProofStep(
                "B1-S",
                ProofRule.MODUS_PONENS,
                east,
                ("B1-A1", "P2"),
                branch="case-1",
            ),
            ProofStep(
                "B2-S",
                ProofRule.AND_ELIMINATION,
                east,
                ("P3",),
                branch="case-2",
            ),
            ProofStep(
                "S1",
                ProofRule.CASE_SPLIT,
                east,
                ("P1", "B1-S", "B2-S"),
            ),
        ),
        "S1",
        east,
    )
    with pytest.raises(ProofCheckError, match="duplicate assumptions"):
        check_direction_proof(duplicate_assumptions)


def test_proof_builder_handles_repeated_disjuncts_by_case_split() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=Or((northeast, northeast)),
        query=DirectionQuery("A", "B"),
    )

    proof = build_direction_proof(problem)

    assert proof.conclusion.direction is Direction.NORTHEAST
    assert any(step.rule is ProofRule.CASE_SPLIT for step in proof.steps)


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
