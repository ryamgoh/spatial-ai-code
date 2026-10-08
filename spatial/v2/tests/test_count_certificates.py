"""Contracts for correlation-preserving Count answer-set certificates."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from spatial.v2.count_certificate_renderers import render_count_answer_set
from spatial.v2.count_certificates import (
    CountAnswerSetCertificate,
    CountCertificateCheckError,
    CountImpossibilityCertificate,
    build_count_answer_set,
    check_count_answer_set,
    count_answer_set_to_dict,
    count_assignment_formula,
    exact_count_formula,
)
from spatial.v2.proof_renderers import render_formula
from spatial.v2.proofs import (
    Contradiction,
    FormulaRefutationCertificate,
    ProofConstructionError,
    ProofRule,
    ProofStep,
    check_formula_refutation,
    formula_refutation_to_dict,
)
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    Iff,
    Not,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
)
from spatial.v2.trace import TraceFormat


def atom(subject: str, direction: Direction, reference: str) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


def refute_assignment(
    problem: SpatialProblem,
    members: tuple[str, ...],
) -> FormulaRefutationCertificate:
    query = problem.query
    assert isinstance(query, CountQuery)
    claim = count_assignment_formula(query, members)
    branch = f"refute-count-{len(members)}"
    steps = [
        ProofStep("P1", ProofRule.PREMISE, problem.premise, premise_index=0),
        ProofStep(
            "A0",
            ProofRule.REFUTATION_ASSUMPTION,
            claim,
            branch=branch,
        ),
    ]

    membership = {
        candidate: RelationConstraint(
            candidate,
            query.reference,
            query.directions,
        )
        for candidate in query.candidates
    }
    for index, candidate in enumerate(query.candidates, start=1):
        conclusion = (
            membership[candidate]
            if candidate in members
            else Not(membership[candidate])
        )
        steps.append(
            ProofStep(
                f"E{index}",
                ProofRule.AND_ELIMINATION,
                conclusion,
                ("A0",),
                branch=branch,
            )
        )

    first, second = query.candidates
    if not members:
        derived = membership[first]
        inputs = ("P1", "E2")
        contradiction_inputs = ("E1", "D1")
    else:
        assert members == query.candidates
        derived = Not(membership[second])
        inputs = ("P1", "E1")
        contradiction_inputs = ("E2", "D1")
    steps.extend(
        (
            ProofStep(
                "D1",
                ProofRule.IFF_ELIMINATION,
                derived,
                inputs,
                branch=branch,
            ),
            ProofStep(
                "C1",
                ProofRule.CONTRADICTION,
                Contradiction(),
                contradiction_inputs,
                branch=branch,
            ),
        )
    )
    certificate = FormulaRefutationCertificate(
        problem,
        claim,
        tuple(steps),
        "A0",
        "C1",
    )
    check_formula_refutation(certificate)
    return certificate


def correlated_problem() -> tuple[SpatialProblem, dict[str, tuple[int, int]]]:
    a_member = atom("A", Direction.NORTHEAST, "R")
    b_member = atom("B", Direction.NORTHEAST, "R")
    problem = SpatialProblem(
        objects=("A", "B", "R"),
        premise=Iff(a_member, Not(b_member)),
        query=CountQuery(
            frozenset({Direction.NORTHEAST}),
            "R",
            ("A", "B"),
        ),
    )
    return problem, {"A": (1, 1), "B": (-1, -1), "R": (0, 0)}


def test_correlated_count_certificate_proves_exactly_one() -> None:
    problem, coordinates = correlated_problem()
    count_zero = refute_assignment(problem, ())
    count_two = refute_assignment(problem, ("A", "B"))

    certificate = build_count_answer_set(
        problem,
        possible_models={1: coordinates},
        impossible_refutations={0: (count_zero,), 2: (count_two,)},
    )

    check_count_answer_set(certificate)
    assert certificate.possible_counts == (1,)
    assert certificate.is_unique
    assert isinstance(certificate.values[0].evidence, CountImpossibilityCertificate)
    assert isinstance(certificate.values[2].evidence, CountImpossibilityCertificate)
    json.dumps(count_answer_set_to_dict(certificate))
    json.dumps(formula_refutation_to_dict(count_zero))

    solver_analysis = SpatialSolverV2("reference").analyze(problem)
    assert certificate.possible_counts == solver_analysis.possible_counts


def test_count_renderer_preserves_joint_evidence() -> None:
    problem, coordinates = correlated_problem()
    certificate = build_count_answer_set(
        problem,
        possible_models={1: coordinates},
        impossible_refutations={
            0: (refute_assignment(problem, ()),),
            2: (refute_assignment(problem, ("A", "B")),),
        },
    )

    natural = render_count_answer_set(
        certificate,
        TraceFormat.NATURAL,
        include_coordinates=False,
    )
    symbolic = render_count_answer_set(
        certificate,
        TraceFormat.SYMBOLIC,
        include_coordinates=False,
    )

    assert "Count 0: impossible" in natural
    assert "membership assignment {}" in natural
    assert "unique count is 1" in natural
    assert "Count-Domain: {1}" in symbolic


def test_count_certificate_can_retain_multiple_possible_counts() -> None:
    a_member = atom("A", Direction.NORTHEAST, "R")
    problem = SpatialProblem(
        objects=("A", "B", "R"),
        premise=a_member,
        query=CountQuery(
            frozenset({Direction.NORTHEAST}),
            "R",
            ("A", "B"),
        ),
    )
    count_zero_claim = count_assignment_formula(problem.query, ())
    branch = "refute-count-0"
    count_zero = FormulaRefutationCertificate(
        problem,
        count_zero_claim,
        (
            ProofStep("P1", ProofRule.PREMISE, a_member, premise_index=0),
            ProofStep(
                "A0",
                ProofRule.REFUTATION_ASSUMPTION,
                count_zero_claim,
                branch=branch,
            ),
            ProofStep(
                "E1",
                ProofRule.AND_ELIMINATION,
                Not(a_member),
                ("A0",),
                branch=branch,
            ),
            ProofStep(
                "C1",
                ProofRule.CONTRADICTION,
                Contradiction(),
                ("P1", "E1"),
                branch=branch,
            ),
        ),
        "A0",
        "C1",
    )
    certificate = build_count_answer_set(
        problem,
        possible_models={
            1: {"A": (1, 1), "B": (-1, -1), "R": (0, 0)},
            2: {"A": (1, 1), "B": (2, 2), "R": (0, 0)},
        },
        impossible_refutations={0: (count_zero,)},
    )

    assert certificate.possible_counts == (1, 2)
    assert not certificate.is_unique


def test_count_formulas_enumerate_complete_membership_assignments() -> None:
    query = CountQuery(
        frozenset({Direction.NORTH}),
        "R",
        ("A", "B", "C"),
    )

    assignment = count_assignment_formula(query, ("A", "C"))
    formula = exact_count_formula(query, 2)

    assert isinstance(assignment, And)
    assert len(assignment.operands) == 3
    assert len(formula.operands) == 3

    coarse = RelationConstraint(
        "A",
        "R",
        frozenset({Direction.NORTHWEST, Direction.NORTH, Direction.NORTHEAST}),
    )
    assert "one of {North, Northeast, Northwest}" in render_formula(
        coarse,
        TraceFormat.NATURAL,
    )
    assert "DIR_IN_{NORTH,NORTHEAST,NORTHWEST}(A,R)" == render_formula(
        coarse,
        TraceFormat.SYMBOLIC,
    )


def test_builder_rejects_missing_and_overlapping_count_evidence() -> None:
    problem, coordinates = correlated_problem()

    with pytest.raises(ProofConstructionError, match="missing evidence for count 2"):
        build_count_answer_set(
            problem,
            possible_models={1: coordinates},
            impossible_refutations={0: (refute_assignment(problem, ()),)},
        )

    with pytest.raises(ProofConstructionError, match="both possible and impossible"):
        build_count_answer_set(
            problem,
            possible_models={0: coordinates, 1: coordinates},
            impossible_refutations={
                0: (refute_assignment(problem, ()),),
                2: (refute_assignment(problem, ("A", "B")),),
            },
        )


def test_checker_rejects_missing_assignment_and_corrupted_refutation() -> None:
    problem, coordinates = correlated_problem()
    certificate = build_count_answer_set(
        problem,
        possible_models={1: coordinates},
        impossible_refutations={
            0: (refute_assignment(problem, ()),),
            2: (refute_assignment(problem, ("A", "B")),),
        },
    )
    count_zero = certificate.values[0]
    assert isinstance(count_zero.evidence, CountImpossibilityCertificate)

    with pytest.raises(CountCertificateCheckError, match="every membership assignment"):
        check_count_answer_set(
            replace(
                certificate,
                values=(
                    replace(
                        count_zero,
                        evidence=replace(count_zero.evidence, assignments=()),
                    ),
                    *certificate.values[1:],
                ),
            )
        )

    assignment = count_zero.evidence.assignments[0]
    corrupted_refutation = replace(assignment.refutation, assumption_step="missing")
    corrupted_assignment = replace(assignment, refutation=corrupted_refutation)
    corrupted_count = replace(
        count_zero,
        evidence=replace(count_zero.evidence, assignments=(corrupted_assignment,)),
    )
    with pytest.raises(CountCertificateCheckError, match="invalid refutation"):
        check_count_answer_set(
            CountAnswerSetCertificate(
                problem,
                (corrupted_count, *certificate.values[1:]),
            )
        )
