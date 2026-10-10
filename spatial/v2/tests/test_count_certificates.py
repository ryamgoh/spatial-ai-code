"""Contracts for correlation-preserving Count answer-set certificates."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from spatial.v2.certificate_generation import build_answer_certificate
from spatial.v2.count_certificate_renderers import render_count_training_trace
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
from spatial.v2.symbolic_trace_codec import parse_symbolic_count_trace
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


def test_oracle_assisted_builder_constructs_correlated_refutations() -> None:
    problem, _ = correlated_problem()
    solver = SpatialSolverV2("reference")
    analysis = solver.analyze(problem)

    certificate = build_answer_certificate(problem, analysis, solver)

    assert isinstance(certificate, CountAnswerSetCertificate)
    assert certificate.possible_counts == (1,)
    for count in (0, 2):
        evidence = certificate.values[count].evidence
        assert isinstance(evidence, CountImpossibilityCertificate)
        assert isinstance(
            evidence.assignments[0].refutation, FormulaRefutationCertificate
        )


def test_symbolic_count_trace_replays_fixed_memberships() -> None:
    problem = SpatialProblem(
        objects=("A", "B", "R"),
        premise=And(
            (
                atom("A", Direction.NORTHEAST, "R"),
                atom("B", Direction.SOUTHWEST, "R"),
            )
        ),
        query=CountQuery(
            frozenset({Direction.NORTHEAST}),
            "R",
            ("A", "B"),
        ),
    )
    solver = SpatialSolverV2("reference")
    certificate = build_answer_certificate(problem, solver.analyze(problem), solver)

    symbolic = render_count_training_trace(certificate, TraceFormat.SYMBOLIC)
    natural = render_count_training_trace(certificate, TraceFormat.NATURAL)
    parsed = parse_symbolic_count_trace(problem, symbolic)

    assert parsed.possible_counts == (1,)
    assert tuple(item.candidate for item in parsed.fixed_memberships) == ("A", "B")
    assert all(
        not item.evidence.assignments
        for item in parsed.values
        if isinstance(item.evidence, CountImpossibilityCertificate)
    )
    assert "Fixed membership: A is entailed" in natural
    assert "Fixed membership: B is impossible" in natural
    assert "No joint assignment of this size agrees" in natural


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

    natural = render_count_training_trace(
        certificate,
        TraceFormat.NATURAL,
    )
    symbolic = render_count_training_trace(
        certificate,
        TraceFormat.SYMBOLIC,
    )
    parsed = parse_symbolic_count_trace(problem, symbolic)

    assert "Count 0: impossible" in natural
    assert "membership assignment {}" in natural
    assert natural.index("X west to east:") < natural.index("Count 1: possible")
    assert '"schema":"spatial-count-trace-v2"' in symbolic
    assert parsed.possible_counts == (1,)
    assert "unique count is 1" in natural


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


def test_compact_count_preserves_correlated_contingent_assignments() -> None:
    problem = SpatialProblem(
        objects=("A", "B", "C", "D", "R"),
        premise=And(
            (
                atom("A", Direction.NORTHEAST, "R"),
                atom("B", Direction.SOUTHWEST, "R"),
                Iff(
                    atom("C", Direction.NORTHEAST, "R"),
                    Not(atom("D", Direction.NORTHEAST, "R")),
                ),
            )
        ),
        query=CountQuery(frozenset({Direction.NORTHEAST}), "R", ("A", "B", "C", "D")),
    )
    solver = SpatialSolverV2("z3")
    certificate = build_answer_certificate(problem, solver.analyze(problem), solver)
    assert certificate.possible_counts == (2,)
    assert tuple(item.candidate for item in certificate.fixed_memberships) == ("A", "B")
    assert certificate.values[1].evidence.assignments[0].members == ("A",)
    assert certificate.values[3].evidence.assignments[0].members == ("A", "C", "D")
    assert (
        sum(
            len(item.evidence.assignments)
            for item in certificate.values
            if isinstance(item.evidence, CountImpossibilityCertificate)
        )
        == 2
    )
    symbolic = render_count_training_trace(certificate, TraceFormat.SYMBOLIC)
    assert parse_symbolic_count_trace(problem, symbolic) == certificate

    # Individual possibility of C and D does not establish joint count 1 or 3.
    missing = replace(certificate.values[1], evidence=CountImpossibilityCertificate(()))
    with pytest.raises(CountCertificateCheckError, match="every membership assignment"):
        check_count_answer_set(
            replace(
                certificate,
                values=(certificate.values[0], missing, *certificate.values[2:]),
            )
        )
    # Fixed evidence cannot simply be omitted to bypass exhaustive coverage.
    with pytest.raises(CountCertificateCheckError, match="every membership assignment"):
        check_count_answer_set(replace(certificate, fixed_memberships=()))
    with pytest.raises(CountCertificateCheckError, match="unique and in query order"):
        check_count_answer_set(
            replace(certificate, fixed_memberships=certificate.fixed_memberships * 2)
        )
    corrupted = replace(certificate.fixed_memberships[0], candidate="C")
    with pytest.raises(CountCertificateCheckError):
        check_count_answer_set(replace(certificate, fixed_memberships=(corrupted,)))


def test_count_compact_parser_rejects_claim_injection_and_old_schema() -> None:
    problem, _ = correlated_problem()
    solver = SpatialSolverV2("z3")
    certificate = build_answer_certificate(problem, solver.analyze(problem), solver)
    payload = json.loads(render_count_training_trace(certificate, TraceFormat.SYMBOLIC))
    payload["evidence"][1]["evidence"]["claim"] = {"kind": "and", "operands": []}
    with pytest.raises(ValueError, match="unexpected or missing fields"):
        parse_symbolic_count_trace(problem, json.dumps(payload))
    del payload["evidence"][1]["evidence"]["claim"]
    payload["schema"] = "spatial-count-trace-v1"
    with pytest.raises(ValueError, match="unsupported symbolic Count trace schema"):
        parse_symbolic_count_trace(problem, json.dumps(payload))
