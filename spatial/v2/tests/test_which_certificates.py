"""Contracts for complete Which membership certificates."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from spatial.v2.model_certificates import ContingencyCertificate
from spatial.v2.proofs import ProofConstructionError
from spatial.v2.solver import (
    And,
    Direction,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
    direction_signs,
)
from spatial.v2.trace import TraceFormat
from spatial.v2.which_certificate_renderers import render_which_answer_set
from spatial.v2.which_certificates import (
    MembershipEntailmentCertificate,
    MembershipImpossibilityCertificate,
    MembershipProofCertificate,
    WhichAnswerSetCertificate,
    WhichCertificateCheckError,
    build_which_answer_set,
    check_which_answer_set,
    which_answer_set_to_dict,
)


def atom(subject: str, direction: Direction, reference: str) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


@pytest.mark.parametrize("direction", list(Direction))
def test_which_entailment_supports_all_eight_directions(
    direction: Direction,
) -> None:
    coordinates = {"A": direction_signs(direction), "R": (0, 0)}
    problem = SpatialProblem(
        objects=("A", "R"),
        premise=atom("A", direction, "R"),
        query=WhichQuery(frozenset({direction}), "R", ("A",)),
    )

    certificate = build_which_answer_set(
        problem,
        positive_models={"A": coordinates},
        negative_models={},
    )

    assert certificate.possible_entities == ("A",)
    assert certificate.entailed_entities == ("A",)
    assert certificate.is_exact_single


def test_unique_which_certificate_classifies_every_candidate() -> None:
    coordinates = {"A": (1, 1), "B": (-1, -1), "R": (0, 0)}
    problem = SpatialProblem(
        objects=("A", "B", "R"),
        premise=And(
            (
                atom("A", Direction.NORTHEAST, "R"),
                atom("B", Direction.SOUTHWEST, "R"),
            )
        ),
        query=WhichQuery(
            frozenset({Direction.NORTHEAST}),
            "R",
            ("A", "B"),
        ),
    )

    certificate = build_which_answer_set(
        problem,
        positive_models={"A": coordinates},
        negative_models={"B": coordinates},
    )

    check_which_answer_set(certificate)
    assert certificate.possible_entities == ("A",)
    assert certificate.entailed_entities == ("A",)
    assert certificate.contingent_entities == ()
    assert certificate.impossible_entities == ("B",)
    assert certificate.is_exact_single
    assert isinstance(
        certificate.candidates[0].evidence,
        MembershipProofCertificate,
    )
    assert isinstance(
        certificate.candidates[1].evidence,
        MembershipImpossibilityCertificate,
    )
    assert len(certificate.candidates[1].evidence.refutations) == 1
    json.dumps(which_answer_set_to_dict(certificate))


def test_contingent_which_membership_requires_two_models() -> None:
    problem = SpatialProblem(
        objects=("N", "R", "S"),
        premise=And(
            (
                atom("S", Direction.SOUTHEAST, "R"),
                atom("N", Direction.SOUTHWEST, "S"),
            )
        ),
        query=WhichQuery(
            frozenset({Direction.NORTHEAST}),
            "N",
            ("R",),
        ),
    )
    northeast = {"N": (0, 0), "R": (1, 2), "S": (2, 1)}
    northwest = {"N": (1, 0), "R": (0, 2), "S": (2, 1)}

    certificate = build_which_answer_set(
        problem,
        positive_models={"R": northeast},
        negative_models={"R": northwest},
    )

    assert certificate.possible_entities == ("R",)
    assert certificate.entailed_entities == ()
    assert certificate.contingent_entities == ("R",)
    assert certificate.impossible_entities == ()
    assert not certificate.is_exact_single
    assert isinstance(certificate.candidates[0].evidence, ContingencyCertificate)
    natural = render_which_answer_set(
        certificate,
        TraceFormat.NATURAL,
        include_coordinates=False,
    )
    symbolic = render_which_answer_set(
        certificate,
        TraceFormat.SYMBOLIC,
        include_coordinates=False,
    )
    assert "R: contingent" in natural
    assert "possible candidates are R" in natural
    assert "Which-Possible: {R}" in symbolic
    assert "Which-Entailed: {}" in symbolic


def test_which_certificate_matches_independent_solver_analysis() -> None:
    problem = SpatialProblem(
        objects=("N", "R", "S"),
        premise=And(
            (
                atom("S", Direction.SOUTHEAST, "R"),
                atom("N", Direction.SOUTHWEST, "S"),
            )
        ),
        query=WhichQuery(
            frozenset({Direction.NORTHEAST}),
            "N",
            ("R", "S"),
        ),
    )
    solver = SpatialSolverV2("reference")
    analysis = solver.analyze(problem)
    positive_models = {}
    negative_models = {}
    for candidate in problem.query.candidates:
        claim = RelationConstraint(
            candidate,
            problem.query.reference,
            problem.query.directions,
        )
        assessment = solver.assess(problem, claim)
        if assessment.witness is not None:
            positive_models[candidate] = assessment.witness
        if assessment.counterexample is not None:
            negative_models[candidate] = assessment.counterexample

    certificate = build_which_answer_set(
        problem,
        positive_models=positive_models,
        negative_models=negative_models,
    )

    assert certificate.possible_entities == analysis.possible_entities
    assert certificate.entailed_entities == analysis.entailed_entities


def test_coarse_membership_is_proved_by_excluding_its_complement() -> None:
    coordinates = {"N": (0, 0), "R": (1, 2), "S": (2, 1)}
    problem = SpatialProblem(
        objects=("N", "R", "S"),
        premise=And(
            (
                atom("S", Direction.SOUTHEAST, "R"),
                atom("N", Direction.SOUTHWEST, "S"),
            )
        ),
        query=WhichQuery(
            frozenset(
                {
                    Direction.NORTHWEST,
                    Direction.NORTH,
                    Direction.NORTHEAST,
                }
            ),
            "N",
            ("R",),
        ),
    )

    certificate = build_which_answer_set(
        problem,
        positive_models={"R": coordinates},
        negative_models={},
    )

    evidence = certificate.candidates[0].evidence
    assert isinstance(evidence, MembershipEntailmentCertificate)
    assert tuple(
        next(iter(refutation.claim.allowed)) for refutation in evidence.refutations
    ) == (
        Direction.EAST,
        Direction.SOUTHEAST,
        Direction.SOUTH,
        Direction.SOUTHWEST,
        Direction.WEST,
    )
    corrupted = replace(
        certificate.candidates[0],
        evidence=replace(evidence, refutations=evidence.refutations[:-1]),
    )
    with pytest.raises(WhichCertificateCheckError, match="complement direction"):
        check_which_answer_set(WhichAnswerSetCertificate(problem, (corrupted,)))


def test_which_certificate_allows_no_possible_candidate() -> None:
    coordinates = {"A": (0, -1), "B": (-1, -1), "R": (0, 0)}
    problem = SpatialProblem(
        objects=("A", "B", "R"),
        premise=And(
            (
                atom("A", Direction.SOUTH, "R"),
                atom("B", Direction.SOUTHWEST, "R"),
            )
        ),
        query=WhichQuery(
            frozenset(
                {
                    Direction.NORTHWEST,
                    Direction.NORTH,
                    Direction.NORTHEAST,
                }
            ),
            "R",
            ("A", "B"),
        ),
    )

    certificate = build_which_answer_set(
        problem,
        positive_models={},
        negative_models={"A": coordinates, "B": coordinates},
    )

    assert certificate.possible_entities == ()
    assert certificate.entailed_entities == ()
    assert certificate.impossible_entities == ("A", "B")
    assert not certificate.is_exact_single


def test_builder_rejects_missing_or_undeclared_membership_models() -> None:
    problem = SpatialProblem(
        objects=("N", "R", "S"),
        premise=And(
            (
                atom("S", Direction.SOUTHEAST, "R"),
                atom("N", Direction.SOUTHWEST, "S"),
            )
        ),
        query=WhichQuery(
            frozenset({Direction.NORTHEAST}),
            "N",
            ("R",),
        ),
    )
    northeast = {"N": (0, 0), "R": (1, 2), "S": (2, 1)}

    with pytest.raises(ProofConstructionError, match="countermodel.*R"):
        build_which_answer_set(
            problem,
            positive_models={"R": northeast},
            negative_models={},
        )

    northwest = {"N": (1, 0), "R": (0, 2), "S": (2, 1)}
    with pytest.raises(ProofConstructionError, match="positive model.*R"):
        build_which_answer_set(
            problem,
            positive_models={},
            negative_models={"R": northwest},
        )

    with pytest.raises(ProofConstructionError, match="undeclared.*S"):
        build_which_answer_set(
            problem,
            positive_models={"S": northeast},
            negative_models={},
        )


def test_checker_rejects_incomplete_and_corrupted_which_evidence() -> None:
    coordinates = {"A": (1, 1), "B": (-1, -1), "R": (0, 0)}
    problem = SpatialProblem(
        objects=("A", "B", "R"),
        premise=And(
            (
                atom("A", Direction.NORTHEAST, "R"),
                atom("B", Direction.SOUTHWEST, "R"),
            )
        ),
        query=WhichQuery(
            frozenset({Direction.NORTHEAST}),
            "R",
            ("A", "B"),
        ),
    )
    certificate = build_which_answer_set(
        problem,
        positive_models={"A": coordinates},
        negative_models={"B": coordinates},
    )

    with pytest.raises(WhichCertificateCheckError, match="every declared candidate"):
        check_which_answer_set(
            replace(certificate, candidates=certificate.candidates[:-1])
        )

    entailed = certificate.candidates[0]
    assert isinstance(entailed.evidence, MembershipProofCertificate)
    corrupted = replace(
        entailed,
        evidence=replace(
            entailed.evidence,
            proof=replace(
                entailed.evidence.proof,
                conclusion_step="missing",
            ),
        ),
    )
    with pytest.raises(WhichCertificateCheckError, match="invalid direction proof"):
        check_which_answer_set(
            WhichAnswerSetCertificate(
                problem,
                (corrupted, certificate.candidates[1]),
            )
        )
