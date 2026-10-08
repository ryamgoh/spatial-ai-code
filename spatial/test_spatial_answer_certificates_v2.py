"""Contracts for complete Direction answer-set certificates."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from spatial_answer_certificate_renderers_v2 import render_direction_answer_set
from spatial_answer_certificates_v2 import (
    AnswerSetCheckError,
    DirectionAnswerSetCertificate,
    DirectionCandidateCertificate,
    answer_set_to_dict,
    build_direction_answer_set,
    check_direction_answer_set,
)
from spatial_model_certificates_v2 import SpatialModelCertificate
from spatial_proofs_v2 import ProofConstructionError
from spatial_solver_v2 import (
    And,
    Direction,
    DirectionQuery,
    RelationConstraint,
    SpatialProblem,
)
from spatial_trace_v2 import TraceFormat


def atom(subject: str, direction: Direction, reference: str) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


def test_unique_answer_set_covers_all_eight_candidates() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=northeast,
        query=DirectionQuery("A", "B"),
    )

    certificate = build_direction_answer_set(
        problem,
        {Direction.NORTHEAST: {"A": (1, 1), "B": (0, 0)}},
    )

    check_direction_answer_set(certificate)
    assert certificate.possible_directions == (Direction.NORTHEAST,)
    assert certificate.is_unique
    assert len(certificate.candidates) == len(Direction)
    assert sum(not item.possible for item in certificate.candidates) == 7
    json.dumps(answer_set_to_dict(certificate))


def test_ambiguous_ordinal_answer_set_has_two_models_and_two_refutations() -> None:
    objects = ("N", "R", "S")
    problem = SpatialProblem(
        objects=objects,
        premise=And(
            (
                atom("S", Direction.SOUTHEAST, "R"),
                atom("N", Direction.SOUTHWEST, "S"),
            )
        ),
        query=DirectionQuery(
            "R",
            "N",
            frozenset(
                {
                    Direction.NORTHEAST,
                    Direction.NORTHWEST,
                    Direction.SOUTHEAST,
                    Direction.SOUTHWEST,
                }
            ),
        ),
    )
    certificate = build_direction_answer_set(
        problem,
        {
            Direction.NORTHEAST: {
                "N": (0, 0),
                "R": (1, 2),
                "S": (2, 1),
            },
            Direction.NORTHWEST: {
                "N": (1, 0),
                "R": (0, 2),
                "S": (2, 1),
            },
        },
    )

    assert certificate.possible_directions == (
        Direction.NORTHEAST,
        Direction.NORTHWEST,
    )
    assert not certificate.is_unique
    assert sum(item.possible for item in certificate.candidates) == 2
    assert sum(not item.possible for item in certificate.candidates) == 2
    natural = render_direction_answer_set(
        certificate,
        TraceFormat.NATURAL,
        include_coordinates=False,
    )
    symbolic = render_direction_answer_set(
        certificate,
        TraceFormat.SYMBOLIC,
        include_coordinates=False,
    )
    assert "possible directions are Northeast, Northwest" in natural
    assert "Direction-Domain: {Northeast, Northwest}" in symbolic


def test_builder_rejects_a_missing_model_for_a_possible_candidate() -> None:
    problem = SpatialProblem(
        objects=("N", "R", "S"),
        premise=And(
            (
                atom("S", Direction.SOUTHEAST, "R"),
                atom("N", Direction.SOUTHWEST, "S"),
            )
        ),
        query=DirectionQuery(
            "R",
            "N",
            frozenset({Direction.NORTHEAST, Direction.NORTHWEST}),
        ),
    )

    with pytest.raises(ProofConstructionError, match="non-refutable.*Northwest"):
        build_direction_answer_set(
            problem,
            {
                Direction.NORTHEAST: {
                    "N": (0, 0),
                    "R": (1, 2),
                    "S": (2, 1),
                }
            },
        )


def test_builder_rejects_models_for_undeclared_candidates() -> None:
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=atom("A", Direction.NORTH, "B"),
        query=DirectionQuery("A", "B", frozenset({Direction.NORTH})),
    )

    with pytest.raises(ProofConstructionError, match="undeclared.*South"):
        build_direction_answer_set(
            problem,
            {
                Direction.NORTH: {"A": (0, 1), "B": (0, 0)},
                Direction.SOUTH: {"A": (0, -1), "B": (0, 0)},
            },
        )


def test_checker_rejects_incomplete_duplicate_and_mismatched_evidence() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=northeast,
        query=DirectionQuery("A", "B"),
    )
    certificate = build_direction_answer_set(
        problem,
        {Direction.NORTHEAST: {"A": (1, 1), "B": (0, 0)}},
    )

    with pytest.raises(AnswerSetCheckError, match="cover every declared direction"):
        check_direction_answer_set(
            replace(certificate, candidates=certificate.candidates[:-1])
        )

    duplicated = replace(
        certificate,
        candidates=(certificate.candidates[0], *certificate.candidates[:-1]),
    )
    with pytest.raises(AnswerSetCheckError, match="cover every declared direction"):
        check_direction_answer_set(duplicated)

    northeast_evidence = next(
        item for item in certificate.candidates if item.direction is Direction.NORTHEAST
    )
    southwest_refutation = next(
        item.evidence
        for item in certificate.candidates
        if item.direction is Direction.SOUTHWEST
    )
    assert isinstance(northeast_evidence.evidence, SpatialModelCertificate)
    mismatched = DirectionCandidateCertificate(
        Direction.NORTHEAST,
        southwest_refutation,
    )
    mismatched_candidates = tuple(
        mismatched if item.direction is Direction.NORTHEAST else item
        for item in certificate.candidates
    )
    with pytest.raises(AnswerSetCheckError, match="refutation certifies the wrong"):
        check_direction_answer_set(
            DirectionAnswerSetCertificate(problem, mismatched_candidates)
        )
