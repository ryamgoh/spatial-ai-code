"""Contracts for constructive spatial model and countermodel certificates."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from spatial_explanation_renderers_v2 import TraceFormat
from spatial_model_certificate_renderers_v2 import (
    render_contingency_certificate,
    render_model_certificate,
)
from spatial_model_certificates_v2 import (
    ContingencyCertificate,
    CoordinateAssignment,
    ModelCheckError,
    SpatialModelCertificate,
    check_contingency_certificate,
    check_model_certificate,
    contingency_certificate_to_dict,
    model_certificate_to_dict,
)
from spatial_solver_v2 import (
    And,
    Direction,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialProblem,
    direction_signs,
)


def atom(subject: str, direction: Direction, reference: str) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


def assignments(**coordinates: tuple[int, int]) -> tuple[CoordinateAssignment, ...]:
    return tuple(
        CoordinateAssignment(name, point[0], point[1])
        for name, point in coordinates.items()
    )


@pytest.mark.parametrize("direction", list(Direction))
def test_model_certificate_supports_all_eight_directions(direction: Direction) -> None:
    claim = atom("A", direction, "B")
    x_sign, y_sign = direction_signs(direction)
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=claim,
        query=DirectionQuery("A", "B"),
    )
    certificate = SpatialModelCertificate(
        problem,
        claim,
        assignments(A=(x_sign, y_sign), B=(0, 0)),
        True,
    )

    check_model_certificate(certificate)
    json.dumps(model_certificate_to_dict(certificate))


def test_contingency_certificate_checks_supporting_and_falsifying_models() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=Or((northeast, northwest)),
        query=DirectionQuery("A", "B"),
    )
    certificate = ContingencyCertificate(
        problem,
        northeast,
        SpatialModelCertificate(
            problem,
            northeast,
            assignments(A=(1, 1), B=(0, 0)),
            True,
        ),
        SpatialModelCertificate(
            problem,
            northeast,
            assignments(A=(-1, 1), B=(0, 0)),
            False,
        ),
    )

    check_contingency_certificate(certificate)
    assert "Status: contingent" in render_contingency_certificate(
        certificate,
        TraceFormat.SYMBOLIC,
    )
    assert "possible but not entailed" in render_contingency_certificate(
        certificate,
        TraceFormat.NATURAL,
    )
    json.dumps(contingency_certificate_to_dict(certificate))


def test_model_checker_evaluates_the_full_boolean_formula_language() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    southwest = atom("A", Direction.SOUTHWEST, "B")
    premise = And(
        (
            Or((northeast, northwest)),
            Not(southwest),
            Implies(northeast, Not(southwest)),
            Iff(northwest, northwest),
        )
    )
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=premise,
        query=DirectionQuery("A", "B"),
    )
    certificate = SpatialModelCertificate(
        problem,
        northeast,
        assignments(A=(1, 1), B=(0, 0)),
        True,
    )

    check_model_certificate(certificate)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda certificate: replace(
                certificate,
                assignments=certificate.assignments[:1],
            ),
            "cover exactly",
        ),
        (
            lambda certificate: replace(
                certificate,
                assignments=assignments(A=(0, 0), B=(0, 0)),
            ),
            "share both coordinates",
        ),
        (
            lambda certificate: replace(
                certificate,
                assignments=assignments(A=(-1, -1), B=(0, 0)),
            ),
            "does not satisfy the premises",
        ),
        (
            lambda certificate: replace(certificate, expected_claim_value=False),
            "does not make the claim false",
        ),
        (
            lambda certificate: replace(certificate, expected_claim_value=1),
            "must be Boolean",
        ),
    ],
)
def test_model_checker_rejects_corrupted_certificates(mutate, message: str) -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=northeast,
        query=DirectionQuery("A", "B"),
    )
    certificate = SpatialModelCertificate(
        problem,
        northeast,
        assignments(A=(1, 1), B=(0, 0)),
        True,
    )

    with pytest.raises(ModelCheckError, match=message):
        check_model_certificate(mutate(certificate))


def test_contingency_checker_rejects_mismatched_claims() -> None:
    northeast = atom("A", Direction.NORTHEAST, "B")
    northwest = atom("A", Direction.NORTHWEST, "B")
    problem = SpatialProblem(
        objects=("A", "B"),
        premise=Or((northeast, northwest)),
        query=DirectionQuery("A", "B"),
    )
    witness = SpatialModelCertificate(
        problem,
        northeast,
        assignments(A=(1, 1), B=(0, 0)),
        True,
    )
    counterexample = SpatialModelCertificate(
        problem,
        northwest,
        assignments(A=(-1, 1), B=(0, 0)),
        False,
    )

    with pytest.raises(ModelCheckError, match="declared problem and claim"):
        check_contingency_certificate(
            ContingencyCertificate(
                problem,
                northeast,
                witness,
                counterexample,
            )
        )


def test_model_renderer_can_hide_coordinates_without_losing_status() -> None:
    northeast = atom("obj_0", Direction.NORTHEAST, "obj_1")
    problem = SpatialProblem(
        objects=("obj_0", "obj_1"),
        premise=northeast,
        query=DirectionQuery("obj_0", "obj_1"),
    )
    certificate = SpatialModelCertificate(
        problem,
        northeast,
        assignments(obj_0=(1, 1), obj_1=(0, 0)),
        True,
    )

    rendered = render_model_certificate(
        certificate,
        TraceFormat.NATURAL,
        {"obj_0": "Bakery", "obj_1": "Library"},
        include_coordinates=False,
    )

    assert "supporting model" in rendered
    assert "Coordinates" not in rendered
    assert "obj_0" not in rendered
