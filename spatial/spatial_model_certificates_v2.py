"""Constructive model and countermodel certificates for spatial claims."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from spatial_solver_v2 import (
    And,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialFormula,
    SpatialProblem,
    direction_between,
)


class ModelCheckError(ValueError):
    """A coordinate certificate does not establish its declared claim status."""


@dataclass(frozen=True)
class CoordinateAssignment:
    object: str
    x: int
    y: int


@dataclass(frozen=True)
class SpatialModelCertificate:
    problem: SpatialProblem
    claim: SpatialFormula
    assignments: tuple[CoordinateAssignment, ...]
    expected_claim_value: bool

    @property
    def coordinates(self) -> dict[str, tuple[int, int]]:
        return {
            assignment.object: (assignment.x, assignment.y)
            for assignment in self.assignments
        }


@dataclass(frozen=True)
class ContingencyCertificate:
    problem: SpatialProblem
    claim: SpatialFormula
    witness: SpatialModelCertificate
    counterexample: SpatialModelCertificate


def _evaluate(
    formula: SpatialFormula,
    coordinates: Mapping[str, tuple[int, int]],
) -> bool:
    if isinstance(formula, RelationConstraint):
        try:
            direction = direction_between(
                coordinates[formula.subject],
                coordinates[formula.reference],
            )
        except KeyError as exc:
            raise ModelCheckError(
                f"formula references an unassigned object: {exc.args[0]}"
            ) from exc
        return direction in formula.allowed
    if isinstance(formula, Not):
        return not _evaluate(formula.operand, coordinates)
    if isinstance(formula, And):
        return all(_evaluate(operand, coordinates) for operand in formula.operands)
    if isinstance(formula, Or):
        return any(_evaluate(operand, coordinates) for operand in formula.operands)
    if isinstance(formula, Implies):
        return not _evaluate(formula.antecedent, coordinates) or _evaluate(
            formula.consequent,
            coordinates,
        )
    if isinstance(formula, Iff):
        return _evaluate(formula.left, coordinates) == _evaluate(
            formula.right,
            coordinates,
        )
    raise TypeError(f"unsupported spatial formula: {type(formula).__name__}")


def evaluate_formula_in_model(
    formula: SpatialFormula,
    coordinates: Mapping[str, tuple[int, int]],
) -> bool:
    """Evaluate one supported formula in an already validated coordinate model."""
    return _evaluate(formula, coordinates)


def check_model_certificate(certificate: SpatialModelCertificate) -> None:
    """Validate one model or countermodel without consulting an SMT solver."""
    problem = certificate.problem
    if type(certificate.expected_claim_value) is not bool:
        raise ModelCheckError("expected_claim_value must be Boolean")
    names = tuple(assignment.object for assignment in certificate.assignments)
    if len(names) != len(set(names)):
        raise ModelCheckError("coordinate assignments must use unique object names")
    if set(names) != set(problem.objects):
        raise ModelCheckError(
            "coordinate assignments must cover exactly the problem objects"
        )
    if any(
        type(value) is not int
        for assignment in certificate.assignments
        for value in (assignment.x, assignment.y)
    ):
        raise ModelCheckError("coordinates must be integers")
    coordinates = certificate.coordinates
    if len(set(coordinates.values())) != len(coordinates):
        raise ModelCheckError("distinct objects cannot share both coordinates")
    if not _evaluate(problem.premise, coordinates):
        raise ModelCheckError("coordinate assignment does not satisfy the premises")
    actual = _evaluate(certificate.claim, coordinates)
    if actual is not certificate.expected_claim_value:
        expected = "true" if certificate.expected_claim_value else "false"
        raise ModelCheckError(
            f"coordinate assignment does not make the claim {expected}"
        )


def check_contingency_certificate(certificate: ContingencyCertificate) -> None:
    """Validate supporting and falsifying models for the same claim."""
    for label, model, expected in (
        ("witness", certificate.witness, True),
        ("counterexample", certificate.counterexample, False),
    ):
        if model.problem != certificate.problem or model.claim != certificate.claim:
            raise ModelCheckError(
                f"{label} does not certify the declared problem and claim"
            )
        if model.expected_claim_value is not expected:
            raise ModelCheckError(f"{label} has the wrong expected claim value")
        check_model_certificate(model)


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_serialize(item) for item in value)
    return value


def model_certificate_to_dict(certificate: SpatialModelCertificate) -> dict[str, Any]:
    check_model_certificate(certificate)
    payload = _serialize(certificate)
    assert isinstance(payload, dict)
    return {"type": type(certificate).__name__, **payload}


def contingency_certificate_to_dict(
    certificate: ContingencyCertificate,
) -> dict[str, Any]:
    check_contingency_certificate(certificate)
    payload = _serialize(certificate)
    assert isinstance(payload, dict)
    return {"type": type(certificate).__name__, **payload}
