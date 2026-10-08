"""Complete Direction answer-set certificates from models and refutations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spatial_model_certificates_v2 import (
    ModelCheckError,
    SpatialModelCertificate,
    build_model_certificate,
    check_model_certificate,
)
from spatial_proofs_v2 import (
    DirectionRefutationCertificate,
    ProofCheckError,
    ProofConstructionError,
    build_direction_refutation,
    check_direction_refutation,
)
from spatial_serialization_v2 import tagged_dataclass_to_dict
from spatial_solver_v2 import (
    Direction,
    DirectionQuery,
    RelationConstraint,
    SpatialProblem,
)


class AnswerSetCheckError(ValueError):
    """A Direction answer-set certificate is incomplete or inconsistent."""


@dataclass(frozen=True)
class DirectionCandidateCertificate:
    direction: Direction
    evidence: SpatialModelCertificate | DirectionRefutationCertificate

    @property
    def possible(self) -> bool:
        return isinstance(self.evidence, SpatialModelCertificate)


@dataclass(frozen=True)
class DirectionAnswerSetCertificate:
    problem: SpatialProblem
    candidates: tuple[DirectionCandidateCertificate, ...]

    @property
    def possible_directions(self) -> tuple[Direction, ...]:
        return tuple(item.direction for item in self.candidates if item.possible)

    @property
    def is_unique(self) -> bool:
        return len(self.possible_directions) == 1


def _candidate_claim(
    query: DirectionQuery,
    direction: Direction,
) -> RelationConstraint:
    return RelationConstraint(
        query.target,
        query.reference,
        frozenset({direction}),
    )


def check_direction_answer_set(certificate: DirectionAnswerSetCertificate) -> None:
    """Require exhaustive, non-overlapping evidence for every declared candidate."""
    problem = certificate.problem
    query = problem.query
    if not isinstance(query, DirectionQuery):
        raise AnswerSetCheckError("Direction answer sets require a DirectionQuery")
    expected_order = tuple(
        direction for direction in Direction if direction in query.candidate_directions
    )
    actual_order = tuple(item.direction for item in certificate.candidates)
    if actual_order != expected_order:
        raise AnswerSetCheckError(
            "candidate evidence must cover every declared direction exactly once "
            "in canonical order"
        )

    for item in certificate.candidates:
        expected_claim = _candidate_claim(query, item.direction)
        if isinstance(item.evidence, SpatialModelCertificate):
            if (
                item.evidence.problem != problem
                or item.evidence.claim != expected_claim
                or item.evidence.expected_claim_value is not True
            ):
                raise AnswerSetCheckError(
                    f"{item.direction.value} model certifies the wrong candidate"
                )
            try:
                check_model_certificate(item.evidence)
            except ModelCheckError as exc:
                raise AnswerSetCheckError(
                    f"{item.direction.value} has an invalid model: {exc}"
                ) from exc
        elif isinstance(item.evidence, DirectionRefutationCertificate):
            if (
                item.evidence.problem != problem
                or item.evidence.claim != expected_claim
            ):
                raise AnswerSetCheckError(
                    f"{item.direction.value} refutation certifies the wrong candidate"
                )
            try:
                check_direction_refutation(item.evidence)
            except ProofCheckError as exc:
                raise AnswerSetCheckError(
                    f"{item.direction.value} has an invalid refutation: {exc}"
                ) from exc
        else:
            raise AnswerSetCheckError(
                f"{item.direction.value} has unsupported candidate evidence"
            )

    if not certificate.possible_directions:
        raise AnswerSetCheckError(
            "a consistent Direction answer set needs a possible value"
        )


def build_direction_answer_set(
    problem: SpatialProblem,
    possible_models: Mapping[Direction, Mapping[str, tuple[int, int]]],
) -> DirectionAnswerSetCertificate:
    """Build complete evidence from proposed models plus automatic refutations."""
    query = problem.query
    if not isinstance(query, DirectionQuery):
        raise ProofConstructionError("answer-set construction supports DirectionQuery")
    unknown = set(possible_models) - set(query.candidate_directions)
    if unknown:
        names = ", ".join(sorted(direction.value for direction in unknown))
        raise ProofConstructionError(
            f"models supplied for undeclared candidates: {names}"
        )

    evidence = []
    for direction in Direction:
        if direction not in query.candidate_directions:
            continue
        claim = _candidate_claim(query, direction)
        coordinates = possible_models.get(direction)
        if coordinates is not None:
            try:
                model = build_model_certificate(problem, claim, coordinates, True)
            except ModelCheckError as exc:
                raise ProofConstructionError(
                    f"invalid {direction.value} model: {exc}"
                ) from exc
            evidence.append(DirectionCandidateCertificate(direction, model))
            continue
        try:
            refutation = build_direction_refutation(problem, direction)
        except ProofConstructionError as exc:
            raise ProofConstructionError(
                f"missing a model for non-refutable candidate {direction.value}"
            ) from exc
        evidence.append(DirectionCandidateCertificate(direction, refutation))

    certificate = DirectionAnswerSetCertificate(problem, tuple(evidence))
    check_direction_answer_set(certificate)
    return certificate


def answer_set_to_dict(
    certificate: DirectionAnswerSetCertificate,
) -> dict[str, Any]:
    check_direction_answer_set(certificate)
    return tagged_dataclass_to_dict(certificate)
