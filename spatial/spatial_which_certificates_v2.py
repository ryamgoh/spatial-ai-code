"""Complete three-valued certificates for Which-query membership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from spatial_model_certificates_v2 import (
    ContingencyCertificate,
    CoordinateAssignment,
    ModelCheckError,
    SpatialModelCertificate,
    check_contingency_certificate,
    check_model_certificate,
)
from spatial_proofs_v2 import (
    DirectionProofCertificate,
    DirectionRefutationCertificate,
    ProofCheckError,
    ProofConstructionError,
    build_direction_proof,
    build_direction_refutation,
    check_direction_proof,
    check_direction_refutation,
)
from spatial_serialization_v2 import tagged_dataclass_to_dict
from spatial_solver_v2 import (
    Direction,
    DirectionQuery,
    RelationConstraint,
    SpatialProblem,
    WhichQuery,
)


class WhichCertificateCheckError(ValueError):
    """A Which certificate does not establish its declared membership statuses."""


class MembershipStatus(str, Enum):
    ENTAILED = "entailed"
    CONTINGENT = "contingent"
    IMPOSSIBLE = "impossible"


@dataclass(frozen=True)
class MembershipProofCertificate:
    witness: SpatialModelCertificate
    proof: DirectionProofCertificate


@dataclass(frozen=True)
class MembershipEntailmentCertificate:
    witness: SpatialModelCertificate
    refutations: tuple[DirectionRefutationCertificate, ...]


@dataclass(frozen=True)
class MembershipImpossibilityCertificate:
    countermodel: SpatialModelCertificate
    refutations: tuple[DirectionRefutationCertificate, ...]


MembershipEvidence = (
    MembershipProofCertificate
    | MembershipEntailmentCertificate
    | ContingencyCertificate
    | MembershipImpossibilityCertificate
)


@dataclass(frozen=True)
class WhichCandidateCertificate:
    candidate: str
    evidence: MembershipEvidence

    @property
    def status(self) -> MembershipStatus:
        if isinstance(
            self.evidence,
            (MembershipProofCertificate, MembershipEntailmentCertificate),
        ):
            return MembershipStatus.ENTAILED
        if isinstance(self.evidence, ContingencyCertificate):
            return MembershipStatus.CONTINGENT
        if isinstance(self.evidence, MembershipImpossibilityCertificate):
            return MembershipStatus.IMPOSSIBLE
        raise TypeError(
            f"unsupported membership evidence: {type(self.evidence).__name__}"
        )


@dataclass(frozen=True)
class WhichAnswerSetCertificate:
    problem: SpatialProblem
    candidates: tuple[WhichCandidateCertificate, ...]

    @property
    def possible_entities(self) -> tuple[str, ...]:
        return tuple(
            item.candidate
            for item in self.candidates
            if item.status is not MembershipStatus.IMPOSSIBLE
        )

    @property
    def entailed_entities(self) -> tuple[str, ...]:
        return tuple(
            item.candidate
            for item in self.candidates
            if item.status is MembershipStatus.ENTAILED
        )

    @property
    def contingent_entities(self) -> tuple[str, ...]:
        return tuple(
            item.candidate
            for item in self.candidates
            if item.status is MembershipStatus.CONTINGENT
        )

    @property
    def impossible_entities(self) -> tuple[str, ...]:
        return tuple(
            item.candidate
            for item in self.candidates
            if item.status is MembershipStatus.IMPOSSIBLE
        )

    @property
    def is_exact_single(self) -> bool:
        return (
            len(self.entailed_entities) == 1
            and self.possible_entities == self.entailed_entities
        )


def _membership_claim(query: WhichQuery, candidate: str) -> RelationConstraint:
    return RelationConstraint(candidate, query.reference, query.directions)


def _direction_projection(
    problem: SpatialProblem,
    query: WhichQuery,
    candidate: str,
) -> SpatialProblem:
    return SpatialProblem(
        problem.objects,
        problem.premise,
        DirectionQuery(candidate, query.reference),
    )


def _canonical_directions(directions: frozenset[Direction]) -> tuple[Direction, ...]:
    return tuple(direction for direction in Direction if direction in directions)


def _check_model(
    model: SpatialModelCertificate,
    problem: SpatialProblem,
    claim: RelationConstraint,
    expected_value: bool,
    candidate: str,
) -> None:
    if (
        model.problem != problem
        or model.claim != claim
        or model.expected_claim_value is not expected_value
    ):
        raise WhichCertificateCheckError(
            f"{candidate} model certifies the wrong membership claim"
        )
    try:
        check_model_certificate(model)
    except ModelCheckError as exc:
        raise WhichCertificateCheckError(
            f"{candidate} has an invalid membership model: {exc}"
        ) from exc


def _check_refutations(
    refutations: tuple[DirectionRefutationCertificate, ...],
    projection: SpatialProblem,
    expected_directions: frozenset[Direction],
    candidate: str,
    description: str,
) -> None:
    if any(len(refutation.claim.allowed) != 1 for refutation in refutations):
        raise WhichCertificateCheckError(
            f"{candidate} has a refutation without one exact direction"
        )
    actual_directions = tuple(
        next(iter(refutation.claim.allowed)) for refutation in refutations
    )
    if actual_directions != _canonical_directions(expected_directions):
        raise WhichCertificateCheckError(
            f"{candidate} must refute every {description} direction exactly once"
        )
    for refutation in refutations:
        if refutation.problem != projection:
            raise WhichCertificateCheckError(
                f"{candidate} refutation uses the wrong Direction projection"
            )
        try:
            check_direction_refutation(refutation)
        except ProofCheckError as exc:
            raise WhichCertificateCheckError(
                f"{candidate} has an invalid direction refutation: {exc}"
            ) from exc


def check_which_answer_set(certificate: WhichAnswerSetCertificate) -> None:
    """Replay complete membership evidence without consulting an SMT solver."""
    problem = certificate.problem
    query = problem.query
    if not isinstance(query, WhichQuery):
        raise WhichCertificateCheckError("Which certificates require a WhichQuery")
    actual_candidates = tuple(item.candidate for item in certificate.candidates)
    if actual_candidates != query.candidates:
        raise WhichCertificateCheckError(
            "membership evidence must cover every declared candidate exactly once "
            "in query order"
        )

    all_directions = frozenset(Direction)
    for item in certificate.candidates:
        claim = _membership_claim(query, item.candidate)
        projection = _direction_projection(problem, query, item.candidate)
        if isinstance(item.evidence, MembershipProofCertificate):
            _check_model(item.evidence.witness, problem, claim, True, item.candidate)
            if item.evidence.proof.problem != projection:
                raise WhichCertificateCheckError(
                    f"{item.candidate} proof uses the wrong Direction projection"
                )
            try:
                check_direction_proof(item.evidence.proof)
            except ProofCheckError as exc:
                raise WhichCertificateCheckError(
                    f"{item.candidate} has an invalid direction proof: {exc}"
                ) from exc
            if item.evidence.proof.conclusion.direction not in query.directions:
                raise WhichCertificateCheckError(
                    f"{item.candidate} proof concludes a non-matching direction"
                )
        elif isinstance(item.evidence, MembershipEntailmentCertificate):
            _check_model(item.evidence.witness, problem, claim, True, item.candidate)
            _check_refutations(
                item.evidence.refutations,
                projection,
                all_directions - query.directions,
                item.candidate,
                "complement",
            )
        elif isinstance(item.evidence, ContingencyCertificate):
            if item.evidence.problem != problem or item.evidence.claim != claim:
                raise WhichCertificateCheckError(
                    f"{item.candidate} contingency certifies the wrong membership claim"
                )
            try:
                check_contingency_certificate(item.evidence)
            except ModelCheckError as exc:
                raise WhichCertificateCheckError(
                    f"{item.candidate} has invalid contingency evidence: {exc}"
                ) from exc
        elif isinstance(item.evidence, MembershipImpossibilityCertificate):
            _check_model(
                item.evidence.countermodel,
                problem,
                claim,
                False,
                item.candidate,
            )
            _check_refutations(
                item.evidence.refutations,
                projection,
                query.directions,
                item.candidate,
                "matching",
            )
        else:
            raise WhichCertificateCheckError(
                f"{item.candidate} has unsupported membership evidence"
            )


def _model_certificate(
    problem: SpatialProblem,
    claim: RelationConstraint,
    coordinates: Mapping[str, tuple[int, int]],
    expected_value: bool,
    candidate: str,
) -> SpatialModelCertificate:
    supplied = set(coordinates)
    expected = set(problem.objects)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ProofConstructionError(
            f"{candidate} membership model has wrong object coverage: "
            + "; ".join(details)
        )
    model = SpatialModelCertificate(
        problem,
        claim,
        tuple(
            CoordinateAssignment(name, *coordinates[name]) for name in problem.objects
        ),
        expected_value,
    )
    try:
        check_model_certificate(model)
    except ModelCheckError as exc:
        raise ProofConstructionError(
            f"invalid {candidate} membership model: {exc}"
        ) from exc
    return model


def _build_refutations(
    projection: SpatialProblem,
    directions: frozenset[Direction],
) -> tuple[DirectionRefutationCertificate, ...]:
    return tuple(
        build_direction_refutation(projection, direction)
        for direction in _canonical_directions(directions)
    )


def build_which_answer_set(
    problem: SpatialProblem,
    *,
    positive_models: Mapping[str, Mapping[str, tuple[int, int]]],
    negative_models: Mapping[str, Mapping[str, tuple[int, int]]],
) -> WhichAnswerSetCertificate:
    """Build complete Which evidence from proposed models and checked exclusions."""
    query = problem.query
    if not isinstance(query, WhichQuery):
        raise ProofConstructionError("Which construction requires a WhichQuery")
    declared = set(query.candidates)
    unknown = (set(positive_models) | set(negative_models)) - declared
    if unknown:
        raise ProofConstructionError(
            "models supplied for undeclared candidates: " + ", ".join(sorted(unknown))
        )

    all_directions = frozenset(Direction)
    candidates = []
    for candidate in query.candidates:
        claim = _membership_claim(query, candidate)
        projection = _direction_projection(problem, query, candidate)
        positive_coordinates = positive_models.get(candidate)
        negative_coordinates = negative_models.get(candidate)
        positive = (
            _model_certificate(
                problem,
                claim,
                positive_coordinates,
                True,
                candidate,
            )
            if positive_coordinates is not None
            else None
        )
        negative = (
            _model_certificate(
                problem,
                claim,
                negative_coordinates,
                False,
                candidate,
            )
            if negative_coordinates is not None
            else None
        )

        if positive is not None and negative is not None:
            evidence: MembershipEvidence = ContingencyCertificate(
                problem,
                claim,
                positive,
                negative,
            )
        elif positive is not None:
            try:
                proof = build_direction_proof(projection)
            except ProofConstructionError:
                try:
                    refutations = _build_refutations(
                        projection,
                        all_directions - query.directions,
                    )
                except ProofConstructionError as exc:
                    raise ProofConstructionError(
                        f"missing a countermodel for contingent candidate {candidate}"
                    ) from exc
                evidence = MembershipEntailmentCertificate(positive, refutations)
            else:
                if proof.conclusion.direction not in query.directions:
                    raise ProofConstructionError(
                        f"positive model for {candidate} conflicts with its direction proof"
                    )
                evidence = MembershipProofCertificate(positive, proof)
        elif negative is not None:
            try:
                refutations = _build_refutations(projection, query.directions)
            except ProofConstructionError as exc:
                raise ProofConstructionError(
                    f"missing a positive model for non-refutable candidate {candidate}"
                ) from exc
            evidence = MembershipImpossibilityCertificate(negative, refutations)
        else:
            raise ProofConstructionError(
                f"candidate {candidate} needs a positive or negative membership model"
            )
        candidates.append(WhichCandidateCertificate(candidate, evidence))

    certificate = WhichAnswerSetCertificate(problem, tuple(candidates))
    check_which_answer_set(certificate)
    return certificate


def which_answer_set_to_dict(
    certificate: WhichAnswerSetCertificate,
) -> dict[str, Any]:
    check_which_answer_set(certificate)
    return tagged_dataclass_to_dict(certificate)
