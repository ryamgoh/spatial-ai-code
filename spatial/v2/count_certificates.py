"""Correlation-preserving certificates for complete Count answer sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from spatial.v2.model_certificates import (
    ModelCheckError,
    SpatialModelCertificate,
    build_model_certificate,
    check_model_certificate,
)
from spatial.v2.proofs import (
    FormulaRefutationCertificate,
    ProofCheckError,
    ProofConstructionError,
    check_formula_refutation,
)
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    And,
    CountQuery,
    Not,
    Or,
    SpatialFormula,
    SpatialProblem,
    WhichQuery,
    membership_constraint,
)
from spatial.v2.which_certificates import (
    MembershipEvidence,
    MembershipStatus,
    WhichCandidateCertificate,
    WhichCertificateCheckError,
    check_which_candidate_certificate,
)


class CountCertificateCheckError(ValueError):
    """A Count certificate does not establish its declared possibility set."""


@dataclass(frozen=True)
class CountAssignmentRefutation:
    members: tuple[str, ...]
    refutation: FormulaRefutationCertificate | CountMembershipConflict


@dataclass(frozen=True)
class CountMembershipConflict:
    candidate: str
    evidence: MembershipEvidence


@dataclass(frozen=True)
class CountImpossibilityCertificate:
    assignments: tuple[CountAssignmentRefutation, ...]


@dataclass(frozen=True)
class CountValueCertificate:
    count: int
    evidence: SpatialModelCertificate | CountImpossibilityCertificate

    @property
    def possible(self) -> bool:
        return isinstance(self.evidence, SpatialModelCertificate)


@dataclass(frozen=True)
class CountAnswerSetCertificate:
    problem: SpatialProblem
    values: tuple[CountValueCertificate, ...]

    @property
    def possible_counts(self) -> tuple[int, ...]:
        return tuple(item.count for item in self.values if item.possible)

    @property
    def is_unique(self) -> bool:
        return len(self.possible_counts) == 1


def count_assignments(
    query: CountQuery,
    count: int,
) -> tuple[tuple[str, ...], ...]:
    """Return candidate subsets of one count in deterministic query order."""
    if not 0 <= count <= len(query.candidates):
        raise ValueError(f"count must be between 0 and {len(query.candidates)}")
    return tuple(combinations(query.candidates, count))


def count_assignment_formula(
    query: CountQuery,
    members: tuple[str, ...],
) -> And:
    """Describe one complete true/false membership assignment."""
    member_set = set(members)
    canonical = tuple(
        candidate for candidate in query.candidates if candidate in member_set
    )
    if members != canonical:
        raise ValueError("count-assignment members must be unique and in query order")
    return And(
        tuple(
            membership_constraint(candidate, query)
            if candidate in member_set
            else Not(membership_constraint(candidate, query))
            for candidate in query.candidates
        )
    )


def exact_count_formula(query: CountQuery, count: int) -> SpatialFormula:
    """Describe all complete membership assignments having exactly one count."""
    alternatives = tuple(
        count_assignment_formula(query, members)
        for members in count_assignments(query, count)
    )
    return alternatives[0] if len(alternatives) == 1 else Or(alternatives)


def check_count_answer_set(certificate: CountAnswerSetCertificate) -> None:
    """Replay exhaustive Count evidence without consulting an SMT solver."""
    problem = certificate.problem
    query = problem.query
    if not isinstance(query, CountQuery):
        raise CountCertificateCheckError("Count certificates require a CountQuery")
    expected_counts = tuple(range(len(query.candidates) + 1))
    actual_counts = tuple(item.count for item in certificate.values)
    if actual_counts != expected_counts:
        raise CountCertificateCheckError(
            "count evidence must cover every value from zero through the candidate "
            "count exactly once"
        )

    for item in certificate.values:
        if isinstance(item.evidence, SpatialModelCertificate):
            claim = exact_count_formula(query, item.count)
            if (
                item.evidence.problem != problem
                or item.evidence.claim != claim
                or item.evidence.expected_claim_value is not True
            ):
                raise CountCertificateCheckError(
                    f"count {item.count} model certifies the wrong count formula"
                )
            try:
                check_model_certificate(item.evidence)
            except ModelCheckError as exc:
                raise CountCertificateCheckError(
                    f"count {item.count} has an invalid model: {exc}"
                ) from exc
            continue

        if not isinstance(item.evidence, CountImpossibilityCertificate):
            raise CountCertificateCheckError(
                f"count {item.count} has unsupported evidence"
            )
        expected_assignments = count_assignments(query, item.count)
        actual_assignments = tuple(
            assignment.members for assignment in item.evidence.assignments
        )
        if actual_assignments != expected_assignments:
            raise CountCertificateCheckError(
                f"count {item.count} must refute every membership assignment "
                "exactly once in canonical order"
            )
        for assignment in item.evidence.assignments:
            expected_claim = count_assignment_formula(query, assignment.members)
            refutation = assignment.refutation
            if isinstance(refutation, FormulaRefutationCertificate):
                if refutation.problem != problem or refutation.claim != expected_claim:
                    raise CountCertificateCheckError(
                        f"count {item.count} refutation certifies the wrong assignment"
                    )
                try:
                    check_formula_refutation(refutation)
                except ProofCheckError as exc:
                    raise CountCertificateCheckError(
                        f"count {item.count} has an invalid refutation: {exc}"
                    ) from exc
                continue

            if not isinstance(refutation, CountMembershipConflict):
                raise CountCertificateCheckError(
                    f"count {item.count} has unsupported assignment evidence"
                )
            which_query = WhichQuery(
                query.directions,
                query.reference,
                query.candidates,
            )
            which_problem = SpatialProblem(
                problem.objects,
                problem.premise,
                which_query,
            )
            candidate_evidence = WhichCandidateCertificate(
                refutation.candidate,
                refutation.evidence,
            )
            try:
                check_which_candidate_certificate(
                    which_problem,
                    which_query,
                    candidate_evidence,
                )
            except WhichCertificateCheckError as exc:
                raise CountCertificateCheckError(
                    f"count {item.count} has an invalid membership conflict: {exc}"
                ) from exc
            assigned_true = refutation.candidate in assignment.members
            status = candidate_evidence.status
            if (assigned_true and status is not MembershipStatus.IMPOSSIBLE) or (
                not assigned_true and status is not MembershipStatus.ENTAILED
            ):
                raise CountCertificateCheckError(
                    f"count {item.count} membership evidence does not contradict "
                    "the assignment"
                )

    if not certificate.possible_counts:
        raise CountCertificateCheckError(
            "a consistent Count answer set needs at least one possible value"
        )


def build_count_answer_set(
    problem: SpatialProblem,
    *,
    possible_models: Mapping[int, Mapping[str, tuple[int, int]]],
    impossible_refutations: Mapping[
        int,
        Sequence[FormulaRefutationCertificate | CountMembershipConflict],
    ],
) -> CountAnswerSetCertificate:
    """Build complete Count evidence from models and assignment refutations."""
    query = problem.query
    if not isinstance(query, CountQuery):
        raise ProofConstructionError("Count construction requires a CountQuery")
    valid_counts = set(range(len(query.candidates) + 1))
    unknown = (set(possible_models) | set(impossible_refutations)) - valid_counts
    if unknown:
        names = ", ".join(str(count) for count in sorted(unknown))
        raise ProofConstructionError(f"evidence supplied for invalid counts: {names}")
    overlap = set(possible_models) & set(impossible_refutations)
    if overlap:
        names = ", ".join(str(count) for count in sorted(overlap))
        raise ProofConstructionError(
            f"counts cannot be both possible and impossible: {names}"
        )

    values = []
    for count in range(len(query.candidates) + 1):
        coordinates = possible_models.get(count)
        refutations = impossible_refutations.get(count)
        if coordinates is not None:
            try:
                evidence: SpatialModelCertificate | CountImpossibilityCertificate = (
                    build_model_certificate(
                        problem,
                        exact_count_formula(query, count),
                        coordinates,
                        True,
                    )
                )
            except ModelCheckError as exc:
                raise ProofConstructionError(
                    f"invalid model for count {count}: {exc}"
                ) from exc
        elif refutations is not None:
            assignments = count_assignments(query, count)
            if len(refutations) != len(assignments):
                raise ProofConstructionError(
                    f"count {count} needs one refutation for every membership assignment"
                )
            evidence = CountImpossibilityCertificate(
                tuple(
                    CountAssignmentRefutation(members, refutation)
                    for members, refutation in zip(assignments, refutations)
                )
            )
        else:
            raise ProofConstructionError(f"missing evidence for count {count}")
        values.append(CountValueCertificate(count, evidence))

    certificate = CountAnswerSetCertificate(problem, tuple(values))
    try:
        check_count_answer_set(certificate)
    except CountCertificateCheckError as exc:
        raise ProofConstructionError(f"invalid Count evidence: {exc}") from exc
    return certificate


def count_answer_set_to_dict(
    certificate: CountAnswerSetCertificate,
) -> dict[str, Any]:
    check_count_answer_set(certificate)
    return tagged_dataclass_to_dict(certificate)
