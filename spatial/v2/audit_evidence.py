"""Structured model-theoretic evidence for spatial audit reports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from spatial.v2.difficulty import AxisProof, derive_axis_proof
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    CountAnalysis,
    CountQuery,
    Direction,
    DirectionAnalysis,
    DirectionQuery,
    QueryAnalysis,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
    WhichAnalysis,
    WhichQuery,
    direction_between,
)

Coordinates = dict[str, tuple[int, int]]


class ClaimStatus(str, Enum):
    ENTAILED = "entailed"
    CONTINGENT = "contingent"
    IMPOSSIBLE = "impossible"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True)
class ClaimAuditEvidence:
    status: ClaimStatus
    witness: Coordinates | None = None
    counterexample: Coordinates | None = None
    axis_proof: AxisProof | None = None
    negation_unsatisfiable: bool = False


@dataclass(frozen=True)
class DirectionAuditCase:
    direction: Direction
    evidence: ClaimAuditEvidence


@dataclass(frozen=True)
class DirectionAuditEvidence:
    target: str
    reference: str
    cases: tuple[DirectionAuditCase, ...]
    engine: str

    @property
    def possible_directions(self) -> tuple[Direction, ...]:
        return tuple(
            case.direction
            for case in self.cases
            if case.evidence.status in {ClaimStatus.ENTAILED, ClaimStatus.CONTINGENT}
        )


@dataclass(frozen=True)
class MembershipAuditEvidence:
    candidate: str
    evidence: ClaimAuditEvidence
    possible_directions: tuple[Direction, ...]


@dataclass(frozen=True)
class WhichAuditEvidence:
    reference: str
    directions: frozenset[Direction]
    memberships: tuple[MembershipAuditEvidence, ...]
    engine: str


@dataclass(frozen=True)
class CountAuditCase:
    count: int
    status: ClaimStatus
    members: tuple[str, ...] = ()
    witness: Coordinates | None = None


@dataclass(frozen=True)
class CountAuditEvidence:
    reference: str
    directions: frozenset[Direction]
    memberships: tuple[MembershipAuditEvidence, ...]
    counts: tuple[CountAuditCase, ...]
    engine: str

    @property
    def possible_counts(self) -> tuple[int, ...]:
        return tuple(
            case.count
            for case in self.counts
            if case.status in {ClaimStatus.ENTAILED, ClaimStatus.CONTINGENT}
        )


QueryAuditEvidence = DirectionAuditEvidence | WhichAuditEvidence | CountAuditEvidence


def audit_evidence_to_dict(evidence: QueryAuditEvidence) -> dict[str, Any]:
    """Return a deterministic JSON-compatible representation for audits."""
    return tagged_dataclass_to_dict(evidence)


def _claim_status(consistent: bool, possible: bool, entailed: bool) -> ClaimStatus:
    if not consistent:
        return ClaimStatus.INCONSISTENT
    if entailed:
        return ClaimStatus.ENTAILED
    if possible:
        return ClaimStatus.CONTINGENT
    return ClaimStatus.IMPOSSIBLE


class AuditEvidenceBuilder:
    """Build typed audit evidence from structured problems and solver results."""

    def __init__(self, solver: SpatialSolverV2 | None = None) -> None:
        self._solver = solver or SpatialSolverV2()

    def build(
        self,
        problem: SpatialProblem,
        analysis: QueryAnalysis | None = None,
    ) -> QueryAuditEvidence:
        analysis = analysis or self._solver.analyze(problem)
        if isinstance(problem.query, DirectionQuery) and isinstance(
            analysis, DirectionAnalysis
        ):
            return self._build_direction(problem, analysis)
        if isinstance(problem.query, WhichQuery) and isinstance(
            analysis, WhichAnalysis
        ):
            return self._build_which(problem, analysis)
        if isinstance(problem.query, CountQuery) and isinstance(
            analysis, CountAnalysis
        ):
            return self._build_count(problem, analysis)
        raise TypeError("analysis type does not match the problem query")

    def _evidence(
        self,
        problem: SpatialProblem,
        claim: RelationConstraint,
        axis_proof: AxisProof | None = None,
    ) -> ClaimAuditEvidence:
        assessed = self._solver.assess(problem, claim)
        if assessed.error:
            raise RuntimeError(assessed.error)
        status = _claim_status(
            assessed.consistent,
            assessed.possible,
            assessed.entailed,
        )
        return ClaimAuditEvidence(
            status=status,
            witness=assessed.witness,
            counterexample=assessed.counterexample,
            axis_proof=axis_proof if status is ClaimStatus.ENTAILED else None,
            negation_unsatisfiable=status is ClaimStatus.ENTAILED,
        )

    def _build_direction(
        self,
        problem: SpatialProblem,
        analysis: DirectionAnalysis,
    ) -> DirectionAuditEvidence:
        query = problem.query
        assert isinstance(query, DirectionQuery)
        cases = []
        for direction in Direction:
            if direction not in query.candidate_directions:
                continue
            claim = RelationConstraint(
                query.target,
                query.reference,
                frozenset({direction}),
            )
            cases.append(
                DirectionAuditCase(
                    direction,
                    self._evidence(
                        problem,
                        claim,
                        derive_axis_proof(
                            problem,
                            query.target,
                            query.reference,
                            direction,
                        ),
                    ),
                )
            )
        return DirectionAuditEvidence(
            query.target,
            query.reference,
            tuple(cases),
            analysis.engine,
        )

    def _membership_evidence(
        self,
        problem: SpatialProblem,
        query: WhichQuery | CountQuery,
    ) -> tuple[MembershipAuditEvidence, ...]:
        memberships = []
        for candidate in query.candidates:
            direction_problem = SpatialProblem(
                problem.objects,
                problem.premise,
                DirectionQuery(candidate, query.reference),
            )
            direction_analysis = self._solver.analyze(direction_problem)
            assert isinstance(direction_analysis, DirectionAnalysis)
            if direction_analysis.error:
                raise RuntimeError(direction_analysis.error)

            possible_directions = direction_analysis.possible_directions
            matching = tuple(
                direction
                for direction in possible_directions
                if direction in query.directions
            )
            nonmatching = tuple(
                direction
                for direction in possible_directions
                if direction not in query.directions
            )
            status = _claim_status(
                direction_analysis.consistent,
                bool(matching),
                direction_analysis.consistent and not nonmatching,
            )
            requested = (
                next(iter(query.directions)) if len(query.directions) == 1 else None
            )
            memberships.append(
                MembershipAuditEvidence(
                    candidate,
                    ClaimAuditEvidence(
                        status=status,
                        witness=(
                            direction_analysis.witnesses[matching[0]]
                            if matching
                            else None
                        ),
                        counterexample=(
                            direction_analysis.witnesses[nonmatching[0]]
                            if nonmatching
                            else None
                        ),
                        axis_proof=(
                            derive_axis_proof(
                                problem,
                                candidate,
                                query.reference,
                                requested,
                            )
                            if status is ClaimStatus.ENTAILED and requested is not None
                            else None
                        ),
                        negation_unsatisfiable=status is ClaimStatus.ENTAILED,
                    ),
                    possible_directions,
                )
            )
        return tuple(memberships)

    def _build_which(
        self,
        problem: SpatialProblem,
        analysis: WhichAnalysis,
    ) -> WhichAuditEvidence:
        query = problem.query
        assert isinstance(query, WhichQuery)
        return WhichAuditEvidence(
            query.reference,
            query.directions,
            self._membership_evidence(problem, query),
            analysis.engine,
        )

    def _build_count(
        self,
        problem: SpatialProblem,
        analysis: CountAnalysis,
    ) -> CountAuditEvidence:
        query = problem.query
        assert isinstance(query, CountQuery)
        possible = set(analysis.possible_counts)
        unique = len(possible) == 1
        cases = []
        for count in range(len(query.candidates) + 1):
            if not analysis.consistent:
                status = ClaimStatus.INCONSISTENT
            elif count not in possible:
                status = ClaimStatus.IMPOSSIBLE
            elif unique:
                status = ClaimStatus.ENTAILED
            else:
                status = ClaimStatus.CONTINGENT
            witness = analysis.witnesses.get(count)
            members = (
                tuple(
                    candidate
                    for candidate in query.candidates
                    if direction_between(witness[candidate], witness[query.reference])
                    in query.directions
                )
                if witness is not None
                else ()
            )
            cases.append(CountAuditCase(count, status, members, witness))
        return CountAuditEvidence(
            query.reference,
            query.directions,
            self._membership_evidence(problem, query),
            tuple(cases),
            analysis.engine,
        )
