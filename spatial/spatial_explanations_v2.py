"""Structured, data-agnostic explanations for V2 spatial analyses.

The solver establishes model-theoretic truth. This module turns those results
into typed evidence and optionally renders that evidence as deterministic text.
It does not parse prompts, inspect answer menus, or know dataset schemas.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from spatial_solver_v2 import (
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
    conjunctive_atoms,
    direction_signs,
    membership_constraint,
)

Coordinates = dict[str, tuple[int, int]]


class ClaimStatus(str, Enum):
    ENTAILED = "entailed"
    CONTINGENT = "contingent"
    IMPOSSIBLE = "impossible"
    INCONSISTENT = "inconsistent"


class Axis(str, Enum):
    X = "x"
    Y = "y"


class AxisRelation(str, Enum):
    LESS = "<"
    EQUAL = "="
    GREATER = ">"


@dataclass(frozen=True)
class AxisDerivation:
    axis: Axis
    subject: str
    relation: AxisRelation
    reference: str
    path: tuple[str, ...]
    premise_indices: tuple[int, ...]


@dataclass(frozen=True)
class AxisProof:
    x: AxisDerivation
    y: AxisDerivation
    direction: Direction


@dataclass(frozen=True)
class ClaimEvidence:
    status: ClaimStatus
    witness: Coordinates | None = None
    counterexample: Coordinates | None = None
    axis_proof: AxisProof | None = None
    negation_unsatisfiable: bool = False


@dataclass(frozen=True)
class DirectionCase:
    direction: Direction
    evidence: ClaimEvidence


@dataclass(frozen=True)
class DirectionExplanation:
    target: str
    reference: str
    cases: tuple[DirectionCase, ...]
    engine: str

    @property
    def possible_directions(self) -> tuple[Direction, ...]:
        return tuple(
            case.direction
            for case in self.cases
            if case.evidence.status in {ClaimStatus.ENTAILED, ClaimStatus.CONTINGENT}
        )


@dataclass(frozen=True)
class MembershipExplanation:
    candidate: str
    evidence: ClaimEvidence


@dataclass(frozen=True)
class WhichExplanation:
    reference: str
    directions: frozenset[Direction]
    memberships: tuple[MembershipExplanation, ...]
    engine: str


@dataclass(frozen=True)
class CountCase:
    count: int
    status: ClaimStatus
    witness: Coordinates | None = None


@dataclass(frozen=True)
class CountExplanation:
    reference: str
    directions: frozenset[Direction]
    memberships: tuple[MembershipExplanation, ...]
    counts: tuple[CountCase, ...]
    engine: str

    @property
    def possible_counts(self) -> tuple[int, ...]:
        return tuple(
            case.count
            for case in self.counts
            if case.status in {ClaimStatus.ENTAILED, ClaimStatus.CONTINGENT}
        )


QueryExplanation = DirectionExplanation | WhichExplanation | CountExplanation


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


def explanation_to_dict(explanation: QueryExplanation) -> dict[str, Any]:
    """Return a deterministic JSON-compatible representation for generators."""
    payload = _serialize(explanation)
    if not isinstance(payload, dict):
        raise TypeError("explanation did not serialize to an object")
    return {"type": type(explanation).__name__, **payload}


def _claim_status(consistent: bool, possible: bool, entailed: bool) -> ClaimStatus:
    if not consistent:
        return ClaimStatus.INCONSISTENT
    if entailed:
        return ClaimStatus.ENTAILED
    if possible:
        return ClaimStatus.CONTINGENT
    return ClaimStatus.IMPOSSIBLE


@dataclass(frozen=True)
class _Edge:
    destination: str
    strict: bool
    premise_index: int


def _axis_adjacency(
    problem: SpatialProblem,
    axis: Axis,
) -> dict[str, list[_Edge]] | None:
    atoms = conjunctive_atoms(problem.premise)
    if atoms is None or any(len(atom.allowed) != 1 for atom in atoms):
        return None

    axis_index = 0 if axis is Axis.X else 1
    adjacency = {obj: [] for obj in problem.objects}
    for premise_index, atom in enumerate(atoms):
        direction = next(iter(atom.allowed))
        sign = direction_signs(direction)[axis_index]
        if sign == 0:
            adjacency[atom.subject].append(_Edge(atom.reference, False, premise_index))
            adjacency[atom.reference].append(_Edge(atom.subject, False, premise_index))
        elif sign > 0:
            adjacency[atom.reference].append(_Edge(atom.subject, True, premise_index))
        else:
            adjacency[atom.subject].append(_Edge(atom.reference, True, premise_index))
    for edges in adjacency.values():
        edges.sort(key=lambda edge: (edge.destination, edge.premise_index))
    return adjacency


def _axis_path(
    problem: SpatialProblem,
    axis: Axis,
    subject: str,
    reference: str,
    sign: int,
) -> AxisDerivation | None:
    adjacency = _axis_adjacency(problem, axis)
    if adjacency is None:
        return None

    if sign > 0:
        start, end, require_strict = reference, subject, True
        relation = AxisRelation.GREATER
    elif sign < 0:
        start, end, require_strict = subject, reference, True
        relation = AxisRelation.LESS
    else:
        start, end, require_strict = subject, reference, False
        relation = AxisRelation.EQUAL

    initial = (start, False)
    frontier = deque([initial])
    parents: dict[tuple[str, bool], tuple[tuple[str, bool], int]] = {}
    visited = {initial}
    final: tuple[str, bool] | None = None
    while frontier:
        state = frontier.popleft()
        node, has_strict = state
        if node == end and (has_strict or not require_strict):
            final = state
            break
        for edge in adjacency[node]:
            if not require_strict and edge.strict:
                continue
            next_state = (edge.destination, has_strict or edge.strict)
            if next_state in visited:
                continue
            visited.add(next_state)
            parents[next_state] = (state, edge.premise_index)
            frontier.append(next_state)
    if final is None:
        return None

    nodes = [final[0]]
    premise_indices: list[int] = []
    cursor = final
    while cursor != initial:
        previous, premise_index = parents[cursor]
        premise_indices.append(premise_index)
        nodes.append(previous[0])
        cursor = previous
    nodes.reverse()
    premise_indices.reverse()
    if sign > 0:
        nodes.reverse()
        premise_indices.reverse()
    return AxisDerivation(
        axis,
        subject,
        relation,
        reference,
        tuple(nodes),
        tuple(premise_indices),
    )


def _axis_proof(
    problem: SpatialProblem,
    subject: str,
    reference: str,
    direction: Direction,
) -> AxisProof | None:
    x_sign, y_sign = direction_signs(direction)
    x = _axis_path(problem, Axis.X, subject, reference, x_sign)
    y = _axis_path(problem, Axis.Y, subject, reference, y_sign)
    if x is None or y is None:
        return None
    return AxisProof(x, y, direction)


class SpatialExplainerV2:
    """Build typed explanations from structured problems and solver results."""

    def __init__(self, solver: SpatialSolverV2 | None = None) -> None:
        self._solver = solver or SpatialSolverV2()

    def explain(
        self,
        problem: SpatialProblem,
        analysis: QueryAnalysis | None = None,
    ) -> QueryExplanation:
        analysis = analysis or self._solver.analyze(problem)
        if isinstance(problem.query, DirectionQuery) and isinstance(
            analysis, DirectionAnalysis
        ):
            return self._explain_direction(problem, analysis)
        if isinstance(problem.query, WhichQuery) and isinstance(
            analysis, WhichAnalysis
        ):
            return self._explain_which(problem, analysis)
        if isinstance(problem.query, CountQuery) and isinstance(
            analysis, CountAnalysis
        ):
            return self._explain_count(problem, analysis)
        raise TypeError("analysis type does not match the problem query")

    def _evidence(
        self,
        problem: SpatialProblem,
        claim: RelationConstraint,
        axis_proof: AxisProof | None = None,
    ) -> ClaimEvidence:
        assessed = self._solver.assess(problem, claim)
        if assessed.error:
            raise RuntimeError(assessed.error)
        status = _claim_status(
            assessed.consistent,
            assessed.possible,
            assessed.entailed,
        )
        return ClaimEvidence(
            status=status,
            witness=assessed.witness,
            counterexample=assessed.counterexample,
            axis_proof=axis_proof if status is ClaimStatus.ENTAILED else None,
            negation_unsatisfiable=status is ClaimStatus.ENTAILED,
        )

    def _explain_direction(
        self,
        problem: SpatialProblem,
        analysis: DirectionAnalysis,
    ) -> DirectionExplanation:
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
                DirectionCase(
                    direction,
                    self._evidence(
                        problem,
                        claim,
                        _axis_proof(
                            problem,
                            query.target,
                            query.reference,
                            direction,
                        ),
                    ),
                )
            )
        return DirectionExplanation(
            query.target,
            query.reference,
            tuple(cases),
            analysis.engine,
        )

    def _membership_explanations(
        self,
        problem: SpatialProblem,
        query: WhichQuery | CountQuery,
    ) -> tuple[MembershipExplanation, ...]:
        return tuple(
            MembershipExplanation(
                candidate,
                self._evidence(problem, membership_constraint(candidate, query)),
            )
            for candidate in query.candidates
        )

    def _explain_which(
        self,
        problem: SpatialProblem,
        analysis: WhichAnalysis,
    ) -> WhichExplanation:
        query = problem.query
        assert isinstance(query, WhichQuery)
        return WhichExplanation(
            query.reference,
            query.directions,
            self._membership_explanations(problem, query),
            analysis.engine,
        )

    def _explain_count(
        self,
        problem: SpatialProblem,
        analysis: CountAnalysis,
    ) -> CountExplanation:
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
            cases.append(CountCase(count, status, analysis.witnesses.get(count)))
        return CountExplanation(
            query.reference,
            query.directions,
            self._membership_explanations(problem, query),
            tuple(cases),
            analysis.engine,
        )


def _label(value: str, labels: Mapping[str, str]) -> str:
    return labels.get(value, value)


def _render_directions(directions: frozenset[Direction]) -> str:
    return "/".join(
        direction.value for direction in Direction if direction in directions
    )


def _render_coordinates(
    coordinates: Coordinates,
    labels: Mapping[str, str],
) -> str:
    return ", ".join(
        f"{_label(obj, labels)}=({x}, {y})" for obj, (x, y) in coordinates.items()
    )


def _render_axis(derivation: AxisDerivation, labels: Mapping[str, str]) -> str:
    path = " -> ".join(_label(obj, labels) for obj in derivation.path)
    premises = ", ".join(str(index + 1) for index in derivation.premise_indices)
    subject = _label(derivation.subject, labels)
    reference = _label(derivation.reference, labels)
    return (
        f"On the {derivation.axis.value}-axis, premises {premises} give path "
        f"{path}; therefore {subject}.{derivation.axis.value} "
        f"{derivation.relation.value} {reference}.{derivation.axis.value}."
    )


def _render_evidence(
    subject: str,
    relation: str,
    reference: str,
    evidence: ClaimEvidence,
    labels: Mapping[str, str],
) -> list[str]:
    claim = f"{_label(subject, labels)} is {relation} of {_label(reference, labels)}"
    lines = [f"{claim}: {evidence.status.value}."]
    if evidence.axis_proof is not None:
        lines.extend(
            (
                _render_axis(evidence.axis_proof.x, labels),
                _render_axis(evidence.axis_proof.y, labels),
                f"Combining both axes gives {evidence.axis_proof.direction.value}.",
            )
        )
    elif evidence.negation_unsatisfiable:
        lines.append("Its negation is inconsistent with the premises.")
    if evidence.witness is not None and evidence.status is ClaimStatus.CONTINGENT:
        lines.append(
            f"Satisfying witness: {_render_coordinates(evidence.witness, labels)}."
        )
    if (
        evidence.counterexample is not None
        and evidence.status is ClaimStatus.CONTINGENT
    ):
        lines.append(
            f"Counterexample witness: {_render_coordinates(evidence.counterexample, labels)}."
        )
    return lines


def render_explanation(
    explanation: QueryExplanation,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render a structured explanation without assuming a dataset format."""
    labels = labels or {}
    lines: list[str] = []
    if isinstance(explanation, DirectionExplanation):
        for case in explanation.cases:
            if case.evidence.status is ClaimStatus.IMPOSSIBLE:
                continue
            lines.extend(
                _render_evidence(
                    explanation.target,
                    case.direction.value,
                    explanation.reference,
                    case.evidence,
                    labels,
                )
            )
        impossible = [
            case.direction.value
            for case in explanation.cases
            if case.evidence.status is ClaimStatus.IMPOSSIBLE
        ]
        if impossible:
            lines.append("Impossible alternatives: " + ", ".join(impossible) + ".")
    elif isinstance(explanation, WhichExplanation):
        relation = _render_directions(explanation.directions)
        for membership in explanation.memberships:
            lines.extend(
                _render_evidence(
                    membership.candidate,
                    relation,
                    explanation.reference,
                    membership.evidence,
                    labels,
                )
            )
    else:
        possible = explanation.possible_counts
        lines.append("Possible counts: " + ", ".join(map(str, possible)) + ".")
        for case in explanation.counts:
            if case.witness is not None:
                lines.append(
                    f"Count {case.count} witness: "
                    f"{_render_coordinates(case.witness, labels)}."
                )
        relation = _render_directions(explanation.directions)
        for membership in explanation.memberships:
            lines.extend(
                _render_evidence(
                    membership.candidate,
                    relation,
                    explanation.reference,
                    membership.evidence,
                    labels,
                )
            )
    return "\n".join(lines)
