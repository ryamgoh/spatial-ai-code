"""Replayable proof certificates for SpatialEntail V2 problems.

This module is deliberately independent of the SMT implementation.  It builds
and checks typed derivations from the structured premises themselves.  The
initial proof language covers exact positive-conjunction Direction problems;
Boolean branching, ambiguous certificates, Which, and Count are explicit
future extensions rather than solver-status fallbacks.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from spatial_solver_v2 import (
    Direction,
    DirectionQuery,
    RelationConstraint,
    SpatialProblem,
    conjunctive_atoms,
    direction_signs,
)


class ProofConstructionError(ValueError):
    """The requested problem is outside the supported proof-first fragment."""


class ProofCheckError(ValueError):
    """A proof certificate contains an invalid or unsupported inference."""


class ProofRule(str, Enum):
    PREMISE = "premise"
    DIRECTION_DECOMPOSITION = "direction-decomposition"
    AXIS_INVERSION = "axis-inversion"
    AXIS_TRANSITIVITY = "axis-transitivity"
    DIRECTION_RECOMPOSITION = "direction-recomposition"


class ProofAxis(str, Enum):
    X = "x"
    Y = "y"


class OrderRelation(str, Enum):
    LESS = "<"
    EQUAL = "="
    GREATER = ">"


@dataclass(frozen=True)
class AxisFact:
    axis: ProofAxis
    subject: str
    relation: OrderRelation
    reference: str


@dataclass(frozen=True)
class DirectionClaim:
    subject: str
    direction: Direction
    reference: str


ProofConclusion = RelationConstraint | AxisFact | DirectionClaim


@dataclass(frozen=True)
class ProofStep:
    id: str
    rule: ProofRule
    conclusion: ProofConclusion
    inputs: tuple[str, ...] = ()
    premise_index: int | None = None


@dataclass(frozen=True)
class DirectionProofCertificate:
    problem: SpatialProblem
    steps: tuple[ProofStep, ...]
    conclusion_step: str

    @property
    def conclusion(self) -> DirectionClaim:
        step = next(
            (item for item in self.steps if item.id == self.conclusion_step),
            None,
        )
        if step is None or not isinstance(step.conclusion, DirectionClaim):
            raise ProofCheckError("conclusion_step must identify a DirectionClaim")
        return step.conclusion

    @property
    def support_premise_indices(self) -> tuple[int, ...]:
        reachable = _reachable_step_ids(self)
        return tuple(
            sorted(
                step.premise_index
                for step in self.steps
                if step.id in reachable and step.premise_index is not None
            )
        )


def _relation_from_sign(sign: int) -> OrderRelation:
    return {
        -1: OrderRelation.LESS,
        0: OrderRelation.EQUAL,
        1: OrderRelation.GREATER,
    }[sign]


def _relation_sign(relation: OrderRelation) -> int:
    return {
        OrderRelation.LESS: -1,
        OrderRelation.EQUAL: 0,
        OrderRelation.GREATER: 1,
    }[relation]


def _inverse_relation(relation: OrderRelation) -> OrderRelation:
    return _relation_from_sign(-_relation_sign(relation))


def _axis_fact(atom: RelationConstraint, axis: ProofAxis) -> AxisFact:
    if len(atom.allowed) != 1:
        raise ProofConstructionError("proof premises must use exact directions")
    direction = next(iter(atom.allowed))
    axis_index = 0 if axis is ProofAxis.X else 1
    return AxisFact(
        axis,
        atom.subject,
        _relation_from_sign(direction_signs(direction)[axis_index]),
        atom.reference,
    )


def _inverse_fact(fact: AxisFact) -> AxisFact:
    return AxisFact(
        fact.axis,
        fact.reference,
        _inverse_relation(fact.relation),
        fact.subject,
    )


def _compose_relations(
    first: OrderRelation,
    second: OrderRelation,
) -> OrderRelation | None:
    first_sign = _relation_sign(first)
    second_sign = _relation_sign(second)
    if first_sign == 0:
        return second
    if second_sign == 0 or first_sign == second_sign:
        return first
    return None


def _compose_facts(first: AxisFact, second: AxisFact) -> AxisFact | None:
    if first.axis is not second.axis or first.reference != second.subject:
        return None
    relation = _compose_relations(first.relation, second.relation)
    if relation is None:
        return None
    return AxisFact(first.axis, first.subject, relation, second.reference)


def _reachable_step_ids(certificate: DirectionProofCertificate) -> frozenset[str]:
    steps = {step.id: step for step in certificate.steps}
    reachable: set[str] = set()
    frontier = [certificate.conclusion_step]
    while frontier:
        step_id = frontier.pop()
        if step_id in reachable:
            continue
        step = steps.get(step_id)
        if step is None:
            raise ProofCheckError(f"proof references unknown step: {step_id}")
        reachable.add(step_id)
        frontier.extend(step.inputs)
    return frozenset(reachable)


def _check_step(
    step: ProofStep,
    previous: Mapping[str, ProofStep],
    atoms: tuple[RelationConstraint, ...],
) -> None:
    inputs = tuple(previous.get(step_id) for step_id in step.inputs)
    if any(item is None for item in inputs):
        raise ProofCheckError(f"{step.id} depends on an unknown or later step")
    resolved_inputs = tuple(item for item in inputs if item is not None)

    if step.rule is ProofRule.PREMISE:
        if step.inputs or step.premise_index is None:
            raise ProofCheckError(f"{step.id} is not a valid premise step")
        if not 0 <= step.premise_index < len(atoms):
            raise ProofCheckError(f"{step.id} has an invalid premise index")
        if step.conclusion != atoms[step.premise_index]:
            raise ProofCheckError(f"{step.id} does not match its indexed premise")
        return

    if step.premise_index is not None:
        raise ProofCheckError(f"{step.id} assigns a premise index to a derived step")

    if step.rule is ProofRule.DIRECTION_DECOMPOSITION:
        if len(resolved_inputs) != 1 or not isinstance(
            resolved_inputs[0].conclusion, RelationConstraint
        ):
            raise ProofCheckError(f"{step.id} must decompose one spatial premise")
        if not isinstance(step.conclusion, AxisFact):
            raise ProofCheckError(f"{step.id} must conclude an axis fact")
        expected = _axis_fact(resolved_inputs[0].conclusion, step.conclusion.axis)
        if step.conclusion != expected:
            raise ProofCheckError(f"{step.id} has an invalid direction decomposition")
        return

    if step.rule is ProofRule.AXIS_INVERSION:
        if len(resolved_inputs) != 1 or not isinstance(
            resolved_inputs[0].conclusion, AxisFact
        ):
            raise ProofCheckError(f"{step.id} must invert one axis fact")
        if step.conclusion != _inverse_fact(resolved_inputs[0].conclusion):
            raise ProofCheckError(f"{step.id} has an invalid axis inversion")
        return

    if step.rule is ProofRule.AXIS_TRANSITIVITY:
        if len(resolved_inputs) != 2 or not all(
            isinstance(item.conclusion, AxisFact) for item in resolved_inputs
        ):
            raise ProofCheckError(f"{step.id} must compose two axis facts")
        expected = _compose_facts(
            resolved_inputs[0].conclusion,
            resolved_inputs[1].conclusion,
        )
        if expected is None or step.conclusion != expected:
            raise ProofCheckError(f"{step.id} has an invalid transitivity step")
        return

    if step.rule is ProofRule.DIRECTION_RECOMPOSITION:
        if len(resolved_inputs) != 2 or not all(
            isinstance(item.conclusion, AxisFact) for item in resolved_inputs
        ):
            raise ProofCheckError(f"{step.id} must combine two axis facts")
        if not isinstance(step.conclusion, DirectionClaim):
            raise ProofCheckError(f"{step.id} must conclude a direction claim")
        facts = {item.conclusion.axis: item.conclusion for item in resolved_inputs}
        if set(facts) != {ProofAxis.X, ProofAxis.Y}:
            raise ProofCheckError(f"{step.id} requires one X and one Y fact")
        x_fact = facts[ProofAxis.X]
        y_fact = facts[ProofAxis.Y]
        claim = step.conclusion
        actual_signs = []
        for fact in (x_fact, y_fact):
            sign = _relation_sign(fact.relation)
            if fact.subject == claim.subject and fact.reference == claim.reference:
                actual_signs.append(sign)
            elif fact.subject == claim.reference and fact.reference == claim.subject:
                actual_signs.append(-sign)
            else:
                raise ProofCheckError(f"{step.id} combines facts about different pairs")
        expected_signs = direction_signs(claim.direction)
        if tuple(actual_signs) != expected_signs:
            raise ProofCheckError(f"{step.id} recomposes the wrong direction")
        return

    raise ProofCheckError(f"unsupported proof rule: {step.rule}")


def check_direction_proof(certificate: DirectionProofCertificate) -> None:
    """Replay a Direction certificate without consulting an SMT solver."""
    problem = certificate.problem
    query = problem.query
    if not isinstance(query, DirectionQuery):
        raise ProofCheckError("Direction certificates require a DirectionQuery")
    atoms = conjunctive_atoms(problem.premise)
    if atoms is None or any(len(atom.allowed) != 1 for atom in atoms):
        raise ProofCheckError(
            "Direction certificates currently require exact positive conjunctions"
        )

    previous: dict[str, ProofStep] = {}
    for step in certificate.steps:
        if not step.id or step.id in previous:
            raise ProofCheckError("proof step identifiers must be non-empty and unique")
        _check_step(step, previous, atoms)
        previous[step.id] = step

    if certificate.conclusion_step not in previous:
        raise ProofCheckError("conclusion_step does not identify a proof step")
    conclusion = certificate.conclusion
    if (
        conclusion.subject != query.target
        or conclusion.reference != query.reference
        or conclusion.direction not in query.candidate_directions
    ):
        raise ProofCheckError("proof conclusion does not answer the DirectionQuery")
    _reachable_step_ids(certificate)


@dataclass(frozen=True)
class _GraphEdge:
    destination: str
    strict: bool
    step_id: str
    invert: bool


_AxisPath = tuple[tuple[str, bool], ...]


def _proof_graph(
    objects: tuple[str, ...],
    steps: tuple[ProofStep, ...],
    axis: ProofAxis,
) -> dict[str, list[_GraphEdge]]:
    adjacency = {obj: [] for obj in objects}
    for step in steps:
        fact = step.conclusion
        if not isinstance(fact, AxisFact) or fact.axis is not axis:
            continue
        if fact.relation is OrderRelation.LESS:
            adjacency[fact.subject].append(
                _GraphEdge(fact.reference, True, step.id, False)
            )
        elif fact.relation is OrderRelation.GREATER:
            adjacency[fact.reference].append(
                _GraphEdge(fact.subject, True, step.id, True)
            )
        elif fact.relation is OrderRelation.EQUAL:
            adjacency[fact.subject].append(
                _GraphEdge(fact.reference, False, step.id, False)
            )
            adjacency[fact.reference].append(
                _GraphEdge(fact.subject, False, step.id, True)
            )
    for edges in adjacency.values():
        edges.sort(key=lambda item: (item.destination, item.step_id))
    return adjacency


def _find_axis_path(
    adjacency: Mapping[str, list[_GraphEdge]],
    subject: str,
    reference: str,
    sign: int,
) -> _AxisPath | None:
    if sign > 0:
        start, end, require_strict = reference, subject, True
    elif sign < 0:
        start, end, require_strict = subject, reference, True
    else:
        start, end, require_strict = subject, reference, False

    initial = (start, False)
    frontier = deque([initial])
    visited = {initial}
    parents: dict[tuple[str, bool], tuple[tuple[str, bool], _GraphEdge]] = {}
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
            parents[next_state] = (state, edge)
            frontier.append(next_state)
    if final is None:
        return None

    edge_steps: list[tuple[str, bool]] = []
    cursor = final
    while cursor != initial:
        previous, edge = parents[cursor]
        edge_steps.append((edge.step_id, edge.invert))
        cursor = previous
    edge_steps.reverse()
    return tuple(edge_steps)


def _derive_axis_path(
    axis: ProofAxis,
    edge_steps: _AxisPath,
    steps: list[ProofStep],
) -> str:
    if not edge_steps:
        raise ProofConstructionError("axis proof path cannot be empty")
    by_id = {step.id: step for step in steps}
    if len(edge_steps) == 1:
        return edge_steps[0][0]

    oriented_steps = []
    for index, (step_id, invert) in enumerate(edge_steps, start=1):
        if not invert:
            oriented_steps.append(step_id)
            continue
        fact = by_id[step_id].conclusion
        assert isinstance(fact, AxisFact)
        inverse_id = f"PATH-{axis.value.upper()}-{index}-INV"
        inverse = ProofStep(
            inverse_id,
            ProofRule.AXIS_INVERSION,
            _inverse_fact(fact),
            (step_id,),
        )
        steps.append(inverse)
        by_id[inverse_id] = inverse
        oriented_steps.append(inverse_id)

    current_id = oriented_steps[0]
    for index, next_id in enumerate(oriented_steps[1:], start=1):
        first = by_id[current_id].conclusion
        second = by_id[next_id].conclusion
        assert isinstance(first, AxisFact) and isinstance(second, AxisFact)
        conclusion = _compose_facts(first, second)
        if conclusion is None:
            raise ProofConstructionError("axis path is not transitively composable")
        current_id = f"D-{axis.value.upper()}-{index}"
        derived = ProofStep(
            current_id,
            ProofRule.AXIS_TRANSITIVITY,
            conclusion,
            (
                oriented_steps[0]
                if index == 1
                else f"D-{axis.value.upper()}-{index - 1}",
                next_id,
            ),
        )
        steps.append(derived)
        by_id[current_id] = derived
    return current_id


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


def proof_to_dict(certificate: DirectionProofCertificate) -> dict[str, Any]:
    """Return a deterministic JSON-compatible proof representation."""
    check_direction_proof(certificate)
    payload = _serialize(certificate)
    if not isinstance(payload, dict):
        raise TypeError("proof certificate did not serialize to an object")
    return {"type": type(certificate).__name__, **payload}


def build_direction_proof(
    problem: SpatialProblem,
    direction: Direction | None = None,
) -> DirectionProofCertificate:
    """Construct and replay a proof for one entailed exact Direction answer."""
    query = problem.query
    if not isinstance(query, DirectionQuery):
        raise ProofConstructionError(
            "proof construction currently supports DirectionQuery"
        )
    atoms = conjunctive_atoms(problem.premise)
    if atoms is None or any(len(atom.allowed) != 1 for atom in atoms):
        raise ProofConstructionError(
            "proof construction currently requires exact positive conjunctions"
        )

    steps: list[ProofStep] = []
    for premise_index, atom in enumerate(atoms):
        premise_id = f"P{premise_index + 1}"
        steps.append(
            ProofStep(
                premise_id,
                ProofRule.PREMISE,
                atom,
                premise_index=premise_index,
            )
        )
        for axis in ProofAxis:
            direct = _axis_fact(atom, axis)
            direct_id = f"{premise_id}-{axis.value.upper()}"
            steps.append(
                ProofStep(
                    direct_id,
                    ProofRule.DIRECTION_DECOMPOSITION,
                    direct,
                    (premise_id,),
                )
            )

    graphs = {
        axis: _proof_graph(problem.objects, tuple(steps), axis) for axis in ProofAxis
    }

    candidates = [direction] if direction is not None else list(Direction)
    candidates = [item for item in candidates if item in query.candidate_directions]
    paths_by_direction: dict[
        Direction,
        dict[ProofAxis, _AxisPath],
    ] = {}
    for candidate in candidates:
        x_sign, y_sign = direction_signs(candidate)
        paths = {
            ProofAxis.X: _find_axis_path(
                graphs[ProofAxis.X], query.target, query.reference, x_sign
            ),
            ProofAxis.Y: _find_axis_path(
                graphs[ProofAxis.Y], query.target, query.reference, y_sign
            ),
        }
        if all(path is not None for path in paths.values()):
            paths_by_direction[candidate] = {
                axis: path for axis, path in paths.items() if path is not None
            }

    if direction is not None and direction not in paths_by_direction:
        raise ProofConstructionError(
            f"premises do not derive {direction.value} for the query pair"
        )
    if direction is None and len(paths_by_direction) != 1:
        names = ", ".join(item.value for item in paths_by_direction) or "none"
        raise ProofConstructionError(
            "proof construction requires one entailed direction; "
            f"derived candidates: {names}"
        )
    conclusion_direction = direction or next(iter(paths_by_direction))
    paths = paths_by_direction[conclusion_direction]
    x_step = _derive_axis_path(ProofAxis.X, paths[ProofAxis.X], steps)
    y_step = _derive_axis_path(ProofAxis.Y, paths[ProofAxis.Y], steps)
    conclusion_id = "Q-DIR"
    steps.append(
        ProofStep(
            conclusion_id,
            ProofRule.DIRECTION_RECOMPOSITION,
            DirectionClaim(
                query.target,
                conclusion_direction,
                query.reference,
            ),
            (x_step, y_step),
        )
    )
    candidate = DirectionProofCertificate(problem, tuple(steps), conclusion_id)
    reachable = _reachable_step_ids(candidate)
    certificate = DirectionProofCertificate(
        problem,
        tuple(step for step in steps if step.id in reachable),
        conclusion_id,
    )
    check_direction_proof(certificate)
    return certificate
