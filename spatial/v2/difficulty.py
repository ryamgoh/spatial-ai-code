"""Structural proof-difficulty measurements for generated spatial problems."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionAnalysis,
    DirectionQuery,
    Not,
    Or,
    QueryAnalysis,
    SpatialProblem,
    SpatialSolverV2,
    WhichAnalysis,
    WhichQuery,
    conjunctive_atoms,
    direction_constraint,
    direction_signs,
    membership_constraint,
)


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


def derive_axis_proof(
    problem: SpatialProblem,
    subject: str,
    reference: str,
    direction: Direction,
) -> AxisProof | None:
    """Return the shortest replay-independent axis support for one direction."""
    x_sign, y_sign = direction_signs(direction)
    x = _axis_path(problem, Axis.X, subject, reference, x_sign)
    y = _axis_path(problem, Axis.Y, subject, reference, y_sign)
    if x is None or y is None:
        return None
    return AxisProof(x, y, direction)


def _opposite(direction: Direction) -> Direction:
    x_sign, y_sign = direction_signs(direction)
    return next(
        candidate
        for candidate in Direction
        if direction_signs(candidate) == (-x_sign, -y_sign)
    )


def _has_direct_membership_fact(problem: SpatialProblem) -> bool:
    query = problem.query
    if not isinstance(query, (WhichQuery, CountQuery)):
        return False
    candidates = set(query.candidates)
    atoms = conjunctive_atoms(problem.premise) or ()
    for atom in atoms:
        if atom.reference == query.reference and atom.subject in candidates:
            allowed = atom.allowed
        elif atom.subject == query.reference and atom.reference in candidates:
            allowed = frozenset(_opposite(direction) for direction in atom.allowed)
        else:
            continue
        if allowed <= query.directions:
            return True
    return False


def _possibility_count(analysis: QueryAnalysis) -> int:
    if isinstance(analysis, DirectionAnalysis):
        return len(analysis.possible_directions)
    if isinstance(analysis, WhichAnalysis):
        return len(analysis.possible_entities)
    return len(analysis.possible_counts)


def measure_difficulty(
    problem: SpatialProblem,
    analysis: QueryAnalysis,
    solver: SpatialSolverV2,
) -> dict[str, Any]:
    """Measure positive axis paths and sufficient complete-answer support.

    Distractors are jointly removable while preserving possible answers and
    candidate membership statuses. Support protects the reported axis paths;
    it is sufficient rather than globally minimal and may depend on input order.
    Boolean depths are not inferred from these positive atomic path metrics.
    """
    query = problem.query
    direct_query_relation = False
    x_depth: int | None = None
    y_depth: int | None = None
    x_support: set[int] = set()
    y_support: set[int] = set()
    membership_proofs: list[dict[str, Any]] = []

    if isinstance(query, DirectionQuery):
        query_pair = {query.target, query.reference}
        atoms = conjunctive_atoms(problem.premise) or ()
        direct_query_relation = any(
            {atom.subject, atom.reference} == query_pair for atom in atoms
        )
        for direction in analysis.possible_directions:
            claim = direction_constraint(query.target, query.reference, direction)
            assessment = solver.assess(problem, claim)
            if not assessment.entailed:
                continue
            proof = derive_axis_proof(
                problem,
                query.target,
                query.reference,
                direction,
            )
            if proof is not None:
                x_support.update(proof.x.premise_indices)
                y_support.update(proof.y.premise_indices)
                x_depth = len(proof.x.premise_indices)
                y_depth = len(proof.y.premise_indices)
            break
    else:
        assert isinstance(query, (WhichQuery, CountQuery))
        direct_query_relation = _has_direct_membership_fact(problem)
        if len(query.directions) == 1:
            direction = next(iter(query.directions))
            for candidate in query.candidates:
                claim = membership_constraint(candidate, query)
                assessment = solver.assess(problem, claim)
                if not assessment.entailed:
                    continue
                proof = derive_axis_proof(
                    problem,
                    candidate,
                    query.reference,
                    direction,
                )
                if proof is None:
                    continue
                membership_x = set(proof.x.premise_indices)
                membership_y = set(proof.y.premise_indices)
                x_support.update(membership_x)
                y_support.update(membership_y)
                membership_proofs.append(
                    {
                        "candidate": candidate,
                        "x_depth": len(membership_x),
                        "y_depth": len(membership_y),
                        "axes_independent": bool(
                            membership_x
                            and membership_y
                            and membership_x.isdisjoint(membership_y)
                        ),
                        "premise_indices": sorted(membership_x | membership_y),
                    }
                )

    positive_support = x_support | y_support
    units = conjunctive_atoms(problem.premise)
    if units is None:
        units = (
            problem.premise.operands
            if isinstance(problem.premise, And)
            else (problem.premise,)
        )
    support = set(range(len(units)))
    try:
        expected = solver.semantic_signature(problem)
    except RuntimeError:
        expected = None
    support_checks_complete = expected is not None
    for index in range(len(units)):
        if expected is None:
            break
        if index in positive_support:
            continue
        retained = support - {index}
        # The formula schema forbids an empty And; represent no constraints by
        # excluded middle without adding any spatial assumptions.
        premise = (
            And(tuple(units[item] for item in sorted(retained)))
            if retained
            else Or((units[0], Not(units[0])))
        )
        reduced = SpatialProblem(problem.objects, premise, problem.query)
        try:
            unchanged = solver.semantic_signature(reduced) == expected
        except RuntimeError:
            # A timeout cannot establish removability. Keep this premise.
            support_checks_complete = False
            continue
        if unchanged:
            support = retained
    membership_x_depths = [proof["x_depth"] for proof in membership_proofs]
    membership_y_depths = [proof["y_depth"] for proof in membership_proofs]
    premise_count = len(units)
    return {
        "possibility_count": _possibility_count(analysis),
        "direct_query_relation": direct_query_relation,
        "x_depth": x_depth,
        "y_depth": y_depth,
        "axes_independent": bool(
            x_support and y_support and x_support.isdisjoint(y_support)
        ),
        "membership_proofs": membership_proofs,
        "min_membership_x_depth": min(membership_x_depths, default=None),
        "max_membership_x_depth": max(membership_x_depths, default=None),
        "min_membership_y_depth": min(membership_y_depths, default=None),
        "max_membership_y_depth": max(membership_y_depths, default=None),
        "supporting_premise_indices": sorted(support),
        "positive_axis_premise_indices": sorted(positive_support),
        "support_semantics": "jointly-sufficient-answer-and-membership",
        "support_checks_complete": support_checks_complete,
        "premise_indexing": "flattened-atoms"
        if conjunctive_atoms(problem.premise) is not None
        else "top-level-formulas",
        "num_distractor_premises": premise_count - len(support),
    }
