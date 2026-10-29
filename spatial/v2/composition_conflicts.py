"""Small replayable axis-path conflicts for atomic composition premises.

This is evidence construction, not a replacement solver. Z3 must independently
confirm the cited subset is inconsistent. Equality paths also detect forced
coincidence, which strict-cycle detection alone cannot establish.
"""

from collections import deque
from dataclasses import dataclass, replace
from itertools import combinations, pairwise

from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    analyze,
    validate_problem,
)
from spatial.v2.solver import And, SpatialProblem, SpatialSolverV2, direction_signs


@dataclass(frozen=True)
class Edge:
    premise_index: int
    axis: int
    start: str
    end: str
    strict: bool


@dataclass(frozen=True)
class Conflict:
    problem: SpatialProblem
    kind: str
    paths: tuple[tuple[Edge, ...], ...]

    @property
    def premise_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted({edge.premise_index for path in self.paths for edge in path})
        )


def _edges(problem: SpatialProblem, axis: int) -> tuple[Edge, ...]:
    edges = []
    for i, atom in enumerate(problem.premise.operands):
        sign = direction_signs(next(iter(atom.allowed)))[axis]
        a, b = atom.subject, atom.reference
        if sign == 0:
            edges.extend((Edge(i, axis, a, b, False), Edge(i, axis, b, a, False)))
        else:
            start, end = (a, b) if sign < 0 else (b, a)
            edges.append(Edge(i, axis, start, end, True))
    return tuple(edges)


def _path(edges: tuple[Edge, ...], start: str, end: str) -> tuple[Edge, ...] | None:
    queue = deque([(start, ())])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for edge in edges:
            if edge.start == node and edge.end not in visited:
                visited.add(edge.end)
                queue.append((edge.end, (*path, edge)))
    return None


def build_conflict(problem: SpatialProblem) -> Conflict:
    validate_problem(problem)
    axes = tuple(_edges(problem, axis) for axis in (0, 1))
    for edges in axes:
        for edge in edges:
            if not edge.strict:
                continue
            back = _path(edges, edge.end, edge.start)
            if back is not None:
                result = Conflict(problem, "strict-cycle", ((edge, *back),))
                check_conflict(result)
                return result
    for a, b in combinations(problem.objects, 2):
        paths = tuple(
            _path(tuple(edge for edge in edges if not edge.strict), a, b)
            for edges in axes
        )
        if all(path is not None for path in paths):
            result = Conflict(problem, "forced-coincidence", paths)
            check_conflict(result)
            return result
    raise UnsupportedEvidence(
        "no supported strict-cycle or forced-coincidence conflict"
    )


def check_conflict(conflict: Conflict) -> None:
    """Replay paths without a solver; no minimal-conflict claim is made."""
    validate_problem(conflict.problem)
    allowed = {edge for axis in (0, 1) for edge in _edges(conflict.problem, axis)}
    for path in conflict.paths:
        if not path or any(edge not in allowed for edge in path):
            raise VerificationError("conflict cites an invalid premise edge")
        if len({edge.axis for edge in path}) != 1:
            raise VerificationError("conflict path mixes axes")
        if any(a.end != b.start for a, b in pairwise(path)):
            raise VerificationError("conflict path is disconnected")
    if conflict.kind == "strict-cycle":
        if len(conflict.paths) != 1:
            raise VerificationError("strict cycle needs one path")
        path = conflict.paths[0]
        if path[0].start != path[-1].end or not any(edge.strict for edge in path):
            raise VerificationError("cycle must close and contain a strict edge")
    elif conflict.kind == "forced-coincidence":
        if len(conflict.paths) != 2:
            raise VerificationError("coincidence needs both axes")
        x, y = conflict.paths
        if (
            x[0].axis != 0
            or y[0].axis != 1
            or x[0].start == x[-1].end
            or (x[0].start, x[-1].end) != (y[0].start, y[-1].end)
            or any(edge.strict for path in conflict.paths for edge in path)
        ):
            raise VerificationError(
                "coincidence requires equality of the same distinct pair"
            )
    else:
        raise VerificationError("unknown conflict kind")


def verify_conflict(conflict: Conflict, solver: SpatialSolverV2) -> None:
    check_conflict(conflict)
    subset = replace(
        conflict.problem,
        premise=And(
            tuple(
                conflict.problem.premise.operands[i] for i in conflict.premise_indices
            )
        ),
    )
    if analyze(subset, solver=solver).consistent:
        raise VerificationError("cited conflict subset is satisfiable")


def render_conflict(conflict: Conflict, *, symbolic: bool) -> str:
    check_conflict(conflict)
    lines = []
    for path in conflict.paths:
        for edge in path:
            axis = "xy"[edge.axis]
            if symbolic:
                lines.append(
                    f"p{edge.premise_index + 1}: {axis}({edge.start}) "
                    f"{'<' if edge.strict else '='} {axis}({edge.end})"
                )
            else:
                relation = "less than" if edge.strict else "equal to"
                lines.append(
                    f"From p{edge.premise_index + 1}, {edge.start}'s {axis}-coordinate "
                    f"is {relation} {edge.end}'s {axis}-coordinate."
                )
        start, end, axis = path[0].start, path[-1].end, "xy"[path[0].axis]
        strict = any(edge.strict for edge in path)
        if symbolic:
            lines.append(
                f"Transitivity: {axis}({start}) {'<' if strict else '='} {axis}({end})"
            )
        else:
            lines.append(
                f"Composing this path gives {start}'s {axis}-coordinate "
                f"{'less than' if strict else 'equal to'} {end}'s {axis}-coordinate."
            )
    if conflict.kind == "strict-cycle":
        lines.append("A strict order cannot be less than itself: contradiction.")
    else:
        a, b = conflict.paths[0][0].start, conflict.paths[0][-1].end
        lines.append(
            f"{a} and {b} would share both coordinates, violating the distinct-point axiom."
        )
    ids = ", ".join(f"p{i + 1}" for i in conflict.premise_indices)
    lines.append(f"Conflicting premises: {ids}. No minimality claim is made.")
    return "\n".join(lines)
