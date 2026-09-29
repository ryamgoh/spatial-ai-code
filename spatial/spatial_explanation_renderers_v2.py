"""Coordinate-free natural-language and symbolic training traces."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product

from spatial_explanations_v2 import (
    ClaimStatus,
    CountExplanation,
    DirectionExplanation,
    MembershipExplanation,
    QueryExplanation,
    WhichExplanation,
)
from spatial_solver_v2 import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    Iff,
    Implies,
    Not,
    Or,
    RelationConstraint,
    SpatialFormula,
    SpatialProblem,
    WhichQuery,
    conjunctive_atoms,
    direction_signs,
)


class TraceFormat(str, Enum):
    NATURAL = "natural"
    SYMBOLIC = "symbolic"


class StateMode(str, Enum):
    FINAL_ONLY = "final-only"
    DELTA = "delta"
    FULL = "full"


def _label(value: str, labels: Mapping[str, str]) -> str:
    return labels.get(value, value)


def _directions(directions: Collection[Direction]) -> str:
    return ", ".join(
        direction.value for direction in Direction if direction in directions
    )


def _relation_text(atom: RelationConstraint, labels: Mapping[str, str]) -> str:
    subject = _label(atom.subject, labels)
    reference = _label(atom.reference, labels)
    relation = " or ".join(
        direction.value for direction in Direction if direction in atom.allowed
    )
    return f"{subject} is {relation} of {reference}"


def _formula_text(formula: SpatialFormula, labels: Mapping[str, str]) -> str:
    if isinstance(formula, RelationConstraint):
        return _relation_text(formula, labels)
    if isinstance(formula, Not):
        return f"NOT ({_formula_text(formula.operand, labels)})"
    if isinstance(formula, (And, Or)):
        operator = " AND " if isinstance(formula, And) else " OR "
        return operator.join(
            f"({_formula_text(operand, labels)})" for operand in formula.operands
        )
    if isinstance(formula, Implies):
        return (
            f"({_formula_text(formula.antecedent, labels)}) IF-THEN "
            f"({_formula_text(formula.consequent, labels)})"
        )
    if isinstance(formula, Iff):
        return (
            f"({_formula_text(formula.left, labels)}) IFF "
            f"({_formula_text(formula.right, labels)})"
        )
    raise TypeError(f"unsupported spatial formula: {type(formula).__name__}")


class _UnionFind:
    def __init__(self, values: tuple[str, ...]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


@dataclass(frozen=True, order=True)
class _AxisFact:
    lower: str
    relation: str
    higher: str


@dataclass(frozen=True)
class _AxisSnapshot:
    facts: frozenset[_AxisFact]
    paths: tuple[tuple[tuple[str, ...], ...], ...]


def _direct_axis_fact(atom: RelationConstraint, axis_index: int) -> _AxisFact:
    sign = direction_signs(next(iter(atom.allowed)))[axis_index]
    if sign < 0:
        return _AxisFact(atom.subject, "<", atom.reference)
    if sign > 0:
        return _AxisFact(atom.reference, "<", atom.subject)
    first, second = sorted((atom.subject, atom.reference))
    return _AxisFact(first, "=", second)


def _axis_snapshot(
    objects: tuple[str, ...],
    atoms: tuple[RelationConstraint, ...],
    axis_index: int,
) -> _AxisSnapshot:
    direct_facts = tuple(_direct_axis_fact(atom, axis_index) for atom in atoms)
    equality = _UnionFind(objects)
    for fact in direct_facts:
        if fact.relation == "=":
            equality.union(fact.lower, fact.higher)

    classes: dict[str, list[str]] = {}
    for obj in objects:
        classes.setdefault(equality.find(obj), []).append(obj)
    adjacency = {root: set() for root in classes}
    for fact in direct_facts:
        if fact.relation == "=":
            continue
        lower_root = equality.find(fact.lower)
        higher_root = equality.find(fact.higher)
        if lower_root != higher_root:
            adjacency[lower_root].add(higher_root)

    closure: dict[str, set[str]] = {}
    for root in classes:
        reachable: set[str] = set()
        frontier = list(adjacency[root])
        while frontier:
            following = frontier.pop()
            if following in reachable:
                continue
            reachable.add(following)
            frontier.extend(adjacency[following])
        closure[root] = reachable

    facts = {
        _AxisFact(first, "=", second)
        for members in classes.values()
        for first, second in combinations(sorted(members), 2)
    }
    facts.update(
        _AxisFact(lower, "<", higher)
        for lower_root, higher_roots in closure.items()
        for higher_root in higher_roots
        for lower, higher in product(classes[lower_root], classes[higher_root])
    )

    reduced = {
        lower: {
            higher
            for higher in following
            if not any(
                intermediate != higher and higher in closure[intermediate]
                for intermediate in following
            )
        }
        for lower, following in adjacency.items()
    }

    indegree = {root: 0 for root in classes}
    for following in reduced.values():
        for higher in following:
            indegree[higher] += 1

    paths: list[tuple[str, ...]] = []

    def visit(root: str, path: tuple[str, ...]) -> None:
        following = sorted(reduced[root])
        if not following:
            paths.append(path + (root,))
            return
        for higher in following:
            visit(higher, path + (root,))

    for root in sorted(root for root, degree in indegree.items() if degree == 0):
        visit(root, ())
    return _AxisSnapshot(
        frozenset(facts),
        tuple(tuple(tuple(sorted(classes[root])) for root in path) for path in paths),
    )


def _class_label(members: tuple[str, ...], labels: Mapping[str, str]) -> str:
    rendered = sorted(_label(obj, labels) for obj in members)
    return rendered[0] if len(rendered) == 1 else "[" + " = ".join(rendered) + "]"


def _symbolic_axis_state(
    snapshot: _AxisSnapshot,
    labels: Mapping[str, str],
) -> str:
    return "; ".join(
        " < ".join(_class_label(members, labels) for members in path)
        for path in snapshot.paths
    )


def _natural_axis_state(
    snapshot: _AxisSnapshot,
    axis_index: int,
    labels: Mapping[str, str],
) -> str:
    paths = []
    for path in snapshot.paths:
        positions = []
        for members in path:
            rendered = sorted(_label(obj, labels) for obj in members)
            positions.append(
                rendered[0]
                if len(rendered) == 1
                else " and ".join(rendered) + " at the same coordinate"
            )
        paths.append(", then ".join(positions))
    orientation = "west to east" if axis_index == 0 else "south to north"
    return f"from {orientation}: " + "; ".join(paths)


def _axis_word(axis_index: int, sign: int) -> str:
    if axis_index == 0:
        return {-1: "west", 0: "on the same X coordinate", 1: "east"}[sign]
    return {-1: "south", 0: "on the same Y coordinate", 1: "north"}[sign]


def _axis_domain(directions: tuple[Direction, ...], axis_index: int) -> tuple[int, ...]:
    return tuple(
        sign
        for sign in (-1, 0, 1)
        if any(
            direction_signs(direction)[axis_index] == sign for direction in directions
        )
    )


def _symbolic_axis_domain(
    subject: str,
    reference: str,
    signs: tuple[int, ...],
    labels: Mapping[str, str],
) -> str:
    subject = _label(subject, labels)
    reference = _label(reference, labels)
    facts = []
    for sign in signs:
        if sign < 0:
            facts.append(f"{subject} < {reference}")
        elif sign > 0:
            facts.append(f"{reference} < {subject}")
        else:
            facts.append(f"{subject} = {reference}")
    if not facts:
        return "none"
    return "{" + ", ".join(facts) + "}" if len(facts) > 1 else facts[0]


def _render_axis_fact(
    fact: _AxisFact,
    axis_index: int,
    labels: Mapping[str, str],
    trace_format: TraceFormat,
    describe_higher: bool = False,
) -> str:
    lower = _label(fact.lower, labels)
    higher = _label(fact.higher, labels)
    if trace_format is TraceFormat.SYMBOLIC:
        return f"{lower} {fact.relation} {higher}"
    if fact.relation == "=":
        axis = "X" if axis_index == 0 else "Y"
        return f"{lower} and {higher} have the same {axis} coordinate"
    if describe_higher:
        direction = "east" if axis_index == 0 else "north"
        return f"{higher} is {direction} of {lower}"
    direction = "west" if axis_index == 0 else "south"
    return f"{lower} is {direction} of {higher}"


def _delta_line(
    axis_index: int,
    facts: frozenset[_AxisFact],
    labels: Mapping[str, str],
    trace_format: TraceFormat,
) -> str:
    axis = "X" if axis_index == 0 else "Y"
    rendered = [
        _render_axis_fact(fact, axis_index, labels, trace_format)
        for fact in sorted(facts)
    ]
    if trace_format is TraceFormat.SYMBOLIC:
        return f"Delta-{axis}: {{{', '.join(rendered)}}}"
    consequences = "; ".join(rendered) if rendered else "none"
    return f"New {axis}-axis consequences: {consequences}."


def _state_lines(
    x_snapshot: _AxisSnapshot,
    y_snapshot: _AxisSnapshot,
    labels: Mapping[str, str],
    trace_format: TraceFormat,
    suffix: str = "",
) -> tuple[str, str]:
    if trace_format is TraceFormat.SYMBOLIC:
        return (
            f"X-State{suffix}: {_symbolic_axis_state(x_snapshot, labels)}",
            f"Y-State{suffix}: {_symbolic_axis_state(y_snapshot, labels)}",
        )
    return (
        f"X-axis ordering{suffix} {_natural_axis_state(x_snapshot, 0, labels)}.",
        f"Y-axis ordering{suffix} {_natural_axis_state(y_snapshot, 1, labels)}.",
    )


def _premise_lines(
    problem: SpatialProblem,
    labels: Mapping[str, str],
    trace_format: TraceFormat,
    state_mode: StateMode,
) -> list[str]:
    atoms = conjunctive_atoms(problem.premise)
    if atoms is None or any(len(atom.allowed) != 1 for atom in atoms):
        return [f"Premise formula: {_formula_text(problem.premise, labels)}."]

    lines: list[str] = []
    if state_mode is StateMode.FINAL_ONLY:
        for index, atom in enumerate(atoms, 1):
            relation = _relation_text(atom, labels)
            x_fact = _render_axis_fact(
                _direct_axis_fact(atom, 0),
                0,
                labels,
                trace_format,
                describe_higher=trace_format is TraceFormat.NATURAL,
            )
            y_fact = _render_axis_fact(
                _direct_axis_fact(atom, 1),
                1,
                labels,
                trace_format,
                describe_higher=trace_format is TraceFormat.NATURAL,
            )
            if trace_format is TraceFormat.NATURAL:
                lines.extend(
                    (
                        f"Premise {index}: {relation}.",
                        f"  X-axis decomposition: {x_fact}.",
                        f"  Y-axis decomposition: {y_fact}.",
                    )
                )
            else:
                lines.append(f"P{index}: {relation} => X[{x_fact}], Y[{y_fact}]")
        final_x = _axis_snapshot(problem.objects, atoms, 0)
        final_y = _axis_snapshot(problem.objects, atoms, 1)
        lines.extend(_state_lines(final_x, final_y, labels, trace_format))
        return lines

    lines.append(
        "Initial: X={}, Y={}"
        if trace_format is TraceFormat.SYMBOLIC
        else "Initially, no axis relations have been processed."
    )
    previous_x = _axis_snapshot(problem.objects, (), 0)
    previous_y = _axis_snapshot(problem.objects, (), 1)
    for index, atom in enumerate(atoms, 1):
        relation = _relation_text(atom, labels)
        lines.append(
            f"P{index}: {relation}"
            if trace_format is TraceFormat.SYMBOLIC
            else f"Premise {index}: {relation}."
        )
        current_atoms = atoms[:index]
        current_x = _axis_snapshot(problem.objects, current_atoms, 0)
        current_y = _axis_snapshot(problem.objects, current_atoms, 1)
        if state_mode is StateMode.DELTA:
            lines.extend(
                (
                    _delta_line(
                        0,
                        current_x.facts - previous_x.facts,
                        labels,
                        trace_format,
                    ),
                    _delta_line(
                        1,
                        current_y.facts - previous_y.facts,
                        labels,
                        trace_format,
                    ),
                )
            )
        else:
            suffix = (
                f" after P{index}"
                if trace_format is TraceFormat.SYMBOLIC
                else f" after premise {index}"
            )
            lines.extend(
                _state_lines(current_x, current_y, labels, trace_format, suffix)
            )
        previous_x, previous_y = current_x, current_y

    if state_mode is StateMode.DELTA:
        lines.extend(
            _state_lines(previous_x, previous_y, labels, trace_format, " (final)")
        )
    return lines


def _direction_lines(
    explanation: DirectionExplanation,
    labels: Mapping[str, str],
    trace_format: TraceFormat,
) -> list[str]:
    possible = explanation.possible_directions
    target = _label(explanation.target, labels)
    reference = _label(explanation.reference, labels)
    x_signs = _axis_domain(possible, 0)
    y_signs = _axis_domain(possible, 1)
    entailed = tuple(
        case.direction
        for case in explanation.cases
        if case.evidence.status is ClaimStatus.ENTAILED
    )

    if not possible:
        inconsistent = any(
            case.evidence.status is ClaimStatus.INCONSISTENT
            for case in explanation.cases
        )
        if trace_format is TraceFormat.SYMBOLIC:
            conclusion = (
                "Premises: inconsistent" if inconsistent else "Direction-Domain: {}"
            )
        else:
            conclusion = (
                "The premises are inconsistent."
                if inconsistent
                else "No candidate direction is possible."
            )
        return [f"Query: direction({target}, {reference})", conclusion]

    if trace_format is TraceFormat.SYMBOLIC:
        return [
            f"Query: direction({target}, {reference})",
            f"X-Query: {_symbolic_axis_domain(explanation.target, explanation.reference, x_signs, labels)}",
            f"Y-Query: {_symbolic_axis_domain(explanation.target, explanation.reference, y_signs, labels)}",
            f"Direction-Domain: {{{_directions(possible)}}}",
            f"Entailed: {_directions(entailed) if entailed else 'none'}",
        ]

    lines = [f"Query: determine the direction of {target} relative to {reference}."]
    for axis_index, signs in enumerate((x_signs, y_signs)):
        axis = "X" if axis_index == 0 else "Y"
        meanings = " or ".join(_axis_word(axis_index, sign) for sign in signs)
        certainty = "fixed" if len(signs) == 1 else "undetermined"
        lines.append(
            f"{axis}-axis: {target} can be {meanings} of {reference}; this axis is {certainty}."
        )
    lines.append(f"Possible directions: {_directions(possible)}.")
    lines.append(
        f"Entailed direction: {_directions(entailed)}."
        if entailed
        else "No exact direction is entailed."
    )
    return lines


def _membership_line(
    membership: MembershipExplanation,
    requested: frozenset[Direction],
    labels: Mapping[str, str],
    trace_format: TraceFormat,
) -> str:
    candidate = _label(membership.candidate, labels)
    domain = _directions(membership.possible_directions) or "none"
    requested_text = _directions(requested)
    if trace_format is TraceFormat.SYMBOLIC:
        return (
            f"{candidate}: Domain={{{domain}}}, Query={{{requested_text}}}, "
            f"Status={membership.evidence.status.value}"
        )
    return (
        f"{candidate}: possible directions are {domain}; against the requested "
        f"region {requested_text}, this candidate is {membership.evidence.status.value}."
    )


def _which_lines(
    explanation: WhichExplanation,
    labels: Mapping[str, str],
    trace_format: TraceFormat,
) -> list[str]:
    reference = _label(explanation.reference, labels)
    requested = _directions(explanation.directions)
    heading = (
        f"Query: members(direction in {{{requested}}}, reference={reference})"
        if trace_format is TraceFormat.SYMBOLIC
        else f"Query: classify which candidates are {requested} of {reference}."
    )
    lines = [heading]
    lines.extend(
        _membership_line(membership, explanation.directions, labels, trace_format)
        for membership in explanation.memberships
    )
    entailed = [
        _label(membership.candidate, labels)
        for membership in explanation.memberships
        if membership.evidence.status is ClaimStatus.ENTAILED
    ]
    contingent = [
        _label(membership.candidate, labels)
        for membership in explanation.memberships
        if membership.evidence.status is ClaimStatus.CONTINGENT
    ]
    if trace_format is TraceFormat.SYMBOLIC:
        lines.extend(
            (
                f"Entailed-Members: {{{', '.join(entailed)}}}",
                f"Contingent-Members: {{{', '.join(contingent)}}}",
            )
        )
    else:
        lines.extend(
            (
                f"Entailed members: {', '.join(entailed) if entailed else 'none'}.",
                f"Contingent members: {', '.join(contingent) if contingent else 'none'}.",
            )
        )
    return lines


def _count_lines(
    explanation: CountExplanation,
    labels: Mapping[str, str],
    trace_format: TraceFormat,
) -> list[str]:
    reference = _label(explanation.reference, labels)
    requested = _directions(explanation.directions)
    heading = (
        f"Query: count(direction in {{{requested}}}, reference={reference})"
        if trace_format is TraceFormat.SYMBOLIC
        else f"Query: count candidates that are {requested} of {reference}."
    )
    lines = [heading]
    lines.extend(
        _membership_line(membership, explanation.directions, labels, trace_format)
        for membership in explanation.memberships
    )
    for case in explanation.counts:
        if case.status not in {ClaimStatus.ENTAILED, ClaimStatus.CONTINGENT}:
            continue
        members = ", ".join(_label(member, labels) for member in case.members)
        if trace_format is TraceFormat.SYMBOLIC:
            lines.append(f"Members(k={case.count})={{{members}}}")
        else:
            description = members if members else "no candidates"
            lines.append(
                f"One jointly realizable case contains {description}, giving count {case.count}."
            )
    counts = ", ".join(map(str, explanation.possible_counts))
    lines.append(
        f"Count-Domain: {{{counts}}}"
        if trace_format is TraceFormat.SYMBOLIC
        else f"Possible counts: {counts if counts else 'none'}."
    )
    lines.append(
        f"Entailed count: {counts}."
        if len(explanation.possible_counts) == 1
        else "No exact count is entailed."
    )
    return lines


def render_training_trace(
    problem: SpatialProblem,
    explanation: QueryExplanation,
    trace_format: TraceFormat | str,
    state_mode: StateMode | str = StateMode.FINAL_ONLY,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render a coordinate-free trace suitable for an SFT target."""
    trace_format = TraceFormat(trace_format)
    state_mode = StateMode(state_mode)
    labels = labels or {}
    lines = _premise_lines(problem, labels, trace_format, state_mode)
    lines.append("Final Deduction:")
    if isinstance(problem.query, DirectionQuery) and isinstance(
        explanation, DirectionExplanation
    ):
        lines.extend(_direction_lines(explanation, labels, trace_format))
    elif isinstance(problem.query, WhichQuery) and isinstance(
        explanation, WhichExplanation
    ):
        lines.extend(_which_lines(explanation, labels, trace_format))
    elif isinstance(problem.query, CountQuery) and isinstance(
        explanation, CountExplanation
    ):
        lines.extend(_count_lines(explanation, labels, trace_format))
    else:
        raise TypeError("explanation type does not match the problem query")
    return "\n".join(lines)
