"""Coordinate-free natural-language and symbolic training traces."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from enum import Enum

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


class TraceStyle(str, Enum):
    AXIOMATIC = "axiomatic"
    SYMBOLIC = "symbolic"


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


def _axis_fact(
    atom: RelationConstraint,
    axis_index: int,
    labels: Mapping[str, str],
) -> str:
    direction = next(iter(atom.allowed))
    sign = direction_signs(direction)[axis_index]
    subject = _label(atom.subject, labels)
    reference = _label(atom.reference, labels)
    if sign < 0:
        return f"{subject} < {reference}"
    if sign > 0:
        return f"{reference} < {subject}"
    return f"{subject} = {reference}"


def _natural_axis_fact(
    atom: RelationConstraint,
    axis_index: int,
    labels: Mapping[str, str],
) -> str:
    direction = next(iter(atom.allowed))
    sign = direction_signs(direction)[axis_index]
    subject = _label(atom.subject, labels)
    reference = _label(atom.reference, labels)
    if sign == 0:
        axis = "X" if axis_index == 0 else "Y"
        return f"{subject} and {reference} have the same {axis} coordinate"
    return f"{subject} is {_axis_word(axis_index, sign)} of {reference}"


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


def _has_path(adjacency: dict[str, set[str]], start: str, end: str) -> bool:
    frontier = [start]
    visited = {start}
    while frontier:
        current = frontier.pop()
        for following in adjacency[current]:
            if following == end:
                return True
            if following not in visited:
                visited.add(following)
                frontier.append(following)
    return False


def _axis_state(
    objects: tuple[str, ...],
    atoms: tuple[RelationConstraint, ...],
    axis_index: int,
    labels: Mapping[str, str],
) -> str:
    equality = _UnionFind(objects)
    for atom in atoms:
        direction = next(iter(atom.allowed))
        if direction_signs(direction)[axis_index] == 0:
            equality.union(atom.subject, atom.reference)

    classes: dict[str, list[str]] = {}
    for obj in objects:
        classes.setdefault(equality.find(obj), []).append(obj)
    adjacency = {root: set() for root in classes}
    for atom in atoms:
        direction = next(iter(atom.allowed))
        sign = direction_signs(direction)[axis_index]
        if sign == 0:
            continue
        lower, higher = (
            (atom.subject, atom.reference)
            if sign < 0
            else (atom.reference, atom.subject)
        )
        lower_root = equality.find(lower)
        higher_root = equality.find(higher)
        if lower_root != higher_root:
            adjacency[lower_root].add(higher_root)

    reduced = {root: set(following) for root, following in adjacency.items()}
    for lower, following in adjacency.items():
        for higher in tuple(following):
            reduced[lower].remove(higher)
            if not _has_path(reduced, lower, higher):
                reduced[lower].add(higher)

    indegree = {root: 0 for root in classes}
    for following in reduced.values():
        for higher in following:
            indegree[higher] += 1

    def class_label(root: str) -> str:
        members = sorted(_label(obj, labels) for obj in classes[root])
        return members[0] if len(members) == 1 else "[" + " = ".join(members) + "]"

    paths: list[tuple[str, ...]] = []

    def visit(root: str, path: tuple[str, ...]) -> None:
        following = sorted(reduced[root], key=class_label)
        if not following:
            paths.append(path + (root,))
            return
        for higher in following:
            visit(higher, path + (root,))

    for root in sorted(
        (root for root, degree in indegree.items() if degree == 0),
        key=class_label,
    ):
        visit(root, ())
    return "; ".join(" < ".join(map(class_label, path)) for path in paths)


def _natural_axis_state(state: str, axis_index: int) -> str:
    paths = []
    for chain in state.split("; "):
        positions = []
        for position in chain.split(" < "):
            if position.startswith("[") and position.endswith("]"):
                members = position[1:-1].split(" = ")
                positions.append(" and ".join(members) + " at the same coordinate")
            else:
                positions.append(position)
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


def _premise_lines(
    problem: SpatialProblem,
    labels: Mapping[str, str],
    style: TraceStyle,
) -> list[str]:
    atoms = conjunctive_atoms(problem.premise)
    if atoms is None or any(len(atom.allowed) != 1 for atom in atoms):
        return [f"Premise formula: {_formula_text(problem.premise, labels)}."]

    lines = []
    for index, atom in enumerate(atoms, 1):
        relation = _relation_text(atom, labels)
        x_fact = _axis_fact(atom, 0, labels)
        y_fact = _axis_fact(atom, 1, labels)
        if style is TraceStyle.AXIOMATIC:
            lines.extend(
                (
                    f"Premise {index}: {relation}.",
                    f"  X-axis decomposition: {_natural_axis_fact(atom, 0, labels)}.",
                    f"  Y-axis decomposition: {_natural_axis_fact(atom, 1, labels)}.",
                )
            )
        else:
            lines.append(f"P{index}: {relation} => X[{x_fact}], Y[{y_fact}]")

    x_state = _axis_state(problem.objects, atoms, 0, labels)
    y_state = _axis_state(problem.objects, atoms, 1, labels)
    if style is TraceStyle.AXIOMATIC:
        lines.extend(
            (
                f"X-axis ordering {_natural_axis_state(x_state, 0)}.",
                f"Y-axis ordering {_natural_axis_state(y_state, 1)}.",
            )
        )
    else:
        lines.extend((f"X-State: {x_state}", f"Y-State: {y_state}"))
    return lines


def _direction_lines(
    explanation: DirectionExplanation,
    labels: Mapping[str, str],
    style: TraceStyle,
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
        if style is TraceStyle.SYMBOLIC:
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

    if style is TraceStyle.SYMBOLIC:
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
    style: TraceStyle,
) -> str:
    candidate = _label(membership.candidate, labels)
    domain = _directions(membership.possible_directions) or "none"
    requested_text = _directions(requested)
    if style is TraceStyle.SYMBOLIC:
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
    style: TraceStyle,
) -> list[str]:
    reference = _label(explanation.reference, labels)
    requested = _directions(explanation.directions)
    heading = (
        f"Query: members(direction in {{{requested}}}, reference={reference})"
        if style is TraceStyle.SYMBOLIC
        else f"Query: classify which candidates are {requested} of {reference}."
    )
    lines = [heading]
    lines.extend(
        _membership_line(membership, explanation.directions, labels, style)
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
    if style is TraceStyle.SYMBOLIC:
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
    style: TraceStyle,
) -> list[str]:
    reference = _label(explanation.reference, labels)
    requested = _directions(explanation.directions)
    heading = (
        f"Query: count(direction in {{{requested}}}, reference={reference})"
        if style is TraceStyle.SYMBOLIC
        else f"Query: count candidates that are {requested} of {reference}."
    )
    lines = [heading]
    lines.extend(
        _membership_line(membership, explanation.directions, labels, style)
        for membership in explanation.memberships
    )
    for case in explanation.counts:
        if case.status not in {ClaimStatus.ENTAILED, ClaimStatus.CONTINGENT}:
            continue
        members = ", ".join(_label(member, labels) for member in case.members)
        if style is TraceStyle.SYMBOLIC:
            lines.append(f"Members(k={case.count})={{{members}}}")
        else:
            description = members if members else "no candidates"
            lines.append(
                f"One jointly realizable case contains {description}, giving count {case.count}."
            )
    counts = ", ".join(map(str, explanation.possible_counts))
    lines.append(
        f"Count-Domain: {{{counts}}}"
        if style is TraceStyle.SYMBOLIC
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
    style: TraceStyle | str,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render a coordinate-free trace suitable for an SFT target."""
    style = TraceStyle(style)
    labels = labels or {}
    lines = _premise_lines(problem, labels, style)
    lines.append("Final Deduction:")
    if isinstance(problem.query, DirectionQuery) and isinstance(
        explanation, DirectionExplanation
    ):
        lines.extend(_direction_lines(explanation, labels, style))
    elif isinstance(problem.query, WhichQuery) and isinstance(
        explanation, WhichExplanation
    ):
        lines.extend(_which_lines(explanation, labels, style))
    elif isinstance(problem.query, CountQuery) and isinstance(
        explanation, CountExplanation
    ):
        lines.extend(_count_lines(explanation, labels, style))
    else:
        raise TypeError("explanation type does not match the problem query")
    return "\n".join(lines)
