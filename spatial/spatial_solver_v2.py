"""Data-agnostic exact eight-direction spatial reasoning.

V2 models the relative position of two distinct objects as one of eight
mutually exclusive atoms.  Each atom is the product of an X comparison and a
Y comparison; equality on one axis is allowed, while equality on both axes is
not.  Premises may constrain a pair to one atom, a coarse half-plane, or the
complement of either.

The public interface accepts a structured ``SpatialProblem`` and returns a
query-specific analysis. Text parsing, dataset schemas, option menus, oracle labels,
and answer policies live in adapters above this module.

This module is experimental and does not alter the V13 solver or datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from itertools import combinations
from typing import Protocol


class Direction(str, Enum):
    NORTH = "North"
    NORTHEAST = "Northeast"
    EAST = "East"
    SOUTHEAST = "Southeast"
    SOUTH = "South"
    SOUTHWEST = "Southwest"
    WEST = "West"
    NORTHWEST = "Northwest"

    @classmethod
    def from_label(cls, value: str) -> Direction | None:
        lowered = value.strip().rstrip(".").lower()
        return next(
            (direction for direction in cls if direction.value.lower() == lowered),
            None,
        )


_DIRECTION_SIGNS: dict[Direction, tuple[int, int]] = {
    Direction.NORTH: (0, 1),
    Direction.NORTHEAST: (1, 1),
    Direction.EAST: (1, 0),
    Direction.SOUTHEAST: (1, -1),
    Direction.SOUTH: (0, -1),
    Direction.SOUTHWEST: (-1, -1),
    Direction.WEST: (-1, 0),
    Direction.NORTHWEST: (-1, 1),
}
_SIGNS_DIRECTION = {signs: direction for direction, signs in _DIRECTION_SIGNS.items()}
_ALL_DIRECTIONS = frozenset(Direction)
_OPPOSITE = {
    direction: _SIGNS_DIRECTION[(-x_sign, -y_sign)]
    for direction, (x_sign, y_sign) in _DIRECTION_SIGNS.items()
}


def _compose_signs(first: int, second: int) -> frozenset[int]:
    if first == 0:
        return frozenset({second})
    if second == 0 or first == second:
        return frozenset({first})
    return frozenset({-1, 0, 1})


@cache
def _compose_directions(
    first: frozenset[Direction], second: frozenset[Direction]
) -> frozenset[Direction]:
    possible: set[Direction] = set()
    for first_direction in first:
        first_x, first_y = _DIRECTION_SIGNS[first_direction]
        for second_direction in second:
            second_x, second_y = _DIRECTION_SIGNS[second_direction]
            for x_sign in _compose_signs(first_x, second_x):
                for y_sign in _compose_signs(first_y, second_y):
                    if (x_sign, y_sign) != (0, 0):
                        possible.add(_SIGNS_DIRECTION[(x_sign, y_sign)])
    return frozenset(possible)


class SpatialFormula:
    """Marker base for structured spatial propositions."""


@dataclass(frozen=True)
class RelationConstraint(SpatialFormula):
    """Allowed exact directions for ``subject`` relative to ``reference``."""

    subject: str
    reference: str
    allowed: frozenset[Direction]


@dataclass(frozen=True)
class Not(SpatialFormula):
    operand: SpatialFormula


@dataclass(frozen=True)
class And(SpatialFormula):
    operands: tuple[SpatialFormula, ...]


@dataclass(frozen=True)
class Or(SpatialFormula):
    operands: tuple[SpatialFormula, ...]


@dataclass(frozen=True)
class Implies(SpatialFormula):
    antecedent: SpatialFormula
    consequent: SpatialFormula


@dataclass(frozen=True)
class Iff(SpatialFormula):
    left: SpatialFormula
    right: SpatialFormula


def conjunctive_atoms(
    formula: SpatialFormula,
) -> tuple[RelationConstraint, ...] | None:
    """Return atoms only when the whole formula is a positive conjunction."""
    if isinstance(formula, RelationConstraint):
        return (formula,)
    if isinstance(formula, And):
        atoms: list[RelationConstraint] = []
        for operand in formula.operands:
            part = conjunctive_atoms(operand)
            if part is None:
                return None
            atoms.extend(part)
        return tuple(atoms)
    return None


def _required_atoms(formula: SpatialFormula) -> tuple[RelationConstraint, ...]:
    """Return positive atoms that every satisfying assignment must satisfy."""
    if isinstance(formula, RelationConstraint):
        return (formula,)
    if isinstance(formula, And):
        return tuple(
            atom for operand in formula.operands for atom in _required_atoms(operand)
        )
    return ()


@dataclass(frozen=True)
class DirectionQuery:
    target: str
    reference: str
    candidate_directions: frozenset[Direction] = _ALL_DIRECTIONS


@dataclass(frozen=True)
class WhichQuery:
    directions: frozenset[Direction]
    reference: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class CountQuery:
    directions: frozenset[Direction]
    reference: str
    candidates: tuple[str, ...]


SpatialQuery = DirectionQuery | WhichQuery | CountQuery


def _membership_constraint(
    candidate: str, query: WhichQuery | CountQuery
) -> RelationConstraint:
    return RelationConstraint(candidate, query.reference, query.directions)


@dataclass(frozen=True)
class SpatialProblem:
    objects: tuple[str, ...]
    premise: SpatialFormula
    query: SpatialQuery

    def __post_init__(self) -> None:
        if not self.objects or len(set(self.objects)) != len(self.objects):
            raise ValueError("objects must be a non-empty tuple of unique names")
        object_set = set(self.objects)
        if isinstance(self.query, DirectionQuery):
            if self.query.target == self.query.reference:
                raise ValueError("a direction query requires two distinct objects")
            if (
                self.query.target not in object_set
                or self.query.reference not in object_set
            ):
                raise ValueError("query references an object absent from the problem")
            if (
                not self.query.candidate_directions
                or not self.query.candidate_directions <= _ALL_DIRECTIONS
            ):
                raise ValueError("candidate_directions must contain exact directions")
        elif isinstance(self.query, (WhichQuery, CountQuery)):
            if self.query.reference not in object_set:
                raise ValueError("query references an object absent from the problem")
            if (
                not self.query.candidates
                or len(set(self.query.candidates)) != len(self.query.candidates)
                or any(
                    candidate not in object_set or candidate == self.query.reference
                    for candidate in self.query.candidates
                )
            ):
                raise ValueError(
                    "query candidates must be unique non-reference objects"
                )
            if (
                not self.query.directions
                or not self.query.directions <= _ALL_DIRECTIONS
            ):
                raise ValueError("query directions must contain exact directions")
        else:
            raise TypeError(f"unsupported spatial query: {type(self.query).__name__}")
        formulas = _walk_formulas(self.premise)
        for formula in formulas:
            if isinstance(formula, (And, Or)) and not formula.operands:
                raise ValueError(
                    f"{type(formula).__name__} requires at least one operand"
                )
        for constraint in (
            formula for formula in formulas if isinstance(formula, RelationConstraint)
        ):
            if constraint.subject == constraint.reference:
                raise ValueError("a relation requires two distinct objects")
            if (
                constraint.subject not in object_set
                or constraint.reference not in object_set
            ):
                raise ValueError(
                    "constraint references an object absent from the problem"
                )
            if not constraint.allowed or not constraint.allowed <= _ALL_DIRECTIONS:
                raise ValueError("constraint must allow at least one exact direction")


def _walk_formulas(formula: SpatialFormula) -> tuple[SpatialFormula, ...]:
    if isinstance(formula, RelationConstraint):
        return (formula,)
    if isinstance(formula, Not):
        return (formula,) + _walk_formulas(formula.operand)
    if isinstance(formula, (And, Or)):
        return (formula,) + tuple(
            child for operand in formula.operands for child in _walk_formulas(operand)
        )
    if isinstance(formula, Implies):
        return (
            (formula,)
            + _walk_formulas(formula.antecedent)
            + _walk_formulas(formula.consequent)
        )
    if isinstance(formula, Iff):
        return (formula,) + _walk_formulas(formula.left) + _walk_formulas(formula.right)
    raise TypeError(f"unsupported spatial formula: {type(formula).__name__}")


@dataclass(frozen=True)
class DirectionAnalysis:
    consistent: bool
    target: str = ""
    reference: str = ""
    possible_directions: tuple[Direction, ...] = ()
    engine: str = ""
    coordinates: dict[str, tuple[int, int]] | None = None
    witnesses: dict[Direction, dict[str, tuple[int, int]]] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class WhichAnalysis:
    consistent: bool
    reference: str
    directions: frozenset[Direction]
    candidates: tuple[str, ...]
    possible_entities: tuple[str, ...] = ()
    entailed_entities: tuple[str, ...] = ()
    witnesses: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)
    engine: str = ""
    error: str | None = None

    @property
    def contingent_entities(self) -> tuple[str, ...]:
        entailed = set(self.entailed_entities)
        return tuple(
            entity for entity in self.possible_entities if entity not in entailed
        )

    @property
    def impossible_entities(self) -> tuple[str, ...]:
        possible = set(self.possible_entities)
        return tuple(entity for entity in self.candidates if entity not in possible)


@dataclass(frozen=True)
class CountAnalysis:
    consistent: bool
    reference: str
    directions: frozenset[Direction]
    candidates: tuple[str, ...]
    possible_counts: tuple[int, ...] = ()
    witnesses: dict[int, dict[str, tuple[int, int]]] = field(default_factory=dict)
    engine: str = ""
    error: str | None = None


QueryAnalysis = DirectionAnalysis | WhichAnalysis | CountAnalysis


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


class _ConstraintEngine(Protocol):
    name: str

    def analyze(
        self,
        objects: tuple[str, ...],
        premise: SpatialFormula,
        target: str,
        reference: str,
        candidate_directions: frozenset[Direction],
    ) -> dict[Direction, dict[str, tuple[int, int]]] | None: ...

    def analyze_which(
        self,
        objects: tuple[str, ...],
        premise: SpatialFormula,
        query: WhichQuery,
    ) -> (
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            dict[str, dict[str, tuple[int, int]]],
        ]
        | None
    ): ...

    def analyze_count(
        self,
        objects: tuple[str, ...],
        premise: SpatialFormula,
        query: CountQuery,
    ) -> dict[int, dict[str, tuple[int, int]]] | None: ...


def _canonical_constraints(
    constraints: tuple[RelationConstraint, ...],
) -> list[tuple[str, str, frozenset[Direction]]]:
    combined: dict[tuple[str, str], frozenset[Direction]] = {}
    for constraint in constraints:
        if constraint.subject < constraint.reference:
            key = (constraint.subject, constraint.reference)
            allowed = constraint.allowed
        else:
            key = (constraint.reference, constraint.subject)
            allowed = frozenset(_OPPOSITE[value] for value in constraint.allowed)
        combined[key] = combined.get(key, _ALL_DIRECTIONS) & allowed
    return [
        (first, second, allowed)
        for (first, second), allowed in sorted(combined.items())
    ]


def _propagate(
    objects: tuple[str, ...], constraints: tuple[RelationConstraint, ...]
) -> tuple[RelationConstraint, ...] | None:
    """Apply sound path-consistency pruning; return ``None`` on contradiction."""
    domains = {
        (first, second): _ALL_DIRECTIONS
        for first in objects
        for second in objects
        if first != second
    }
    for constraint in constraints:
        pair = (constraint.subject, constraint.reference)
        domains[pair] &= constraint.allowed
        if not domains[pair]:
            return None
        reverse = (constraint.reference, constraint.subject)
        domains[reverse] &= frozenset(_OPPOSITE[value] for value in domains[pair])
        if not domains[reverse]:
            return None

    changed = True
    while changed:
        changed = False
        for first in objects:
            for middle in objects:
                if middle == first:
                    continue
                for last in objects:
                    if last in {first, middle}:
                        continue
                    pair = (first, last)
                    narrowed = domains[pair] & _compose_directions(
                        domains[(first, middle)], domains[(middle, last)]
                    )
                    if not narrowed:
                        return None
                    if narrowed == domains[pair]:
                        continue
                    domains[pair] = narrowed
                    domains[(last, first)] = frozenset(
                        _OPPOSITE[value] for value in narrowed
                    )
                    changed = True

    return tuple(
        RelationConstraint(first, second, domains[(first, second)])
        for index, first in enumerate(objects)
        for second in objects[index + 1 :]
        if domains[(first, second)] != _ALL_DIRECTIONS
    )


class _ReferenceEngine:
    """Dependency-free exhaustive engine for small cases and cross-checking."""

    name = "reference"

    @staticmethod
    def _axis_coordinates(
        objects: tuple[str, ...],
        union_find: _UnionFind,
        comparisons: list[tuple[str, str]],
    ) -> dict[str, int] | None:
        adjacency: dict[str, set[str]] = {}
        indegree: dict[str, int] = {}
        roots = {union_find.find(obj) for obj in objects}
        for root in roots:
            adjacency[root] = set()
            indegree[root] = 0
        for lower, higher in comparisons:
            lower_root = union_find.find(lower)
            higher_root = union_find.find(higher)
            if lower_root == higher_root:
                return None
            if higher_root not in adjacency[lower_root]:
                adjacency[lower_root].add(higher_root)
                indegree[higher_root] += 1
        frontier = sorted(root for root, degree in indegree.items() if degree == 0)
        ordered_roots: list[str] = []
        while frontier:
            root = frontier.pop(0)
            ordered_roots.append(root)
            for higher in sorted(adjacency[root]):
                indegree[higher] -= 1
                if indegree[higher] == 0:
                    frontier.append(higher)
                    frontier.sort()
        if len(ordered_roots) != len(roots):
            return None
        rank = {root: index for index, root in enumerate(ordered_roots)}
        return {obj: rank[union_find.find(obj)] for obj in objects}

    @classmethod
    def _coordinates_from_atoms(
        cls,
        objects: tuple[str, ...],
        selected: list[tuple[str, str, Direction]],
    ) -> dict[str, tuple[int, int]] | None:
        x_equal = _UnionFind(objects)
        y_equal = _UnionFind(objects)
        for subject, reference, direction in selected:
            x_sign, y_sign = _DIRECTION_SIGNS[direction]
            if x_sign == 0:
                x_equal.union(subject, reference)
            if y_sign == 0:
                y_equal.union(subject, reference)

        x_comparisons: list[tuple[str, str]] = []
        y_comparisons: list[tuple[str, str]] = []
        for subject, reference, direction in selected:
            x_sign, y_sign = _DIRECTION_SIGNS[direction]
            if x_sign > 0:
                x_comparisons.append((reference, subject))
            elif x_sign < 0:
                x_comparisons.append((subject, reference))
            if y_sign > 0:
                y_comparisons.append((reference, subject))
            elif y_sign < 0:
                y_comparisons.append((subject, reference))

        x_coordinates = cls._axis_coordinates(objects, x_equal, x_comparisons)
        y_coordinates = cls._axis_coordinates(objects, y_equal, y_comparisons)
        if x_coordinates is None or y_coordinates is None:
            return None
        if any(
            x_equal.find(first) == x_equal.find(second)
            and y_equal.find(first) == y_equal.find(second)
            for index, first in enumerate(objects)
            for second in objects[index + 1 :]
        ):
            return None
        return {obj: (x_coordinates[obj], y_coordinates[obj]) for obj in objects}

    def _find_witness(
        self,
        objects: tuple[str, ...],
        constraints: tuple[RelationConstraint, ...],
        assumption: tuple[str, str, Direction] | None = None,
    ) -> dict[str, tuple[int, int]] | None:
        if assumption is not None:
            subject, reference, direction = assumption
            constraints += (
                RelationConstraint(subject, reference, frozenset({direction})),
            )
        canonical = _canonical_constraints(constraints)
        if any(not allowed for _first, _second, allowed in canonical):
            return None
        canonical.sort(key=lambda item: len(item[2]))

        def search(
            index: int, selected: list[tuple[str, str, Direction]]
        ) -> dict[str, tuple[int, int]] | None:
            coordinates = self._coordinates_from_atoms(objects, selected)
            if coordinates is None:
                return None
            if index == len(canonical):
                return coordinates
            first, second, allowed = canonical[index]
            for direction in sorted(allowed, key=lambda item: item.value):
                witness = search(index + 1, selected + [(first, second, direction)])
                if witness is not None:
                    return witness
            return None

        return search(0, [])

    @staticmethod
    def _evaluate_formula(
        formula: SpatialFormula,
        assignments: dict[tuple[str, str], Direction],
    ) -> bool:
        if isinstance(formula, RelationConstraint):
            if formula.subject < formula.reference:
                direction = assignments[(formula.subject, formula.reference)]
            else:
                direction = _OPPOSITE[assignments[(formula.reference, formula.subject)]]
            return direction in formula.allowed
        if isinstance(formula, Not):
            return not _ReferenceEngine._evaluate_formula(formula.operand, assignments)
        if isinstance(formula, And):
            return all(
                _ReferenceEngine._evaluate_formula(operand, assignments)
                for operand in formula.operands
            )
        if isinstance(formula, Or):
            return any(
                _ReferenceEngine._evaluate_formula(operand, assignments)
                for operand in formula.operands
            )
        if isinstance(formula, Implies):
            return not _ReferenceEngine._evaluate_formula(
                formula.antecedent, assignments
            ) or _ReferenceEngine._evaluate_formula(formula.consequent, assignments)
        if isinstance(formula, Iff):
            return _ReferenceEngine._evaluate_formula(
                formula.left, assignments
            ) == _ReferenceEngine._evaluate_formula(formula.right, assignments)
        raise TypeError(f"unsupported spatial formula: {type(formula).__name__}")

    def _find_formula_witness(
        self,
        objects: tuple[str, ...],
        premise: SpatialFormula,
        target: str,
        reference: str,
        assumption: Direction | None = None,
    ) -> dict[str, tuple[int, int]] | None:
        pairs = {
            tuple(sorted((atom.subject, atom.reference)))
            for atom in _walk_formulas(premise)
            if isinstance(atom, RelationConstraint)
        }
        query_pair = tuple(sorted((target, reference)))
        pairs.add(query_pair)
        ordered_pairs = sorted(pairs)
        domains = {pair: _ALL_DIRECTIONS for pair in ordered_pairs}
        if assumption is not None:
            domains[query_pair] = frozenset(
                {assumption if target < reference else _OPPOSITE[assumption]}
            )

        def search(
            index: int, assignments: dict[tuple[str, str], Direction]
        ) -> dict[str, tuple[int, int]] | None:
            selected = [
                (first, second, direction)
                for (first, second), direction in assignments.items()
            ]
            coordinates = self._coordinates_from_atoms(objects, selected)
            if coordinates is None:
                return None
            if index == len(ordered_pairs):
                return (
                    coordinates
                    if self._evaluate_formula(premise, assignments)
                    else None
                )
            pair = ordered_pairs[index]
            for direction in sorted(domains[pair], key=lambda item: item.value):
                witness = search(index + 1, {**assignments, pair: direction})
                if witness is not None:
                    return witness
            return None

        return search(0, {})

    def _analyze_formula(
        self,
        objects: tuple[str, ...],
        premise: SpatialFormula,
        target: str,
        reference: str,
        candidate_directions: frozenset[Direction],
    ) -> dict[Direction, dict[str, tuple[int, int]]] | None:
        if self._find_formula_witness(objects, premise, target, reference) is None:
            return None
        witnesses = {}
        for direction in Direction:
            if direction not in candidate_directions:
                continue
            witness = self._find_formula_witness(
                objects, premise, target, reference, direction
            )
            if witness is not None:
                witnesses[direction] = witness
        return witnesses

    def analyze(
        self,
        objects: tuple[str, ...],
        premise: SpatialFormula,
        target: str,
        reference: str,
        candidate_directions: frozenset[Direction],
    ) -> dict[Direction, dict[str, tuple[int, int]]] | None:
        constraints = conjunctive_atoms(premise)
        if constraints is None:
            return self._analyze_formula(
                objects, premise, target, reference, candidate_directions
            )
        if self._find_witness(objects, constraints) is None:
            return None
        witnesses = {}
        for direction in Direction:
            if direction not in candidate_directions:
                continue
            witness = self._find_witness(
                objects, constraints, (target, reference, direction)
            )
            if witness is not None:
                witnesses[direction] = witness
        return witnesses

    def analyze_which(self, objects, premise, query):
        if (
            self._find_formula_witness(
                objects, premise, query.candidates[0], query.reference
            )
            is None
        ):
            return None
        possible: list[str] = []
        entailed: list[str] = []
        witnesses: dict[str, dict[str, tuple[int, int]]] = {}
        for candidate in query.candidates:
            predicate = _membership_constraint(candidate, query)
            witness = self._find_formula_witness(
                objects,
                And((premise, predicate)),
                candidate,
                query.reference,
            )
            if witness is not None:
                possible.append(candidate)
                witnesses[candidate] = witness
            counterexample = self._find_formula_witness(
                objects,
                And((premise, Not(predicate))),
                candidate,
                query.reference,
            )
            if counterexample is None:
                entailed.append(candidate)
        return tuple(possible), tuple(entailed), witnesses

    def analyze_count(self, objects, premise, query):
        if (
            self._find_formula_witness(
                objects, premise, query.candidates[0], query.reference
            )
            is None
        ):
            return None
        predicates = {
            candidate: _membership_constraint(candidate, query)
            for candidate in query.candidates
        }
        witnesses: dict[int, dict[str, tuple[int, int]]] = {}
        for count in range(len(query.candidates) + 1):
            alternatives = []
            for included in combinations(query.candidates, count):
                included_set = set(included)
                alternatives.append(
                    And(
                        tuple(
                            predicates[candidate]
                            if candidate in included_set
                            else Not(predicates[candidate])
                            for candidate in query.candidates
                        )
                    )
                )
            count_formula: SpatialFormula = (
                alternatives[0] if len(alternatives) == 1 else Or(tuple(alternatives))
            )
            witness = self._find_formula_witness(
                objects,
                And((premise, count_formula)),
                query.candidates[0],
                query.reference,
            )
            if witness is not None:
                witnesses[count] = witness
        return witnesses


class _Z3Engine:
    """Incremental SMT engine for larger disjunctive worlds."""

    name = "z3"

    def __init__(self, timeout_ms: int) -> None:
        try:
            import z3
        except ImportError as exc:
            raise RuntimeError(
                "the z3 backend requires the 'z3-solver' package"
            ) from exc
        self.z3 = z3
        self.timeout_ms = timeout_ms

    def _atom(self, x, y, subject: str, reference: str, direction: Direction):
        z3 = self.z3
        x_sign, y_sign = _DIRECTION_SIGNS[direction]

        def comparison(values, sign: int):
            if sign < 0:
                return values[subject] < values[reference]
            if sign > 0:
                return values[subject] > values[reference]
            return values[subject] == values[reference]

        return z3.And(comparison(x, x_sign), comparison(y, y_sign))

    def _formula(self, x, y, formula: SpatialFormula):
        z3 = self.z3
        if isinstance(formula, RelationConstraint):
            return z3.Or(
                *(
                    self._atom(
                        x,
                        y,
                        formula.subject,
                        formula.reference,
                        direction,
                    )
                    for direction in formula.allowed
                )
            )
        if isinstance(formula, Not):
            return z3.Not(self._formula(x, y, formula.operand))
        if isinstance(formula, And):
            return z3.And(
                *(self._formula(x, y, operand) for operand in formula.operands)
            )
        if isinstance(formula, Or):
            return z3.Or(
                *(self._formula(x, y, operand) for operand in formula.operands)
            )
        if isinstance(formula, Implies):
            return z3.Implies(
                self._formula(x, y, formula.antecedent),
                self._formula(x, y, formula.consequent),
            )
        if isinstance(formula, Iff):
            return self._formula(x, y, formula.left) == self._formula(
                x, y, formula.right
            )
        raise TypeError(f"unsupported spatial formula: {type(formula).__name__}")

    def _region(self, x, y, candidate: str, query: WhichQuery | CountQuery):
        return self.z3.Or(
            *(
                self._atom(x, y, candidate, query.reference, direction)
                for direction in query.directions
            )
        )

    def _check(self, solver) -> bool:
        verdict = solver.check()
        if verdict == self.z3.unknown:
            raise RuntimeError(f"z3 returned unknown: {solver.reason_unknown()}")
        return verdict == self.z3.sat

    @staticmethod
    def _normalize_coordinates(
        objects: tuple[str, ...], model, x, y
    ) -> dict[str, tuple[int, int]]:
        raw_x = {
            obj: model.eval(x[obj], model_completion=True).as_long() for obj in objects
        }
        raw_y = {
            obj: model.eval(y[obj], model_completion=True).as_long() for obj in objects
        }
        x_ranks = {
            value: rank for rank, value in enumerate(sorted(set(raw_x.values())))
        }
        y_ranks = {
            value: rank for rank, value in enumerate(sorted(set(raw_y.values())))
        }
        return {obj: (x_ranks[raw_x[obj]], y_ranks[raw_y[obj]]) for obj in objects}

    def _build_solver(self, objects: tuple[str, ...], premise: SpatialFormula):
        propagated = _propagate(objects, _required_atoms(premise))
        if propagated is None:
            return None

        z3 = self.z3
        x = {obj: z3.Int(f"x_{index}") for index, obj in enumerate(objects)}
        y = {obj: z3.Int(f"y_{index}") for index, obj in enumerate(objects)}
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        for index, first in enumerate(objects):
            for second in objects[index + 1 :]:
                solver.add(z3.Or(x[first] != x[second], y[first] != y[second]))
        for constraint in propagated:
            solver.add(
                z3.Or(
                    *(
                        self._atom(
                            x,
                            y,
                            constraint.subject,
                            constraint.reference,
                            direction,
                        )
                        for direction in constraint.allowed
                    )
                )
            )
        solver.add(self._formula(x, y, premise))
        return solver, x, y

    def analyze(
        self,
        objects: tuple[str, ...],
        premise: SpatialFormula,
        target: str,
        reference: str,
        candidate_directions: frozenset[Direction],
    ) -> dict[Direction, dict[str, tuple[int, int]]] | None:
        built = self._build_solver(objects, premise)
        if built is None:
            return None
        solver, x, y = built
        if not self._check(solver):
            return None

        witnesses: dict[Direction, dict[str, tuple[int, int]]] = {}
        for direction in Direction:
            if direction not in candidate_directions:
                continue
            solver.push()
            solver.add(self._atom(x, y, target, reference, direction))
            if self._check(solver):
                witnesses[direction] = self._normalize_coordinates(
                    objects, solver.model(), x, y
                )
            solver.pop()
        return witnesses

    def analyze_which(
        self,
        objects: tuple[str, ...],
        premise: SpatialFormula,
        query: WhichQuery,
    ) -> (
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            dict[str, dict[str, tuple[int, int]]],
        ]
        | None
    ):
        built = self._build_solver(objects, premise)
        if built is None:
            return None
        solver, x, y = built
        if not self._check(solver):
            return None

        possible: list[str] = []
        entailed: list[str] = []
        witnesses: dict[str, dict[str, tuple[int, int]]] = {}
        for candidate in query.candidates:
            predicate = self._region(x, y, candidate, query)
            solver.push()
            solver.add(predicate)
            if self._check(solver):
                possible.append(candidate)
                witnesses[candidate] = self._normalize_coordinates(
                    objects, solver.model(), x, y
                )
            solver.pop()

            solver.push()
            solver.add(self.z3.Not(predicate))
            if not self._check(solver):
                entailed.append(candidate)
            solver.pop()
        return tuple(possible), tuple(entailed), witnesses

    def analyze_count(
        self,
        objects: tuple[str, ...],
        premise: SpatialFormula,
        query: CountQuery,
    ) -> dict[int, dict[str, tuple[int, int]]] | None:
        built = self._build_solver(objects, premise)
        if built is None:
            return None
        solver, x, y = built
        if not self._check(solver):
            return None

        count = self.z3.Sum(
            *(
                self.z3.If(
                    self._region(x, y, candidate, query),
                    1,
                    0,
                )
                for candidate in query.candidates
            )
        )
        witnesses: dict[int, dict[str, tuple[int, int]]] = {}
        for candidate_count in range(len(query.candidates) + 1):
            solver.push()
            solver.add(count == candidate_count)
            if self._check(solver):
                witnesses[candidate_count] = self._normalize_coordinates(
                    objects, solver.model(), x, y
                )
            solver.pop()
        return witnesses


class SpatialSolverV2:
    """Analyze structured eight-direction problems without dataset knowledge."""

    def __init__(self, backend: str = "auto", timeout_ms: int = 5_000) -> None:
        if backend not in {"auto", "reference", "z3"}:
            raise ValueError(f"unknown solver backend: {backend}")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if backend == "reference":
            self._engine: _ConstraintEngine = _ReferenceEngine()
        else:
            try:
                self._engine = _Z3Engine(timeout_ms)
            except RuntimeError:
                if backend == "z3":
                    raise
                self._engine = _ReferenceEngine()

    def analyze(self, problem: SpatialProblem) -> QueryAnalysis:
        if not isinstance(problem, SpatialProblem):
            raise TypeError("analyze expects a SpatialProblem")
        try:
            if isinstance(problem.query, DirectionQuery):
                return self._analyze_direction(problem, problem.query)
            if isinstance(problem.query, WhichQuery):
                return self._analyze_which(problem, problem.query)
            return self._analyze_count(problem, problem.query)
        except RuntimeError as exc:
            error = str(exc)
            if isinstance(problem.query, DirectionQuery):
                return DirectionAnalysis(False, engine=self._engine.name, error=error)
            if isinstance(problem.query, WhichQuery):
                return WhichAnalysis(
                    False,
                    problem.query.reference,
                    problem.query.directions,
                    problem.query.candidates,
                    engine=self._engine.name,
                    error=error,
                )
            return CountAnalysis(
                False,
                problem.query.reference,
                problem.query.directions,
                problem.query.candidates,
                engine=self._engine.name,
                error=error,
            )

    def _analyze_direction(
        self, problem: SpatialProblem, query: DirectionQuery
    ) -> DirectionAnalysis:
        witnesses = self._engine.analyze(
            problem.objects,
            problem.premise,
            query.target,
            query.reference,
            query.candidate_directions,
        )
        consistent = witnesses is not None
        witnesses = witnesses or {}
        return DirectionAnalysis(
            consistent,
            query.target,
            query.reference,
            tuple(witnesses),
            self._engine.name,
            next(iter(witnesses.values()), None),
            witnesses,
        )

    def _analyze_which(
        self, problem: SpatialProblem, query: WhichQuery
    ) -> WhichAnalysis:
        result = self._engine.analyze_which(problem.objects, problem.premise, query)
        consistent = result is not None
        possible, entailed, witnesses = result or ((), (), {})
        return WhichAnalysis(
            consistent,
            query.reference,
            query.directions,
            query.candidates,
            possible,
            entailed,
            witnesses,
            self._engine.name,
        )

    def _analyze_count(
        self, problem: SpatialProblem, query: CountQuery
    ) -> CountAnalysis:
        witnesses = self._engine.analyze_count(problem.objects, problem.premise, query)
        consistent = witnesses is not None
        witnesses = witnesses or {}
        return CountAnalysis(
            consistent,
            query.reference,
            query.directions,
            query.candidates,
            tuple(witnesses),
            witnesses,
            self._engine.name,
        )
