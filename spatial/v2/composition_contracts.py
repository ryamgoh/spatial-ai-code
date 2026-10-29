"""Composition-v1 six-option interface, independent of historical menu grading.

Serialization is ascending, undelimited uppercase letters (e.g. AB), or E/F
alone. Stage 1 requires one terminal ``Answer: <letters>`` line; Stage 2
accepts only the serialized answer. No legacy parser behavior is changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

from spatial.v2.grading import AnswerMode
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionAnalysis,
    DirectionQuery,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
    WhichAnalysis,
    WhichQuery,
)

VERSION = "spatialentail-composition-v1"
Value = str | int


class VerificationError(ValueError):
    """An example failed verification; it must not receive a semantic label."""


class UnsupportedEvidence(ValueError):
    """The current explanation path does not support this evidence category."""


def validate_problem(problem: SpatialProblem) -> None:
    if not isinstance(problem.premise, And) or not all(
        isinstance(atom, RelationConstraint) and len(atom.allowed) == 1
        for atom in problem.premise.operands
    ):
        raise ValueError("composition-v1 requires a flat conjunction of exact atoms")
    if any(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", obj) is None for obj in problem.objects
    ):
        raise ValueError("object names must be ASCII identifiers")
    query = problem.query
    if isinstance(query, DirectionQuery):
        if query.candidate_directions != frozenset(Direction):
            raise ValueError("Direction analysis must cover all eight directions")
    elif len(query.directions) != 1:
        raise ValueError("Selection and Count require one exact queried direction")


def answer_domain(problem: SpatialProblem) -> tuple[Value, ...]:
    query = problem.query
    if isinstance(query, DirectionQuery):
        return tuple(direction.value for direction in Direction)
    if isinstance(query, WhichQuery):
        return query.candidates
    return tuple(range(len(query.candidates) + 1))


@dataclass(frozen=True)
class Semantics:
    problem: SpatialProblem
    consistent: bool
    possible: frozenset[Value]
    certain: frozenset[Value]
    engine: str
    witnesses: dict[Value, dict[str, tuple[int, int]]]

    @property
    def case(self) -> str:
        if not self.consistent:
            return "inconsistent"
        if isinstance(self.problem.query, WhichQuery):
            if not self.possible:
                return "no-match"
            if len(self.certain) > 1:
                return "multiple-certain-match"
            if len(self.possible) == 1 and self.certain == self.possible:
                return "unique"
            return "underdetermined"
        return "unique" if len(self.possible) == 1 else "underdetermined"


# Deliberately independent of the solver's relation helpers for witness checks.
_SIGNS = {
    "North": (0, 1),
    "Northeast": (1, 1),
    "East": (1, 0),
    "Southeast": (1, -1),
    "South": (0, -1),
    "Southwest": (-1, -1),
    "West": (-1, 0),
    "Northwest": (-1, 1),
}


def check_coordinates(problem: SpatialProblem, coordinates: dict) -> None:
    """Validate a whole map independently of the solver's relation helpers."""
    if set(coordinates) != set(problem.objects):
        raise VerificationError("witness must assign every declared object")
    points = tuple(coordinates.values())
    if any(
        not isinstance(point, tuple)
        or len(point) != 2
        or any(type(component) is not int for component in point)
        for point in points
    ):
        raise VerificationError("witness coordinates must be integer pairs")
    if len(set(points)) != len(points):
        raise VerificationError("witness violates distinct points")

    def signs(subject, reference):
        return tuple(
            (a > b) - (a < b)
            for a, b in zip(coordinates[subject], coordinates[reference])
        )

    for atom in problem.premise.operands:
        direction = next(iter(atom.allowed)).value
        if signs(atom.subject, atom.reference) != _SIGNS[direction]:
            raise VerificationError("witness violates a premise")


def witness_outcome(
    problem: SpatialProblem, coordinates: dict
) -> Value | frozenset[str]:
    """Evaluate a checked map's query, including all simultaneous memberships."""
    check_coordinates(problem, coordinates)

    def signs(subject, reference):
        return tuple(
            (a > b) - (a < b)
            for a, b in zip(coordinates[subject], coordinates[reference])
        )

    query = problem.query
    if isinstance(query, DirectionQuery):
        return next(
            direction
            for direction, expected in _SIGNS.items()
            if signs(query.target, query.reference) == expected
        )
    expected = _SIGNS[next(iter(query.directions)).value]
    members = frozenset(
        obj for obj in query.candidates if signs(obj, query.reference) == expected
    )
    return members if isinstance(query, WhichQuery) else len(members)


def check_witness(problem: SpatialProblem, coordinates: dict, value: Value) -> None:
    outcome = witness_outcome(problem, coordinates)
    valid = (
        value in outcome if isinstance(problem.query, WhichQuery) else outcome == value
    )
    if not valid:
        raise VerificationError("witness does not demonstrate the claimed answer")


def analyze(
    problem: SpatialProblem, *, solver: SpatialSolverV2 | None = None
) -> Semantics:
    """Conclusive full-domain labels; not sufficient alone for synthetic release."""
    validate_problem(problem)
    solver = solver or SpatialSolverV2(backend="z3")
    try:
        result = solver.analyze(problem, include_witnesses=True)
    except RuntimeError as exc:
        raise VerificationError(f"solver failure: {exc}") from exc
    if result.error is not None:
        raise VerificationError(f"inconclusive solver analysis: {result.error}")
    if isinstance(result, DirectionAnalysis):
        possible = frozenset(d.value for d in result.possible_directions)
        certain = possible if len(possible) == 1 else frozenset()
    elif isinstance(result, WhichAnalysis):
        possible = frozenset(result.possible_entities)
        certain = frozenset(result.entailed_entities)
    else:
        possible = frozenset(result.possible_counts)
        certain = possible if len(possible) == 1 else frozenset()
    if not certain <= possible or not possible <= set(answer_domain(problem)):
        raise VerificationError("solver returned an invalid answer domain")
    if not result.consistent and possible:
        raise VerificationError("inconsistent analysis contains possible answers")
    if result.consistent and not possible and not isinstance(problem.query, WhichQuery):
        raise VerificationError(
            "consistent Direction/Count requires a nonempty answer set"
        )
    witnesses = {
        key.value if isinstance(key, Direction) else key: coordinates
        for key, coordinates in result.witnesses.items()
    }
    if set(witnesses) != possible:
        raise VerificationError("witnesses must cover every possible value")
    for value, coordinates in witnesses.items():
        check_witness(problem, coordinates, value)
    return Semantics(
        problem, result.consistent, possible, certain, result.engine, witnesses
    )


def validate_menu(problem: SpatialProblem, menu: tuple[Value, ...]) -> None:
    domain = answer_domain(problem)
    expected_type = int if isinstance(problem.query, CountQuery) else str
    if (
        len(menu) != 4
        or any(type(value) is not expected_type for value in menu)
        or len(set(menu)) != 4
        or not set(menu) <= set(domain)
    ):
        raise ValueError("menu must contain four distinct valid substantive values")


@dataclass(frozen=True)
class Decision:
    letters: str
    semantic_case: str
    escape_reason: str | None = None


def encode(semantics: Semantics, menu: tuple[Value, ...], mode: AnswerMode) -> Decision:
    mode = AnswerMode(mode)
    validate_menu(semantics.problem, menu)
    case = semantics.case
    if not semantics.consistent:
        return Decision("E", case, "inconsistent")
    selected = "".join(
        letter for letter, value in zip("ABCD", menu) if value in semantics.possible
    )
    if mode is AnswerMode.SINGLE:
        if case != "unique":
            return Decision("E", case, case)
        return Decision(selected or "F", case, None if selected else "unique-unlisted")
    if mode is AnswerMode.VISIBLE_POSSIBLE:
        reason = "empty-set" if not semantics.possible else "no-listed-possibilities"
        return Decision(selected or "F", case, None if selected else reason)
    if not semantics.possible:
        return Decision("F", case, "empty-set")
    if not semantics.possible <= set(menu):
        return Decision("F", case, "incomplete-coverage")
    return Decision(selected, case)


def allowed_answers(mode: AnswerMode) -> tuple[str, ...]:
    mode = AnswerMode(mode)
    sizes = (1,) if mode is AnswerMode.SINGLE else range(1, 5)
    return tuple(
        "".join(letters) for size in sizes for letters in combinations("ABCD", size)
    ) + ("E", "F")


def parse_answer(text: str, mode: AnswerMode, *, stage: int = 2) -> str:
    """Reject duplicate, reordered, punctuated, or mixed escape answers."""
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    text = text.strip()
    if stage == 1:
        lines = text.splitlines()
        markers = [line for line in lines if line.startswith("Answer:")]
        if len(markers) != 1 or not lines[-1].startswith("Answer: "):
            raise ValueError("expected exactly one terminal Answer: line")
        text = lines[-1][len("Answer: ") :]
    if text not in allowed_answers(mode):
        raise ValueError("invalid composition-v1 answer serialization")
    return text


def escape_labels(mode: AnswerMode) -> tuple[str, str]:
    mode = AnswerMode(mode)
    return (
        "Cannot be determined",
        "No complete answer can be formed from A-D."
        if mode is AnswerMode.ALL_POSSIBLE
        else "None of the above",
    )


_ASSUMPTIONS = (
    "Named objects are distinct points on fixed global x/y axes. No two objects "
    "coincide; sharing one coordinate is allowed. Directions are exact coordinate "
    "comparisons: North means equal x and greater y; Northeast means greater x "
    "and greater y. There are no distances, unit steps, grid spacing, or map bounds. "
    "Unstated relations are unknown, not false. A valid map satisfies every premise "
    "and these assumptions. Could means in at least one valid map. "
    "Under every contract, select E if no valid map exists."
)
_RULES = {
    AnswerMode.SINGLE: (
        "Select the one A-D value that is the invariant answer in every valid map. "
        "Select F if that unique answer is unlisted; otherwise select E. "
        "For Selection, select E when the premises do not establish exactly one "
        "qualifying answer, including when no object or multiple objects qualify."
    ),
    AnswerMode.VISIBLE_POSSIBLE: (
        "Select all A-D values possible in at least one valid map. "
        "Select F if no listed value is possible. Multiple possibilities do not mean E."
    ),
    AnswerMode.ALL_POSSIBLE: (
        "Select all A-D values only if they represent the complete nonempty possible "
        "set over the declared domain. Select F if any possible value is missing "
        "or the complete set is empty. Multiple possibilities do not mean E."
    ),
}


def _question(problem: SpatialProblem, mode: AnswerMode) -> str:
    query = problem.query
    if isinstance(query, DirectionQuery):
        target, reference = query.target, query.reference
        return {
            AnswerMode.SINGLE: f"In which direction is {target} relative to {reference} in every valid map?",
            AnswerMode.VISIBLE_POSSIBLE: f"Which listed directions could {target} occupy relative to {reference}? Select all that apply.",
            AnswerMode.ALL_POSSIBLE: f"Select the complete set of directions {target} could occupy relative to {reference}.",
        }[mode]
    direction = next(iter(query.directions)).value.lower()
    relation = f"{direction} of {query.reference}"
    if isinstance(query, WhichQuery):
        return {
            AnswerMode.SINGLE: f"Which eligible object is the only one {relation} in every valid map?",
            AnswerMode.VISIBLE_POSSIBLE: f"Which listed eligible objects could be {relation}? Select all that apply.",
            AnswerMode.ALL_POSSIBLE: f"Select the complete set of eligible objects that could be {relation}.",
        }[mode]
    return {
        AnswerMode.SINGLE: f"What exact number of eligible objects are {relation} in every valid map?",
        AnswerMode.VISIBLE_POSSIBLE: f"Which listed counts are possible for the number of eligible objects {relation}? Select all that apply.",
        AnswerMode.ALL_POSSIBLE: f"Select the complete set of possible counts of eligible objects {relation}.",
    }[mode]


def render_prompt(
    problem: SpatialProblem, menu: tuple[Value, ...], mode: AnswerMode
) -> str:
    mode = AnswerMode(mode)
    validate_problem(problem)
    validate_menu(problem, menu)
    query = problem.query
    if isinstance(query, DirectionQuery):
        query_record = f"direction {query.target} {query.reference}"
    else:
        kind = "which" if isinstance(query, WhichQuery) else "count"
        query_record = f"{kind} {next(iter(query.directions)).value} {query.reference}"
    lines = [
        f"Interface: {VERSION}",
        f"Contract: {mode.value}",
        _ASSUMPTIONS,
        _RULES[mode],
        (
            "Eligible objects are independent of the menu. Selection possibilities "
            "need not hold simultaneously; Count possibilities must hold jointly in "
            "one valid map. Zero is an ordinary Count value."
        ),
        (
            "End with Answer: <letters>. Use ascending uppercase letters without spaces "
            "or punctuation. Unique requires one letter; other contracts allow a "
            "nonempty subset of A-D. E and F must each appear alone."
        ),
        "Objects: " + ", ".join(problem.objects),
        "Query: " + query_record,
    ]
    if not isinstance(query, DirectionQuery):
        lines.append("Eligible: " + ", ".join(query.candidates))
    lines.append("Consider a map with multiple objects:")
    lines.extend(
        f"{atom.subject} is to the {next(iter(atom.allowed)).value} of {atom.reference}."
        for atom in problem.premise.operands
    )
    lines.append("Question: " + _question(problem, mode))
    lines.extend(f"{letter}. {value}" for letter, value in zip("ABCD", menu))
    lines.extend(
        f"{letter}. {label}" for letter, label in zip("EF", escape_labels(mode))
    )
    return "\n".join(lines)


def parse_prompt(text: str) -> tuple[SpatialProblem, tuple[Value, ...], AnswerMode]:
    """Parse only this version's complete canonical template, including E/F rules."""
    lines = text.splitlines()

    def field(prefix):
        matches = [line[len(prefix) :] for line in lines if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"expected one {prefix} field")
        return matches[0]

    mode = AnswerMode(field("Contract: "))
    objects = tuple(field("Objects: ").split(", "))
    kind, target, reference = field("Query: ").split()
    if kind == "direction":
        query = DirectionQuery(target, reference)
    elif kind in {"which", "count"}:
        cls = WhichQuery if kind == "which" else CountQuery
        query = cls(
            frozenset({Direction(target)}),
            reference,
            tuple(field("Eligible: ").split(", ")),
        )
    else:
        raise ValueError("unsupported query family")
    start = lines.index("Consider a map with multiple objects:") + 1
    end = next(i for i, line in enumerate(lines) if line.startswith("Question: "))
    atoms = []
    for line in lines[start:end]:
        match = re.fullmatch(r"(\w+) is to the (\w+) of (\w+)\.", line)
        if match is None:
            raise ValueError("invalid atomic sentence")
        subject, direction, reference = match.groups()
        atoms.append(
            RelationConstraint(subject, reference, frozenset({Direction(direction)}))
        )
    problem = SpatialProblem(objects, And(tuple(atoms)), query)
    menu = tuple(field(f"{letter}. ") for letter in "ABCD")
    if isinstance(query, CountQuery):
        menu = tuple(int(value) for value in menu)
    if render_prompt(problem, menu, mode) != text:
        raise ValueError("prompt does not match the full composition-v1 template")
    return problem, menu, mode
