"""Text adapter for the V2 spatial reasoning module."""

from __future__ import annotations

import re
from dataclasses import dataclass

from spatial.v2.solver import (
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
)

_COARSE_DIRECTIONS: dict[str, frozenset[Direction]] = {
    "Northward": frozenset({Direction.NORTHWEST, Direction.NORTH, Direction.NORTHEAST}),
    "Eastward": frozenset({Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST}),
    "Southward": frozenset({Direction.SOUTHEAST, Direction.SOUTH, Direction.SOUTHWEST}),
    "Westward": frozenset({Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST}),
}
_ALL_DIRECTIONS = frozenset(Direction)
_RELATION_WORDS = tuple(direction.value for direction in Direction) + tuple(
    _COARSE_DIRECTIONS
)
_RELATION_PATTERN = "|".join(sorted(_RELATION_WORDS, key=len, reverse=True))
_ATOM_RE = re.compile(
    rf"(.+?) is (?P<negative>not )?(?:to the )?"
    rf"(?P<relation>{_RELATION_PATTERN}) of (.+?)",
    re.IGNORECASE,
)
_STANDALONE_RE = re.compile(r"(.+?) is in the map", re.IGNORECASE)
_DIRECTION_QUERY_RE = re.compile(
    r"In which direction is (.+?) relative to (.+?)\?", re.IGNORECASE
)
_WHICH_QUERY_RE = re.compile(
    rf"Which objects?(?P<map_scope> in the map)? (?:is|are) in the "
    rf"(?P<relation>{_RELATION_PATTERN}) of (?P<reference>.+?)\?",
    re.IGNORECASE,
)
_COUNT_QUERY_RE = re.compile(
    rf"How many objects are in the ({_RELATION_PATTERN}) of (.+?)\?",
    re.IGNORECASE,
)
_OPTION_RE = re.compile(
    r"(?:^|,\s*|\n\s*)([A-Z])\.\s*(.+?)"
    r"(?=(?:,\s*|\n\s*)[A-Z]\.\s*|\s*$)",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedSpatialQuestion:
    problem: SpatialProblem
    options: dict[str, str]


def _normalize_text(text: str) -> str:
    return (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _strip_the(name: str) -> str:
    normalized = name.strip().strip(".")
    if normalized.lower().startswith("the "):
        return normalized[4:].strip()
    return normalized


def _relation_set(name: str) -> frozenset[Direction]:
    if direction := Direction.from_label(name):
        return frozenset({direction})
    canonical = name.strip().title()
    if canonical in _COARSE_DIRECTIONS:
        return _COARSE_DIRECTIONS[canonical]
    raise ValueError(f"unknown spatial relation: {name}")


def _strip_outer_parentheses(expression: str) -> str:
    expression = expression.strip()
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        encloses_all = True
        for index, character in enumerate(expression):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("unbalanced formula parentheses")
                if depth == 0 and index != len(expression) - 1:
                    encloses_all = False
                    break
        if depth != 0:
            raise ValueError("unbalanced formula parentheses")
        if not encloses_all:
            break
        expression = expression[1:-1].strip()
    return expression


def _split_top_level(expression: str, keyword: str) -> tuple[str, ...]:
    delimiter = f" {keyword} "
    upper = expression.upper()
    depth = 0
    start = 0
    parts = []
    index = 0
    while index < len(expression):
        character = expression[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced formula parentheses")
        elif depth == 0 and upper.startswith(delimiter, index):
            parts.append(expression[start:index].strip())
            index += len(delimiter)
            start = index
            continue
        index += 1
    if depth != 0:
        raise ValueError("unbalanced formula parentheses")
    if not parts:
        return (expression,)
    parts.append(expression[start:].strip())
    if any(not part for part in parts):
        raise ValueError(f"{keyword} requires formulas on both sides")
    return tuple(parts)


def _parse_formula(expression: str, objects: set[str]) -> SpatialFormula:
    expression = _strip_outer_parentheses(expression)
    upper = expression.upper()
    if upper.startswith("IF "):
        parts = _split_top_level(expression[3:].strip(), "THEN")
        if len(parts) != 2:
            raise ValueError("IF requires exactly one top-level THEN")
        return Implies(
            _parse_formula(parts[0], objects),
            _parse_formula(parts[1], objects),
        )
    for keyword, formula_type in (("IFF", Iff), ("OR", Or), ("AND", And)):
        parts = _split_top_level(expression, keyword)
        if len(parts) == 1:
            continue
        operands = tuple(_parse_formula(part, objects) for part in parts)
        if formula_type is Iff:
            if len(operands) != 2:
                raise ValueError("IFF requires exactly two operands")
            return Iff(*operands)
        return formula_type(operands)
    if upper.startswith("NOT "):
        return Not(_parse_formula(expression[4:].strip(), objects))

    match = _ATOM_RE.fullmatch(expression)
    if match is None:
        raise ValueError(f"cannot parse controlled formula: {expression}")
    subject = _strip_the(match.group(1))
    reference = _strip_the(match.group(4))
    if not subject or not reference or subject == reference:
        raise ValueError("a relation requires two distinct named objects")
    allowed = _relation_set(match.group("relation"))
    if match.group("negative"):
        allowed = _ALL_DIRECTIONS - allowed
    objects.update((subject, reference))
    return RelationConstraint(subject, reference, allowed)


def spatial_formula_objects(formula: SpatialFormula) -> frozenset[str]:
    """Return every object referenced by one structured formula."""
    if isinstance(formula, RelationConstraint):
        return frozenset((formula.subject, formula.reference))
    if isinstance(formula, Not):
        return spatial_formula_objects(formula.operand)
    if isinstance(formula, (And, Or)):
        return frozenset(
            obj
            for operand in formula.operands
            for obj in spatial_formula_objects(operand)
        )
    if isinstance(formula, Implies):
        return spatial_formula_objects(formula.antecedent) | spatial_formula_objects(
            formula.consequent
        )
    if isinstance(formula, Iff):
        return spatial_formula_objects(formula.left) | spatial_formula_objects(
            formula.right
        )
    raise TypeError(f"unsupported spatial formula: {type(formula).__name__}")


def _relation_phrase(constraint: RelationConstraint) -> str:
    if len(constraint.allowed) == 1:
        return "to the " + next(iter(constraint.allowed)).value
    for name, directions in _COARSE_DIRECTIONS.items():
        if constraint.allowed == directions:
            return "to the " + name
    excluded = _ALL_DIRECTIONS - constraint.allowed
    if len(excluded) == 1:
        return "not " + next(iter(excluded)).value
    raise ValueError("controlled text cannot render this direction set atomically")


def render_spatial_formula(formula: SpatialFormula) -> str:
    """Render one formula in the canonical controlled Boolean grammar."""
    if isinstance(formula, RelationConstraint):
        return (
            f"{formula.subject} is {_relation_phrase(formula)} of {formula.reference}"
        )
    if isinstance(formula, Not):
        return f"NOT ({render_spatial_formula(formula.operand)})"
    if isinstance(formula, (And, Or)):
        operator = " AND " if isinstance(formula, And) else " OR "
        return operator.join(
            f"({render_spatial_formula(operand)})" for operand in formula.operands
        )
    if isinstance(formula, Implies):
        return (
            f"IF ({render_spatial_formula(formula.antecedent)}) THEN "
            f"({render_spatial_formula(formula.consequent)})"
        )
    if isinstance(formula, Iff):
        return (
            f"({render_spatial_formula(formula.left)}) IFF "
            f"({render_spatial_formula(formula.right)})"
        )
    raise TypeError(f"unsupported spatial formula: {type(formula).__name__}")


class SpatialTextAdapter:
    """Parse one controlled spatial question into a structured problem."""

    @staticmethod
    def _split_prompt(text: str) -> tuple[str, str]:
        normalized = _normalize_text(text)
        markers = [
            index
            for marker in ("Please answer", "Question:")
            if (index := normalized.find(marker)) >= 0
        ]
        if not markers:
            raise ValueError("no question found")
        index = min(markers)
        return normalized[:index], normalized[index:]

    @staticmethod
    def _parse_options(question_part: str) -> dict[str, str]:
        tail = question_part.split("Available options", 1)
        if len(tail) != 2:
            return {}
        return {
            match.group(1): match.group(2).strip().rstrip(".,")
            for match in _OPTION_RE.finditer(tail[1].lstrip(" :\n"))
        }

    def parse(self, text: str) -> ParsedSpatialQuestion:
        map_part, question_part = self._split_prompt(text)
        map_part = re.sub(
            r"^\s*Consider a map with multiple (?:locations|objects):\s*",
            "",
            map_part,
            flags=re.IGNORECASE,
        ).strip()

        objects: set[str] = set()
        formulas: list[SpatialFormula] = []
        position = 0
        for sentence in re.finditer(r"[^.]+\.", map_part):
            if map_part[position : sentence.start()].strip():
                raise ValueError("unparsed text between premises")
            raw_sentence = sentence.group(0).strip()
            expression = raw_sentence[:-1].strip()
            standalone_match = _STANDALONE_RE.fullmatch(expression)
            if standalone_match:
                objects.add(_strip_the(standalone_match.group(1)))
            else:
                try:
                    formulas.append(_parse_formula(expression, objects))
                except ValueError as exc:
                    raise ValueError(f"cannot parse premise: {raw_sentence}") from exc
            position = sentence.end()
        if map_part[position:].strip():
            raise ValueError("unterminated or unparsed premise")
        if not formulas:
            raise ValueError("no spatial premises found")

        options = self._parse_options(question_part)
        direction_query = _DIRECTION_QUERY_RE.search(question_part)
        which_query = _WHICH_QUERY_RE.search(question_part)
        count_query = _COUNT_QUERY_RE.search(question_part)
        if direction_query:
            query = DirectionQuery(
                target=_strip_the(direction_query.group(1)),
                reference=_strip_the(direction_query.group(2)),
            )
        elif which_query:
            reference = _strip_the(which_query.group("reference"))
            if which_query.group("map_scope"):
                candidates = tuple(sorted(objects - {reference}))
            else:
                candidates = tuple(
                    candidate
                    for value in options.values()
                    if (candidate := _strip_the(value)) in objects
                    and candidate != reference
                )
            query = WhichQuery(
                directions=_relation_set(which_query.group("relation")),
                reference=reference,
                candidates=candidates,
            )
        elif count_query:
            reference = _strip_the(count_query.group(2))
            query = CountQuery(
                directions=_relation_set(count_query.group(1)),
                reference=reference,
                candidates=tuple(sorted(objects - {reference})),
            )
        else:
            raise ValueError("unsupported or malformed question")
        problem = SpatialProblem(
            objects=tuple(sorted(objects)),
            premise=And(tuple(formulas)),
            query=query,
        )
        return ParsedSpatialQuestion(problem, options)
