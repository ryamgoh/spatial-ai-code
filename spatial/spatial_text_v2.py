"""Text adapter for the V2 spatial reasoning module."""

from __future__ import annotations

import re
from dataclasses import dataclass

from spatial_solver_v2 import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    RelationConstraint,
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
_PREMISE_RE = re.compile(
    rf"(.+?) is (?P<negative>not )?(?:to the )?"
    rf"(?P<relation>{_RELATION_PATTERN}) of (.+?)\.",
    re.IGNORECASE,
)
_STANDALONE_RE = re.compile(r"(.+?) is in the map\.", re.IGNORECASE)
_DIRECTION_QUERY_RE = re.compile(
    r"In which direction is (.+?) relative to (.+?)\?", re.IGNORECASE
)
_WHICH_QUERY_RE = re.compile(
    rf"Which objects? (?:is|are) in the ({_RELATION_PATTERN}) of (.+?)\?",
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


class SpatialTextAdapter:
    """Parse one rendered direction question into a structured problem."""

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
        constraints: list[RelationConstraint] = []
        position = 0
        for sentence in re.finditer(r"[^.]+\.", map_part):
            if map_part[position : sentence.start()].strip():
                raise ValueError("unparsed text between premises")
            raw_sentence = sentence.group(0).strip()
            relation_match = _PREMISE_RE.fullmatch(raw_sentence)
            standalone_match = _STANDALONE_RE.fullmatch(raw_sentence)
            if relation_match:
                subject = _strip_the(relation_match.group(1))
                reference = _strip_the(relation_match.group(4))
                if not subject or not reference or subject == reference:
                    raise ValueError("a relation requires two distinct named objects")
                allowed = _relation_set(relation_match.group("relation"))
                if relation_match.group("negative"):
                    allowed = _ALL_DIRECTIONS - allowed
                constraints.append(RelationConstraint(subject, reference, allowed))
                objects.update((subject, reference))
            elif standalone_match:
                objects.add(_strip_the(standalone_match.group(1)))
            else:
                raise ValueError(f"cannot parse premise: {raw_sentence}")
            position = sentence.end()
        if map_part[position:].strip():
            raise ValueError("unterminated or unparsed premise")
        if not constraints:
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
            reference = _strip_the(which_query.group(2))
            candidates = tuple(
                candidate
                for value in options.values()
                if (candidate := _strip_the(value)) in objects
                and candidate != reference
            )
            query = WhichQuery(
                directions=_relation_set(which_query.group(1)),
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
            premise=And(tuple(constraints)),
            query=query,
        )
        return ParsedSpatialQuestion(problem, options)
