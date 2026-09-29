"""Adapter and audit classification for original SpatialEval TQA rows."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spatial_grading_v2 import (
    AnswerMode,
    MenuAnswer,
    ResolutionStatus,
    encode_menu_answer,
    resolve_answer,
)
from spatial_solver_v2 import (
    And,
    Direction,
    DirectionAnalysis,
    DirectionQuery,
    QueryAnalysis,
    SpatialProblem,
    WhichAnalysis,
    WhichQuery,
    conjunctive_atoms,
)
from spatial_text_v2 import SpatialTextAdapter

ORDINAL_DIRECTIONS = frozenset(
    {
        Direction.NORTHEAST,
        Direction.NORTHWEST,
        Direction.SOUTHEAST,
        Direction.SOUTHWEST,
    }
)


@dataclass(frozen=True)
class SpatialEvalCase:
    row_id: str
    problem: SpatialProblem
    options: dict[str, str]
    oracle_answers: frozenset[Direction | str | int]
    answer_mode: AnswerMode


@dataclass(frozen=True)
class SpatialEvalAudit:
    case: SpatialEvalCase
    analysis: QueryAnalysis
    menu_answer: MenuAnswer
    status: str


class SpatialEvalAdapter:
    """Convert one SpatialEval row into the shared structured problem model."""

    def __init__(self, text_adapter: SpatialTextAdapter | None = None) -> None:
        self.text_adapter = text_adapter or SpatialTextAdapter()

    @staticmethod
    def _oracle_answers(
        row: Mapping[str, Any], options: Mapping[str, str], query
    ) -> frozenset[Direction | str | int]:
        values = [
            options[key]
            for key in re.findall(r"[A-Z]", str(row.get("oracle_option") or "").upper())
            if key in options
        ]
        if not values:
            values = [
                value.strip()
                for value in re.split(r"[,;|]", str(row.get("oracle_answer") or ""))
                if value.strip()
            ]
        if isinstance(query, DirectionQuery):
            answers = {
                direction
                for value in values
                if (direction := Direction.from_label(value)) is not None
            }
        elif isinstance(query, WhichQuery):
            candidates = {
                candidate.lower(): candidate for candidate in query.candidates
            }
            answers = set()
            for value in values:
                normalized = value.strip().rstrip(".")
                if normalized.lower().startswith("the "):
                    normalized = normalized[4:].strip()
                if normalized.lower() in candidates:
                    answers.add(candidates[normalized.lower()])
        else:
            answers = set()
            for value in values:
                try:
                    answers.add(int(value))
                except ValueError:
                    continue
        if not answers:
            raise ValueError("SpatialEval row has no parseable oracle answer")
        return frozenset(answers)

    @staticmethod
    def _validate_direction_options(options: Mapping[str, str]) -> None:
        directions: set[Direction] = set()
        for value in options.values():
            if direction := Direction.from_label(value):
                directions.add(direction)
        if not directions or not directions <= ORDINAL_DIRECTIONS:
            raise ValueError("SpatialEval options must use ordinal directions")

    def parse(
        self,
        row: Mapping[str, Any],
        answer_mode: AnswerMode | str = AnswerMode.SINGLE,
    ) -> SpatialEvalCase:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("SpatialEval row has no text")
        parsed = self.text_adapter.parse(text)
        parsed_constraints = conjunctive_atoms(parsed.problem.premise)
        if parsed_constraints is None:
            raise ValueError("SpatialEval premises must be a conjunction")
        if any(
            len(constraint.allowed) != 1 or not constraint.allowed <= ORDINAL_DIRECTIONS
            for constraint in parsed_constraints
        ):
            raise ValueError("SpatialEval premises must be exact ordinal directions")
        objects = parsed.problem.objects
        parsed_query = parsed.problem.query
        if isinstance(parsed_query, DirectionQuery):
            self._validate_direction_options(parsed.options)
            query = DirectionQuery(
                parsed_query.target,
                parsed_query.reference,
                ORDINAL_DIRECTIONS,
            )
        else:
            query = parsed_query
        problem = SpatialProblem(
            objects=objects,
            premise=And(parsed_constraints),
            query=query,
        )
        return SpatialEvalCase(
            row_id=str(row.get("id") or ""),
            problem=problem,
            options=parsed.options,
            oracle_answers=self._oracle_answers(row, parsed.options, query),
            answer_mode=AnswerMode(answer_mode),
        )


def audit(case: SpatialEvalCase, analysis: QueryAnalysis) -> SpatialEvalAudit:
    resolution = resolve_answer(analysis, case.answer_mode)
    menu_answer = encode_menu_answer(resolution, case.options)
    if isinstance(analysis, DirectionAnalysis):
        possible = set(analysis.possible_directions)
    elif isinstance(analysis, WhichAnalysis):
        possible = set(analysis.possible_entities)
    else:
        possible = set(analysis.possible_counts)
    oracle = set(case.oracle_answers)
    if analysis.error:
        status = "error"
    elif not analysis.consistent:
        status = "inconsistent"
    else:
        resolution = menu_answer.resolution
        expected = set(resolution.values)
        if resolution.status is ResolutionStatus.AMBIGUOUS:
            expected = possible
            if oracle <= expected:
                status = "underdetermined-oracle-possible"
            elif expected < oracle:
                status = "oracle-overinclusive"
            elif oracle.isdisjoint(expected):
                status = "oracle-contradicted"
            else:
                status = "partial-overlap"
        elif oracle == expected:
            status = "exact-match"
        elif oracle < expected:
            status = "oracle-underinclusive"
        elif expected < oracle:
            status = "oracle-overinclusive"
        elif oracle.isdisjoint(expected):
            status = "oracle-contradicted"
        else:
            status = "partial-overlap"
    return SpatialEvalAudit(case, analysis, menu_answer, status)
