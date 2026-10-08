"""Resolve semantic answers, encode menus, and score V2 responses."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum

from spatial.v2.solver import (
    Direction,
    DirectionAnalysis,
    QueryAnalysis,
    WhichAnalysis,
)

AnswerValue = Direction | str | int

_UNDETERMINED = {
    "cannot be determined",
    "undetermined",
    "not enough information",
}
_NONE_OF_OPTIONS = {
    "none of the options",
    "none of these options",
    "none of the given options",
    "none of the above",
}


class AnswerMode(str, Enum):
    """Complete answer contract for a question and its option menu."""

    SINGLE = "single"
    ALL_POSSIBLE = "all-possible"
    VISIBLE_POSSIBLE = "visible-possible"


class ResolutionStatus(str, Enum):
    EXACT = "exact"
    POSSIBILITIES = "possibilities"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no-match"
    INCONSISTENT = "inconsistent"
    ERROR = "error"


@dataclass(frozen=True)
class AnswerResolution:
    analysis: QueryAnalysis
    mode: AnswerMode
    status: ResolutionStatus
    values: tuple[AnswerValue, ...] = ()
    error: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.error is None and self.status in {
            ResolutionStatus.EXACT,
            ResolutionStatus.POSSIBILITIES,
            ResolutionStatus.NO_MATCH,
        }


@dataclass(frozen=True)
class MenuAnswer:
    resolution: AnswerResolution
    letters: frozenset[str] = frozenset()
    status: str = ""
    options: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def raw(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        return (
            ",".join(sorted(self.letters)) if self.letters else "No valid options found"
        )

    @property
    def is_resolved(self) -> bool:
        return self.error is None and bool(self.letters)


@dataclass(frozen=True)
class ResponseScore:
    predicted_letters: frozenset[str]
    expected_letters: frozenset[str]
    selection_valid: bool
    exact_match: bool
    precision: float
    recall: float
    f1: float
    jaccard: float


def _possible_values(analysis: QueryAnalysis) -> tuple[AnswerValue, ...]:
    if isinstance(analysis, DirectionAnalysis):
        return analysis.possible_directions
    if isinstance(analysis, WhichAnalysis):
        return analysis.possible_entities
    return analysis.possible_counts


def _single_value(analysis: QueryAnalysis) -> tuple[AnswerValue, ...] | None:
    possible = _possible_values(analysis)
    if isinstance(analysis, WhichAnalysis):
        return (
            analysis.entailed_entities
            if len(analysis.entailed_entities) == 1
            and set(possible) == set(analysis.entailed_entities)
            else None
        )
    return possible if len(possible) == 1 else None


def resolve_answer(
    analysis: QueryAnalysis,
    mode: AnswerMode | str,
) -> AnswerResolution:
    """Resolve a query without consulting option labels or menu cardinality."""
    mode = AnswerMode(mode)
    if analysis.error:
        return AnswerResolution(
            analysis,
            mode,
            ResolutionStatus.ERROR,
            error=analysis.error,
        )
    if not analysis.consistent:
        return AnswerResolution(
            analysis,
            mode,
            ResolutionStatus.INCONSISTENT,
            error="premises are inconsistent",
        )

    possible = _possible_values(analysis)
    if not possible:
        return AnswerResolution(
            analysis,
            mode,
            ResolutionStatus.NO_MATCH,
        )
    if mode is AnswerMode.SINGLE:
        single = _single_value(analysis)
        if single is not None:
            return AnswerResolution(
                analysis,
                mode,
                ResolutionStatus.EXACT,
                single,
            )
        return AnswerResolution(
            analysis,
            mode,
            ResolutionStatus.AMBIGUOUS,
        )

    return AnswerResolution(
        analysis,
        mode,
        ResolutionStatus.POSSIBILITIES,
        possible,
    )


def _normalized_option(value: str) -> str:
    normalized = value.strip().rstrip(".")
    return (
        normalized[4:].strip() if normalized.lower().startswith("the ") else normalized
    )


def _special_options(
    options: Mapping[str, str],
) -> tuple[frozenset[str], frozenset[str]]:
    undetermined = frozenset(
        key
        for key, value in options.items()
        if _normalized_option(value).lower() in _UNDETERMINED
    )
    none_of_options = frozenset(
        key
        for key, value in options.items()
        if _normalized_option(value).lower() in _NONE_OF_OPTIONS
    )
    return undetermined, none_of_options


def _listed_values(
    analysis: QueryAnalysis,
    options: Mapping[str, str],
) -> dict[AnswerValue, str]:
    listed: dict[AnswerValue, str] = {}
    for letter, raw_value in options.items():
        value = _normalized_option(raw_value)
        if isinstance(analysis, DirectionAnalysis):
            parsed: AnswerValue | None = Direction.from_label(value)
        elif isinstance(analysis, WhichAnalysis):
            parsed = next(
                (
                    candidate
                    for candidate in analysis.candidates
                    if candidate.lower() == value.lower()
                ),
                None,
            )
        else:
            try:
                parsed = int(value)
            except ValueError:
                parsed = None
        if parsed is not None:
            listed[parsed] = letter
    return listed


def _menu_error(
    resolution: AnswerResolution,
    options: Mapping[str, str],
    status: str,
    error: str,
) -> MenuAnswer:
    return MenuAnswer(
        resolution,
        status=status,
        options=dict(options),
        error=error,
    )


def _none_visible_answer(
    resolution: AnswerResolution,
    options: Mapping[str, str],
    none_of_options: frozenset[str],
) -> MenuAnswer:
    if none_of_options:
        return MenuAnswer(
            resolution,
            none_of_options,
            "none-of-options",
            dict(options),
        )
    return _menu_error(
        resolution,
        options,
        "no-match",
        "no possible answer is visible and the menu has no none-of-options option",
    )


def encode_menu_answer(
    resolution: AnswerResolution,
    options: Mapping[str, str],
) -> MenuAnswer:
    """Encode a resolved semantic answer without silently dropping values."""
    if not options:
        return _menu_error(
            resolution,
            options,
            "error",
            "no options found",
        )

    undetermined, none_of_options = _special_options(options)
    if resolution.status is ResolutionStatus.ERROR:
        return _menu_error(
            resolution,
            options,
            "error",
            resolution.error or "semantic resolution failed",
        )
    if resolution.status is ResolutionStatus.INCONSISTENT:
        return _menu_error(
            resolution,
            options,
            "inconsistent",
            resolution.error or "premises are inconsistent",
        )
    if resolution.status is ResolutionStatus.AMBIGUOUS:
        if not undetermined:
            return _menu_error(
                resolution,
                options,
                "ambiguous",
                "menu has no undetermined option",
            )
        return MenuAnswer(
            resolution,
            undetermined,
            "undetermined",
            dict(options),
        )

    if resolution.status is ResolutionStatus.NO_MATCH or not resolution.values:
        if not none_of_options:
            return _menu_error(
                resolution,
                options,
                "no-match",
                "menu has no none-of-options option",
            )
        return MenuAnswer(
            resolution,
            none_of_options,
            "none-of-options",
            dict(options),
        )

    listed = _listed_values(resolution.analysis, options)
    selected = {listed[value] for value in resolution.values if value in listed}
    missing = tuple(value for value in resolution.values if value not in listed)
    if resolution.mode is AnswerMode.VISIBLE_POSSIBLE:
        if not selected:
            return _none_visible_answer(resolution, options, none_of_options)
    elif resolution.mode is AnswerMode.ALL_POSSIBLE and missing:
        if not selected:
            return _none_visible_answer(resolution, options, none_of_options)
        if undetermined:
            return MenuAnswer(
                resolution,
                undetermined,
                "incomplete-menu",
                dict(options),
            )
        rendered = ", ".join(
            value.value if isinstance(value, Direction) else str(value)
            for value in missing
        )
        return _menu_error(
            resolution,
            options,
            "incomplete-menu",
            f"menu is missing possible answer: {rendered}",
        )
    elif missing:
        if resolution.mode is AnswerMode.SINGLE and not selected and none_of_options:
            return MenuAnswer(
                resolution,
                none_of_options,
                "none-of-options",
                dict(options),
            )
        rendered = ", ".join(
            value.value if isinstance(value, Direction) else str(value)
            for value in missing
        )
        return _menu_error(
            resolution,
            options,
            "incomplete-menu",
            f"menu is missing possible answer: {rendered}",
        )
    return MenuAnswer(
        resolution,
        frozenset(selected),
        resolution.status.value,
        dict(options),
    )


def _response_letters(response: str | Collection[str]) -> frozenset[str]:
    if isinstance(response, str):
        answer = response.rsplit("Answer:", 1)[-1]
        return frozenset(re.findall(r"\b[A-Z]\b", answer.upper()))
    return frozenset(str(value).strip().upper() for value in response)


def score_response(
    response: str | Collection[str],
    expected: MenuAnswer,
) -> ResponseScore:
    """Score selected option letters; exact-set equality is authoritative."""
    if not expected.is_resolved:
        raise ValueError("cannot score against an unresolved menu answer")
    predicted = _response_letters(response) & frozenset(expected.options)
    gold = expected.letters
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = predicted | gold
    return ResponseScore(
        predicted,
        gold,
        expected.resolution.mode is not AnswerMode.SINGLE or len(predicted) <= 1,
        predicted == gold,
        precision,
        recall,
        f1,
        overlap / len(union) if union else 1.0,
    )
