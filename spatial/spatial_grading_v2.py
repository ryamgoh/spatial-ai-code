"""Resolve semantic answers, encode menus, and score V2 responses."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum

from spatial_solver_v2 import (
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


class AnswerSemantics(str, Enum):
    """Meaning of the requested answer, independent of its menu encoding."""

    EXACT = "exact"
    ALL_POSSIBLE = "all-possible"


class SelectionMode(str, Enum):
    """Number of ordinary menu options that a response may select."""

    SINGLE_SELECT = "single-select"
    MULTI_SELECT = "multi-select"


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
    semantics: AnswerSemantics
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
    selection_mode: SelectionMode
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


def _exact_values(analysis: QueryAnalysis) -> tuple[AnswerValue, ...] | None:
    possible = _possible_values(analysis)
    if isinstance(analysis, WhichAnalysis):
        return (
            analysis.entailed_entities
            if set(possible) == set(analysis.entailed_entities)
            else None
        )
    return possible if len(possible) == 1 else None


def resolve_answer(
    analysis: QueryAnalysis,
    semantics: AnswerSemantics | str,
) -> AnswerResolution:
    """Resolve a query without consulting option labels or menu cardinality."""
    semantics = AnswerSemantics(semantics)
    if analysis.error:
        return AnswerResolution(
            analysis,
            semantics,
            ResolutionStatus.ERROR,
            error=analysis.error,
        )
    if not analysis.consistent:
        return AnswerResolution(
            analysis,
            semantics,
            ResolutionStatus.INCONSISTENT,
            error="premises are inconsistent",
        )

    possible = _possible_values(analysis)
    exact = _exact_values(analysis)
    if semantics is AnswerSemantics.EXACT:
        if exact is not None:
            return AnswerResolution(
                analysis,
                semantics,
                ResolutionStatus.EXACT,
                exact,
            )
        if not possible:
            return AnswerResolution(
                analysis,
                semantics,
                ResolutionStatus.NO_MATCH,
            )
        return AnswerResolution(
            analysis,
            semantics,
            ResolutionStatus.AMBIGUOUS,
        )

    if not possible:
        return AnswerResolution(
            analysis,
            semantics,
            ResolutionStatus.NO_MATCH,
        )
    return AnswerResolution(
        analysis,
        semantics,
        ResolutionStatus.EXACT if exact is not None else ResolutionStatus.POSSIBILITIES,
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
    selection_mode: SelectionMode,
    options: Mapping[str, str],
    status: str,
    error: str,
) -> MenuAnswer:
    return MenuAnswer(
        resolution,
        selection_mode,
        status=status,
        options=dict(options),
        error=error,
    )


def encode_menu_answer(
    resolution: AnswerResolution,
    options: Mapping[str, str],
    selection_mode: SelectionMode | str,
) -> MenuAnswer:
    """Encode a resolved semantic answer without silently dropping values."""
    selection_mode = SelectionMode(selection_mode)
    if not options:
        return _menu_error(
            resolution,
            selection_mode,
            options,
            "error",
            "no options found",
        )

    undetermined, none_of_options = _special_options(options)
    if resolution.status is ResolutionStatus.ERROR:
        return _menu_error(
            resolution,
            selection_mode,
            options,
            "error",
            resolution.error or "semantic resolution failed",
        )
    if resolution.status is ResolutionStatus.INCONSISTENT:
        return _menu_error(
            resolution,
            selection_mode,
            options,
            "inconsistent",
            resolution.error or "premises are inconsistent",
        )
    if resolution.status is ResolutionStatus.AMBIGUOUS:
        if not undetermined:
            return _menu_error(
                resolution,
                selection_mode,
                options,
                "ambiguous",
                "menu has no undetermined option",
            )
        return MenuAnswer(
            resolution,
            selection_mode,
            undetermined,
            "undetermined",
            dict(options),
        )

    if resolution.status is ResolutionStatus.NO_MATCH or not resolution.values:
        if not none_of_options:
            return _menu_error(
                resolution,
                selection_mode,
                options,
                "no-match",
                "menu has no none-of-options option",
            )
        return MenuAnswer(
            resolution,
            selection_mode,
            none_of_options,
            "none-of-options",
            dict(options),
        )

    listed = _listed_values(resolution.analysis, options)
    selected = {listed[value] for value in resolution.values if value in listed}
    missing = tuple(value for value in resolution.values if value not in listed)
    if missing:
        if (
            resolution.semantics is AnswerSemantics.EXACT
            and len(resolution.values) == 1
            and not selected
            and none_of_options
        ):
            return MenuAnswer(
                resolution,
                selection_mode,
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
            selection_mode,
            options,
            "incomplete-menu",
            f"menu is missing possible answer: {rendered}",
        )
    if selection_mode is SelectionMode.SINGLE_SELECT and len(selected) != 1:
        return _menu_error(
            resolution,
            selection_mode,
            options,
            "selection-mismatch",
            f"single-select menu cannot encode {len(selected)} answers",
        )
    return MenuAnswer(
        resolution,
        selection_mode,
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
        expected.selection_mode is SelectionMode.MULTI_SELECT or len(predicted) <= 1,
        predicted == gold,
        precision,
        recall,
        f1,
        overlap / len(union) if union else 1.0,
    )
