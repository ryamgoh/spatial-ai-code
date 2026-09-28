"""Option grading policies for V2 spatial analyses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from spatial_solver_v2 import (
    CountAnalysis,
    Direction,
    QueryAnalysis,
    WhichAnalysis,
)

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


class AnswerPolicy(str, Enum):
    POSSIBILITY_SET = "possibility-set"
    SINGLE_EXACT = "single-exact"


@dataclass
class Grade:
    raw: str
    analysis: QueryAnalysis
    answer_policy: AnswerPolicy = AnswerPolicy.POSSIBILITY_SET
    status: str = ""
    options: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def letters(self) -> set[str]:
        if self.error or self.raw.startswith("Error"):
            return set()
        return {
            token
            for token in re.split(r"[,;| ]+", self.raw)
            if len(token) == 1 and token.isalpha() and token.isupper()
        }

    @property
    def accept(self) -> bool:
        return self.error is None and bool(self.letters)


def _entity_name(value: str) -> str:
    normalized = value.strip().rstrip(".")
    return (
        normalized[4:].strip() if normalized.lower().startswith("the ") else normalized
    )


def grade(
    analysis: QueryAnalysis,
    options: dict[str, str],
    answer_policy: AnswerPolicy | str = AnswerPolicy.POSSIBILITY_SET,
) -> Grade:
    answer_policy = AnswerPolicy(answer_policy)
    if analysis.error:
        return Grade(
            raw=f"Error: {analysis.error}",
            analysis=analysis,
            answer_policy=answer_policy,
            status="error",
            error=analysis.error,
        )
    if not options:
        return Grade(
            raw="Error: no options found",
            analysis=analysis,
            answer_policy=answer_policy,
            status="error",
            error="no options found",
        )
    if isinstance(analysis, WhichAnalysis):
        return _grade_which(analysis, options, answer_policy)
    if isinstance(analysis, CountAnalysis):
        return _grade_count(analysis, options, answer_policy)

    listed = {
        direction: key
        for key, value in options.items()
        if (direction := Direction.from_label(value)) is not None
    }
    undetermined = {
        key
        for key, value in options.items()
        if value.strip().rstrip(".").lower() in _UNDETERMINED
    }
    none_of_options = {
        key
        for key, value in options.items()
        if value.strip().rstrip(".").lower() in _NONE_OF_OPTIONS
    }

    possible = set(analysis.possible_directions)
    if not analysis.consistent:
        valid = undetermined
        status = "inconsistent"
    elif len(possible) == 1:
        direction = next(iter(possible))
        if direction in listed:
            valid = {listed[direction]}
            status = "entailed"
        else:
            valid = none_of_options
            status = "missing-option"
    elif answer_policy is AnswerPolicy.SINGLE_EXACT:
        valid = undetermined
        status = "ambiguous"
    elif possible and possible <= set(listed):
        valid = {listed[direction] for direction in possible}
        status = "possibility-set"
    else:
        valid = undetermined
        status = "incomplete-menu"

    raw = ",".join(sorted(valid)) if valid else "No valid options found"
    return Grade(
        raw=raw,
        analysis=analysis,
        answer_policy=answer_policy,
        status=status,
        options=options,
    )


def _special_options(options: dict[str, str]) -> tuple[set[str], set[str]]:
    undetermined = {
        key
        for key, value in options.items()
        if value.strip().rstrip(".").lower() in _UNDETERMINED
    }
    none_of_options = {
        key
        for key, value in options.items()
        if value.strip().rstrip(".").lower() in _NONE_OF_OPTIONS
    }
    return undetermined, none_of_options


def _grade_which(
    analysis: WhichAnalysis,
    options: dict[str, str],
    answer_policy: AnswerPolicy,
) -> Grade:
    undetermined, none_of_options = _special_options(options)
    listed = {_entity_name(value): key for key, value in options.items()}
    possible = set(analysis.possible_entities)
    entailed = set(analysis.entailed_entities)
    if not analysis.consistent:
        valid, status = undetermined, "inconsistent"
    elif answer_policy is AnswerPolicy.SINGLE_EXACT and possible != entailed:
        valid, status = undetermined, "ambiguous"
    else:
        selected = (
            possible if answer_policy is AnswerPolicy.POSSIBILITY_SET else entailed
        )
        if not selected:
            valid, status = none_of_options, "entailed"
        elif selected <= set(listed):
            valid = {listed[entity] for entity in selected}
            status = (
                "possibility-set"
                if answer_policy is AnswerPolicy.POSSIBILITY_SET
                and possible != entailed
                else "entailed"
            )
        else:
            valid, status = undetermined, "incomplete-menu"
    raw = ",".join(sorted(valid)) if valid else "No valid options found"
    return Grade(raw, analysis, answer_policy, status, options)


def _grade_count(
    analysis: CountAnalysis,
    options: dict[str, str],
    answer_policy: AnswerPolicy,
) -> Grade:
    undetermined, none_of_options = _special_options(options)
    listed: dict[int, str] = {}
    for key, value in options.items():
        try:
            listed[int(value.strip())] = key
        except ValueError:
            pass
    possible = set(analysis.possible_counts)
    if not analysis.consistent:
        valid, status = undetermined, "inconsistent"
    elif len(possible) == 1:
        count = next(iter(possible))
        valid = {listed[count]} if count in listed else none_of_options
        status = "entailed" if count in listed else "missing-option"
    elif answer_policy is AnswerPolicy.SINGLE_EXACT:
        valid, status = undetermined, "ambiguous"
    elif possible <= set(listed):
        valid = {listed[count] for count in possible}
        status = "possibility-set"
    else:
        valid, status = undetermined, "incomplete-menu"
    raw = ",".join(sorted(valid)) if valid else "No valid options found"
    return Grade(raw, analysis, answer_policy, status, options)
