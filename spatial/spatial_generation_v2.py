"""Policy-driven, solver-validated SpatialMap V2 generation.

This module owns synthetic-data choices.  It constructs structured problems,
asks the data-agnostic solver for their complete semantics, builds a menu under
an explicit answer contract, and round-trips the rendered prompt before a row
may be emitted.  Coordinates are retained only as audit witnesses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from string import ascii_uppercase
from typing import Any

from spatial_explanation_renderers_v2 import (
    StateMode,
    TraceFormat,
    render_training_trace,
)
from spatial_explanations_v2 import (
    QueryExplanation,
    SpatialExplainerV2,
    explanation_to_dict,
)
from spatial_grading_v2 import (
    AnswerMode,
    AnswerResolution,
    MenuAnswer,
    ResolutionStatus,
    encode_menu_answer,
    resolve_answer,
)
from spatial_solver_v2 import (
    And,
    CountQuery,
    Direction,
    DirectionAnalysis,
    DirectionQuery,
    QueryAnalysis,
    RelationConstraint,
    SpatialProblem,
    SpatialSolverV2,
    WhichAnalysis,
    WhichQuery,
    direction_between,
)
from spatial_text_v2 import SpatialTextAdapter

ENTITY_NAMES = (
    "Bakery",
    "Bank",
    "Bookstore",
    "Cafe",
    "Cinema",
    "City Hall",
    "Clinic",
    "Fire Station",
    "Florist",
    "Gallery",
    "Gas Station",
    "Grocery Store",
    "Gym",
    "High School",
    "Hospital",
    "Hotel",
    "Library",
    "Museum",
    "Park",
    "Pharmacy",
    "Police Station",
    "Post Office",
    "Restaurant",
    "Stadium",
    "Supermarket",
    "Theater",
    "Train Station",
    "University",
    "Veterinary Clinic",
    "Zoo",
)


class QueryKind(str, Enum):
    DIRECTION = "direction"
    WHICH = "which"
    COUNT = "count"


class SemanticShape(str, Enum):
    ANY = "any"
    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no-match"


class MenuCoverage(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    ZERO = "zero"


@dataclass(frozen=True)
class GenerationPolicy:
    """Dataset policy; none of these controls belong to the solver."""

    query_kind: QueryKind = QueryKind.DIRECTION
    answer_mode: AnswerMode = AnswerMode.SINGLE
    semantic_shape: SemanticShape = SemanticShape.ANY
    menu_coverage: MenuCoverage = MenuCoverage.FULL
    trace_format: TraceFormat = TraceFormat.NATURAL
    state_mode: StateMode = StateMode.DELTA
    num_entities: int = 6
    num_premises: int = 7
    ordinary_option_target: int = 4
    query_direction: Direction | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_kind", QueryKind(self.query_kind))
        object.__setattr__(self, "answer_mode", AnswerMode(self.answer_mode))
        object.__setattr__(self, "semantic_shape", SemanticShape(self.semantic_shape))
        object.__setattr__(self, "menu_coverage", MenuCoverage(self.menu_coverage))
        object.__setattr__(self, "trace_format", TraceFormat(self.trace_format))
        object.__setattr__(self, "state_mode", StateMode(self.state_mode))
        if self.query_direction is not None:
            object.__setattr__(self, "query_direction", Direction(self.query_direction))
        if not 2 <= self.num_entities <= len(ENTITY_NAMES):
            raise ValueError(f"num_entities must be between 2 and {len(ENTITY_NAMES)}")
        maximum_pairs = self.num_entities * (self.num_entities - 1) // 2
        if not self.num_entities - 1 <= self.num_premises <= maximum_pairs:
            raise ValueError(
                "num_premises must connect every entity and cannot exceed all pairs"
            )
        if not 1 <= self.ordinary_option_target <= len(ascii_uppercase) - 1:
            raise ValueError("ordinary_option_target must be between 1 and 25")
        if self.query_kind is QueryKind.DIRECTION and self.query_direction is not None:
            raise ValueError("query_direction applies only to Which and Count queries")
        if (
            self.answer_mode is AnswerMode.SINGLE
            and self.menu_coverage is MenuCoverage.PARTIAL
        ):
            raise ValueError("partial menu coverage is undefined for SINGLE questions")
        if self.semantic_shape is SemanticShape.NO_MATCH and self.query_kind in {
            QueryKind.DIRECTION,
            QueryKind.COUNT,
        }:
            raise ValueError("no-match is only realizable for Which queries")


def _answer_values(analysis: QueryAnalysis) -> tuple[Direction | str | int, ...]:
    if isinstance(analysis, DirectionAnalysis):
        return analysis.possible_directions
    if isinstance(analysis, WhichAnalysis):
        return analysis.possible_entities
    return analysis.possible_counts


def _render_value(value: Direction | str | int) -> str:
    return value.value if isinstance(value, Direction) else str(value)


def _coordinate_payload(
    coordinates: dict[str, tuple[int, int]],
) -> dict[str, list[int]]:
    return {name: [point[0], point[1]] for name, point in sorted(coordinates.items())}


@dataclass(frozen=True)
class GeneratedSpatialSample:
    problem: SpatialProblem
    options: dict[str, str]
    analysis: QueryAnalysis
    resolution: AnswerResolution
    menu_answer: MenuAnswer
    explanation: QueryExplanation
    prompt: str
    trace: str
    policy: GenerationPolicy
    seed: int
    sample_index: int
    attempt: int
    generation_witness: dict[str, tuple[int, int]]

    @property
    def audit_metadata(self) -> dict[str, Any]:
        return {
            "generation_witness": _coordinate_payload(self.generation_witness),
            "explanation": explanation_to_dict(self.explanation),
        }

    def as_sft_row(self, *, include_audit: bool = False) -> dict[str, Any]:
        answer = ", ".join(sorted(self.menu_answer.letters))
        query_direction = (
            next(iter(self.problem.query.directions)).value
            if isinstance(self.problem.query, (WhichQuery, CountQuery))
            and len(self.problem.query.directions) == 1
            else None
        )
        row: dict[str, Any] = {
            "id": f"spatial-v2-{self.seed}-{self.sample_index}-{self.attempt}",
            "messages": [
                {"role": "system", "content": _system_prompt(self.policy.answer_mode)},
                {"role": "user", "content": self.prompt},
                {
                    "role": "assistant",
                    "content": f"<think>\n{self.trace}\n</think>\nAnswer: {answer}",
                },
            ],
            "metadata": {
                "schema": "spatial-v2",
                "seed": self.seed,
                "sample_index": self.sample_index,
                "attempt": self.attempt,
                "query_kind": self.policy.query_kind.value,
                "query_direction": query_direction,
                "answer_mode": self.policy.answer_mode.value,
                "semantic_shape": self.policy.semantic_shape.value,
                "menu_coverage": self.policy.menu_coverage.value,
                "trace_format": self.policy.trace_format.value,
                "state_mode": self.policy.state_mode.value,
                "num_entities": len(self.problem.objects),
                "num_premises": len(self.problem.premise.operands),
                "possible_values": [
                    _render_value(value) for value in _answer_values(self.analysis)
                ],
                "resolution_status": self.resolution.status.value,
                "menu_status": self.menu_answer.status,
                "oracle_letters": sorted(self.menu_answer.letters),
                "solver_engine": self.analysis.engine,
                "round_trip_verified": True,
            },
        }
        if include_audit:
            row["audit"] = self.audit_metadata
        return row


class _RetryGeneration(Exception):
    pass


class SpatialGeneratorV2:
    """Generate accepted samples only after semantic and text round-trip checks."""

    def __init__(
        self,
        seed: int,
        solver: SpatialSolverV2 | None = None,
    ) -> None:
        self.seed = seed
        self._random = random.Random(seed)
        self._solver = solver or SpatialSolverV2()
        self._explainer = SpatialExplainerV2(self._solver)
        self._text_adapter = SpatialTextAdapter()
        self._emitted = 0

    def generate(
        self,
        policy: GenerationPolicy,
        *,
        max_attempts: int = 500,
    ) -> GeneratedSpatialSample:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        last_reason = "no candidate was constructed"
        for attempt in range(max_attempts):
            try:
                sample = self._generate_candidate(policy, attempt)
                self._emitted += 1
                return sample
            except _RetryGeneration as exc:
                last_reason = str(exc)
        raise RuntimeError(
            f"could not satisfy generation policy after {max_attempts} attempts: "
            f"{last_reason}"
        )

    def _generate_candidate(
        self,
        policy: GenerationPolicy,
        attempt: int,
    ) -> GeneratedSpatialSample:
        objects = tuple(sorted(self._random.sample(ENTITY_NAMES, policy.num_entities)))
        coordinates = self._coordinates(objects)
        query = self._query(policy, objects, coordinates)
        premise = self._premise(policy, objects, coordinates, query)
        problem = SpatialProblem(objects, premise, query)
        analysis = self._solver.analyze(problem)
        if analysis.error or not analysis.consistent:
            raise _RetryGeneration(
                analysis.error or "generated premise was inconsistent"
            )
        if not self._shape_matches(analysis, policy.semantic_shape):
            raise _RetryGeneration("semantic shape did not match")

        resolution = resolve_answer(analysis, policy.answer_mode)
        options = self._menu(policy, analysis, resolution)
        menu_answer = encode_menu_answer(resolution, options)
        if not menu_answer.is_resolved:
            raise _RetryGeneration(menu_answer.error or "menu was unresolved")

        prompt = self._prompt(problem, options, policy.answer_mode)
        self._verify_round_trip(problem, options, prompt, policy, menu_answer)
        explanation = self._explainer.explain(problem, analysis)
        trace = render_training_trace(
            problem,
            explanation,
            policy.trace_format,
            policy.state_mode,
        )
        trace += "\n" + _render_answer_decision(
            resolution,
            menu_answer,
            policy.trace_format,
        )
        return GeneratedSpatialSample(
            problem,
            options,
            analysis,
            resolution,
            menu_answer,
            explanation,
            prompt,
            trace,
            policy,
            self.seed,
            self._emitted,
            attempt,
            coordinates,
        )

    def _coordinates(self, objects: tuple[str, ...]) -> dict[str, tuple[int, int]]:
        radius = max(2, len(objects) // 2)
        points = [
            (x, y)
            for x in range(-radius, radius + 1)
            for y in range(-radius, radius + 1)
        ]
        selected = self._random.sample(points, len(objects))
        return dict(zip(objects, selected, strict=True))

    def _query(
        self,
        policy: GenerationPolicy,
        objects: tuple[str, ...],
        coordinates: dict[str, tuple[int, int]],
    ) -> DirectionQuery | WhichQuery | CountQuery:
        if policy.query_kind is QueryKind.DIRECTION:
            target, reference = self._random.sample(objects, 2)
            return DirectionQuery(target, reference)

        reference = self._random.choice(objects)
        candidates = tuple(obj for obj in objects if obj != reference)
        directions_by_candidate = {
            candidate: direction_between(coordinates[candidate], coordinates[reference])
            for candidate in candidates
        }
        direction = policy.query_direction
        if direction is None and policy.semantic_shape is SemanticShape.UNIQUE:
            counts = {
                candidate_direction: sum(
                    value is candidate_direction
                    for value in directions_by_candidate.values()
                )
                for candidate_direction in Direction
            }
            unique_directions = [
                candidate_direction
                for candidate_direction, count in counts.items()
                if count == 1
            ]
            if policy.query_kind is QueryKind.WHICH and not unique_directions:
                raise _RetryGeneration("coordinate witness has no unique Which answer")
            direction = self._random.choice(unique_directions or list(Direction))
        direction = direction or self._random.choice(list(Direction))
        if (
            policy.query_kind is QueryKind.WHICH
            and policy.semantic_shape is SemanticShape.UNIQUE
            and sum(value is direction for value in directions_by_candidate.values())
            != 1
        ):
            raise _RetryGeneration("requested Which direction is not unique in witness")
        direction_set = frozenset({direction})
        if policy.query_kind is QueryKind.WHICH:
            return WhichQuery(direction_set, reference, candidates)
        return CountQuery(direction_set, reference, candidates)

    def _premise(
        self,
        policy: GenerationPolicy,
        objects: tuple[str, ...],
        coordinates: dict[str, tuple[int, int]],
        query: DirectionQuery | WhichQuery | CountQuery,
    ) -> And:
        if policy.semantic_shape is SemanticShape.UNIQUE and isinstance(
            query, (WhichQuery, CountQuery)
        ):
            pairs = [(candidate, query.reference) for candidate in query.candidates]
        elif policy.semantic_shape is SemanticShape.UNIQUE and isinstance(
            query, DirectionQuery
        ):
            remainder = [
                obj for obj in objects if obj not in {query.target, query.reference}
            ]
            self._random.shuffle(remainder)
            order = [query.reference, query.target, *remainder]
            pairs = [(query.target, query.reference)] + [
                (order[index], self._random.choice(order[:index]))
                for index in range(2, len(order))
            ]
        else:
            order = list(objects)
            self._random.shuffle(order)
            pairs = [
                (order[index], self._random.choice(order[:index]))
                for index in range(1, len(order))
            ]

        seen = {frozenset(pair) for pair in pairs}
        remaining = [
            (first, second)
            for index, first in enumerate(objects)
            for second in objects[index + 1 :]
            if frozenset((first, second)) not in seen
        ]
        self._random.shuffle(remaining)
        pairs.extend(remaining[: policy.num_premises - len(pairs)])
        atoms = tuple(
            RelationConstraint(
                subject,
                reference,
                frozenset(
                    {direction_between(coordinates[subject], coordinates[reference])}
                ),
            )
            for subject, reference in pairs
        )
        return And(atoms)

    @staticmethod
    def _shape_matches(analysis: QueryAnalysis, shape: SemanticShape) -> bool:
        if shape is SemanticShape.ANY:
            return True
        status = resolve_answer(analysis, AnswerMode.SINGLE).status
        expected = {
            SemanticShape.UNIQUE: ResolutionStatus.EXACT,
            SemanticShape.AMBIGUOUS: ResolutionStatus.AMBIGUOUS,
            SemanticShape.NO_MATCH: ResolutionStatus.NO_MATCH,
        }[shape]
        return status is expected

    def _menu(
        self,
        policy: GenerationPolicy,
        analysis: QueryAnalysis,
        resolution: AnswerResolution,
    ) -> dict[str, str]:
        possible = list(_answer_values(analysis))
        if isinstance(analysis, DirectionAnalysis):
            universe: list[Direction | str | int] = list(Direction)
        elif isinstance(analysis, WhichAnalysis):
            universe = list(analysis.candidates)
        else:
            universe = list(range(len(analysis.candidates) + 1))

        if policy.menu_coverage is MenuCoverage.FULL:
            visible = list(possible)
        elif policy.menu_coverage is MenuCoverage.ZERO:
            visible = []
        else:
            if len(possible) < 2:
                raise _RetryGeneration(
                    "partial coverage needs at least two possibilities"
                )
            count = self._random.randint(1, len(possible) - 1)
            visible = self._random.sample(possible, count)

        distractors = [value for value in universe if value not in possible]
        self._random.shuffle(distractors)
        ordinary_target = max(policy.ordinary_option_target, len(visible))
        visible.extend(distractors[: max(0, ordinary_target - len(visible))])
        self._random.shuffle(visible)
        rendered = [_render_value(value) for value in visible]

        if resolution.status is ResolutionStatus.AMBIGUOUS or (
            policy.answer_mode is AnswerMode.ALL_POSSIBLE
            and policy.menu_coverage is MenuCoverage.PARTIAL
        ):
            rendered.append("Cannot be determined")
        elif not possible or policy.menu_coverage is MenuCoverage.ZERO:
            rendered.append("None of the Options")
        if len(rendered) > len(ascii_uppercase):
            raise _RetryGeneration("menu exceeds available option letters")
        return dict(zip(ascii_uppercase, rendered))

    @staticmethod
    def _prompt(
        problem: SpatialProblem,
        options: dict[str, str],
        answer_mode: AnswerMode,
    ) -> str:
        assert isinstance(problem.premise, And)
        premises = " ".join(
            f"{atom.subject} is to the {next(iter(atom.allowed)).value} of "
            f"{atom.reference}."
            for atom in problem.premise.operands
        )
        instruction = {
            AnswerMode.SINGLE: "Select exactly one answer.",
            AnswerMode.ALL_POSSIBLE: "Select the complete set of possible answers.",
            AnswerMode.VISIBLE_POSSIBLE: (
                "Select every listed answer that is possible; unlisted possibilities "
                "do not affect the selection."
            ),
        }[answer_mode]
        query = problem.query
        if isinstance(query, DirectionQuery):
            question = (
                f"In which direction is {query.target} relative to {query.reference}?"
            )
        elif isinstance(query, WhichQuery):
            direction = next(iter(query.directions)).value
            question = (
                f"Which object in the map is in the {direction} of {query.reference}?"
            )
        else:
            direction = next(iter(query.directions)).value
            question = f"How many objects are in the {direction} of {query.reference}?"
        menu = ", ".join(f"{letter}. {value}" for letter, value in options.items())
        return (
            f"Consider a map with multiple locations:\n\n{premises}\n\n"
            f"Question: {instruction} {question} Available options: {menu}"
        )

    def _verify_round_trip(
        self,
        problem: SpatialProblem,
        options: dict[str, str],
        prompt: str,
        policy: GenerationPolicy,
        expected_menu: MenuAnswer,
    ) -> None:
        try:
            parsed = self._text_adapter.parse(prompt)
        except ValueError as exc:
            raise _RetryGeneration(f"rendered prompt did not parse: {exc}") from exc
        if parsed.problem != problem or parsed.options != options:
            raise _RetryGeneration("rendered prompt changed the structured problem")
        reparsed_analysis = self._solver.analyze(parsed.problem)
        reparsed_resolution = resolve_answer(reparsed_analysis, policy.answer_mode)
        reparsed_menu = encode_menu_answer(reparsed_resolution, parsed.options)
        if (
            reparsed_resolution.status != expected_menu.resolution.status
            or reparsed_resolution.values != expected_menu.resolution.values
            or reparsed_menu.status != expected_menu.status
            or reparsed_menu.letters != expected_menu.letters
        ):
            raise _RetryGeneration("round-trip semantic answer changed")


def _system_prompt(mode: AnswerMode) -> str:
    contract = {
        AnswerMode.SINGLE: (
            "Choose exactly one option. If more than one answer remains possible, "
            "choose Cannot be determined."
        ),
        AnswerMode.ALL_POSSIBLE: (
            "Choose every possible answer. If the menu contains only part of the "
            "complete set, choose Cannot be determined."
        ),
        AnswerMode.VISIBLE_POSSIBLE: (
            "Choose every displayed option that is possible. Ignore possible values "
            "that are not displayed."
        ),
    }[mode]
    return (
        "Solve the spatial reasoning problem from the stated relations. "
        f"{contract} Return the selected option letter or letters."
    )


def _render_answer_decision(
    resolution: AnswerResolution,
    menu_answer: MenuAnswer,
    trace_format: TraceFormat,
) -> str:
    letters = ", ".join(sorted(menu_answer.letters))
    possible = ", ".join(
        _render_value(value) for value in _answer_values(resolution.analysis)
    )
    if trace_format is TraceFormat.SYMBOLIC:
        return (
            f"Answer-Mode={resolution.mode.value}; "
            f"Possible={{{possible}}}; "
            f"Resolution={resolution.status.value}; "
            f"Menu-Status={menu_answer.status}; Select={{{letters}}}"
        )
    contract = {
        AnswerMode.SINGLE: "The question requires one invariant answer.",
        AnswerMode.ALL_POSSIBLE: "The question requires the complete possibility set.",
        AnswerMode.VISIBLE_POSSIBLE: (
            "The question requires every displayed option that is possible."
        ),
    }[resolution.mode]
    return (
        f"Answer policy: {contract} The possible values are "
        f"{possible if possible else 'none'}. The menu result is "
        f"{menu_answer.status}, so select {letters}."
    )
