"""Policy-driven, solver-validated SpatialMap V2 generation.

This module owns synthetic-data choices.  It constructs structured problems,
asks the data-agnostic solver for their complete semantics, builds a menu under
an explicit answer contract, and round-trips the rendered prompt before a row
may be emitted.  Coordinates are retained only as audit witnesses.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations, pairwise
from string import ascii_uppercase
from typing import Any

from spatial.v2.answer_certificate_renderers import render_answer_certificate
from spatial.v2.certificate_generation import (
    AnswerCertificate,
    answer_certificate_to_dict,
    build_answer_certificate,
)
from spatial.v2.difficulty import measure_difficulty
from spatial.v2.grading import (
    AnswerMode,
    AnswerResolution,
    MenuAnswer,
    ResolutionStatus,
    encode_menu_answer,
    resolve_answer,
)
from spatial.v2.proofs import ProofConstructionError
from spatial.v2.solver import (
    And,
    CountAnalysis,
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
    direction_signs,
)
from spatial.v2.text import (
    SpatialTextAdapter,
    render_spatial_formula,
    spatial_formula_objects,
)
from spatial.v2.trace import TraceFormat

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
_DIRECTION_BY_SIGNS = {direction_signs(value): value for value in Direction}


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


def _validate_depth_range(name: str, minimum: int, maximum: int | None) -> None:
    if minimum < 1:
        raise ValueError(f"min_{name}_depth must be positive")
    if maximum is not None and maximum < minimum:
        raise ValueError(f"max_{name}_depth cannot be below min_{name}_depth")


@dataclass(frozen=True)
class GenerationPolicy:
    """Dataset policy; none of these controls belong to the solver."""

    query_kind: QueryKind = QueryKind.DIRECTION
    answer_mode: AnswerMode = AnswerMode.SINGLE
    semantic_shape: SemanticShape = SemanticShape.ANY
    menu_coverage: MenuCoverage = MenuCoverage.FULL
    trace_format: TraceFormat = TraceFormat.NATURAL
    num_entities: int = 6
    num_premises: int = 7
    ordinary_option_target: int = 4
    query_direction: Direction | None = None
    target_direction: Direction | None = None
    omit_direct_query_relation: bool = False
    min_axis_depth: int = 1
    max_axis_depth: int | None = None
    require_independent_axes: bool = False
    ambiguity_size: int | None = None
    min_membership_depth: int = 1
    max_membership_depth: int | None = None
    distractor_premises: int = 0

    def __post_init__(self) -> None:
        self._normalize_enums()
        self._validate_size()
        self._validate_answer_contract()
        self._validate_difficulty()

    def _normalize_enums(self) -> None:
        object.__setattr__(self, "query_kind", QueryKind(self.query_kind))
        object.__setattr__(self, "answer_mode", AnswerMode(self.answer_mode))
        object.__setattr__(self, "semantic_shape", SemanticShape(self.semantic_shape))
        object.__setattr__(self, "menu_coverage", MenuCoverage(self.menu_coverage))
        object.__setattr__(self, "trace_format", TraceFormat(self.trace_format))
        if self.query_direction is not None:
            object.__setattr__(self, "query_direction", Direction(self.query_direction))
        if self.target_direction is not None:
            object.__setattr__(
                self, "target_direction", Direction(self.target_direction)
            )

    def _validate_size(self) -> None:
        if not 2 <= self.num_entities <= len(ENTITY_NAMES):
            raise ValueError(f"num_entities must be between 2 and {len(ENTITY_NAMES)}")
        maximum_pairs = self.num_entities * (self.num_entities - 1) // 2
        if not self.num_entities - 1 <= self.num_premises <= maximum_pairs:
            raise ValueError(
                "num_premises must connect every entity and cannot exceed all pairs"
            )
        if not 1 <= self.ordinary_option_target <= len(ascii_uppercase) - 1:
            raise ValueError("ordinary_option_target must be between 1 and 25")

    def _validate_answer_contract(self) -> None:
        if self.query_kind is QueryKind.DIRECTION and self.query_direction is not None:
            raise ValueError("query_direction applies only to Which and Count queries")
        if (
            self.query_kind is not QueryKind.DIRECTION
            and self.target_direction is not None
        ):
            raise ValueError("target_direction applies only to Direction queries")
        if (
            self.target_direction is not None
            and self.semantic_shape is not SemanticShape.UNIQUE
        ):
            raise ValueError("target_direction requires unique semantics")
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

    def _validate_difficulty(self) -> None:
        _validate_depth_range("axis", self.min_axis_depth, self.max_axis_depth)
        _validate_depth_range(
            "membership",
            self.min_membership_depth,
            self.max_membership_depth,
        )
        axis_controlled = self.min_axis_depth > 1 or self.max_axis_depth is not None
        membership_controlled = (
            self.min_membership_depth > 1 or self.max_membership_depth is not None
        )
        if self.query_kind is not QueryKind.DIRECTION and axis_controlled:
            raise ValueError("axis depth controls apply only to Direction queries")
        if (
            self.omit_direct_query_relation
            or axis_controlled
            or membership_controlled
            or self.require_independent_axes
            or self.distractor_premises
        ) and self.semantic_shape is not SemanticShape.UNIQUE:
            raise ValueError("proof controls require unique semantics")
        if self.query_kind is QueryKind.DIRECTION and membership_controlled:
            raise ValueError("membership depth controls apply only to Which or Count")
        if self.require_independent_axes:
            relevant_depth = (
                self.min_axis_depth
                if self.query_kind is QueryKind.DIRECTION
                else self.min_membership_depth
            )
            if relevant_depth < 2:
                raise ValueError("independent axes require a proof depth of at least 2")
        self._validate_ambiguity_control()
        self._validate_distractor_control()
        self._validate_feasibility()
        if self.omit_direct_query_relation and self.num_entities < 3:
            raise ValueError(
                "omitting the direct query relation requires three entities"
            )

    def _validate_ambiguity_control(self) -> None:
        if self.ambiguity_size is not None:
            if self.ambiguity_size < 2:
                raise ValueError("ambiguity_size must be at least 2")
            maximum = {
                QueryKind.DIRECTION: len(Direction),
                QueryKind.WHICH: self.num_entities - 1,
                QueryKind.COUNT: self.num_entities,
            }[self.query_kind]
            if self.ambiguity_size > maximum:
                raise ValueError(
                    f"ambiguity_size can be at most {maximum} for "
                    f"{self.query_kind.value}"
                )
            if self.semantic_shape is not SemanticShape.AMBIGUOUS:
                raise ValueError("ambiguity_size requires ambiguous semantics")

    def _validate_distractor_control(self) -> None:
        if not 0 <= self.distractor_premises <= self.num_premises:
            raise ValueError("distractor_premises must be within the premise count")
        if self.distractor_premises and self.query_kind is not QueryKind.DIRECTION:
            raise ValueError("distractor controls currently require a Direction proof")

    def _validate_feasibility(self) -> None:
        maximum_depth = min(self.num_entities - 1, self.num_premises)
        requested_depth = (
            self.min_axis_depth
            if self.query_kind is QueryKind.DIRECTION
            else self.min_membership_depth
        )
        if requested_depth > maximum_depth:
            raise ValueError("premise and entity budgets cannot satisfy proof depth")

        maximum_pairs = self.num_entities * (self.num_entities - 1) // 2
        protects_a_pair = (
            self.query_kind is QueryKind.DIRECTION and self.omit_direct_query_relation
        ) or self.min_membership_depth > 1
        if protects_a_pair and self.num_premises >= maximum_pairs:
            raise ValueError("premise budget cannot satisfy direct-relation omission")

        minimum_support = requested_depth * (2 if self.require_independent_axes else 1)
        if self.distractor_premises + minimum_support > self.num_premises:
            raise ValueError(
                "premise budget cannot satisfy proof and distractor counts"
            )


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


def _exact_relation(
    subject: str,
    direction: Direction,
    reference: str,
) -> RelationConstraint:
    return RelationConstraint(subject, reference, frozenset({direction}))


def _render_trace(
    certificate: AnswerCertificate,
    resolution: AnswerResolution,
    menu_answer: MenuAnswer,
    policy: GenerationPolicy,
) -> str:
    reasoning = render_answer_certificate(
        certificate,
        policy.trace_format,
        include_coordinates=False,
    )
    decision = _render_answer_decision(
        resolution,
        menu_answer,
        policy.trace_format,
    )
    return f"{reasoning}\n{decision}"


@dataclass(frozen=True)
class GeneratedSpatialSample:
    problem: SpatialProblem
    options: dict[str, str]
    analysis: QueryAnalysis
    resolution: AnswerResolution
    menu_answer: MenuAnswer
    certificate: AnswerCertificate
    prompt: str
    trace: str
    policy: GenerationPolicy
    seed: int
    sample_index: int
    attempt: int
    generation_witness: dict[str, tuple[int, int]]
    difficulty: dict[str, Any]
    rejection_counts: dict[str, int]

    @property
    def base_id(self) -> str:
        return f"spatial-v2-{self.seed}-{self.sample_index}-{self.attempt}"

    @property
    def audit_metadata(self) -> dict[str, Any]:
        return {
            "generation_witness": _coordinate_payload(self.generation_witness),
            "answer_certificate": answer_certificate_to_dict(self.certificate),
        }

    def as_sft_row(self, *, include_audit: bool = False) -> dict[str, Any]:
        answer = ", ".join(sorted(self.menu_answer.letters))
        query_direction = (
            next(iter(self.problem.query.directions)).value
            if isinstance(self.problem.query, (WhichQuery, CountQuery))
            and len(self.problem.query.directions) == 1
            else None
        )
        target_direction = (
            self.analysis.possible_directions[0].value
            if isinstance(self.analysis, DirectionAnalysis)
            and len(self.analysis.possible_directions) == 1
            else None
        )
        row: dict[str, Any] = {
            "id": (
                f"{self.base_id}-{self.policy.answer_mode.value}-"
                f"{self.policy.menu_coverage.value}-"
                f"{self.policy.trace_format.value}"
            ),
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
                "base_id": self.base_id,
                "seed": self.seed,
                "sample_index": self.sample_index,
                "attempt": self.attempt,
                "rejection_counts": dict(sorted(self.rejection_counts.items())),
                "query_kind": self.policy.query_kind.value,
                "query_direction": query_direction,
                "target_direction": target_direction,
                "answer_mode": self.policy.answer_mode.value,
                "semantic_shape": self.policy.semantic_shape.value,
                "menu_coverage": self.policy.menu_coverage.value,
                "trace_format": self.policy.trace_format.value,
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
                "difficulty": self.difficulty,
            },
        }
        if include_audit:
            row["audit"] = self.audit_metadata
        return row

    def with_trace(
        self,
        trace_format: TraceFormat | str,
    ) -> GeneratedSpatialSample:
        """Render another training trace for the same problem and gold answer."""
        trace_format = TraceFormat(trace_format)
        policy = replace(
            self.policy,
            trace_format=trace_format,
        )
        trace = _render_trace(
            self.certificate,
            self.resolution,
            self.menu_answer,
            policy,
        )
        return replace(self, policy=policy, trace=trace)


class _RetryGeneration(Exception):
    pass


def _difficulty_matches(
    policy: GenerationPolicy,
    difficulty: dict[str, Any],
) -> bool:
    if (
        (policy.omit_direct_query_relation and difficulty["direct_query_relation"])
        or (policy.require_independent_axes and not difficulty["axes_independent"])
        or (
            policy.ambiguity_size is not None
            and difficulty["possibility_count"] != policy.ambiguity_size
        )
        or (
            policy.distractor_premises
            and difficulty["num_distractor_premises"] != policy.distractor_premises
        )
    ):
        return False
    if policy.min_membership_depth > 1 or policy.max_membership_depth is not None:
        proofs = difficulty["membership_proofs"]
        depths = [proof[axis] for proof in proofs for axis in ("x_depth", "y_depth")]
        if not _depths_match(
            depths,
            policy.min_membership_depth,
            policy.max_membership_depth,
        ):
            return False
    if policy.min_axis_depth == 1 and policy.max_axis_depth is None:
        return True
    return _depths_match(
        (difficulty["x_depth"], difficulty["y_depth"]),
        policy.min_axis_depth,
        policy.max_axis_depth,
    )


def _depths_match(
    depths: list[int] | tuple[int | None, ...],
    minimum: int,
    maximum: int | None,
) -> bool:
    return bool(depths) and all(
        depth is not None and depth >= minimum and (maximum is None or depth <= maximum)
        for depth in depths
    )


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
        rejections: Counter[str] = Counter()
        for attempt in range(max_attempts):
            try:
                sample = self._generate_candidate(policy, attempt)
                self._emitted += 1
                return replace(
                    sample, rejection_counts=dict(sorted(rejections.items()))
                )
            except _RetryGeneration as exc:
                rejections[str(exc)] += 1
        summary = ", ".join(
            f"{reason}={count}" for reason, count in sorted(rejections.items())
        )
        raise RuntimeError(
            f"could not satisfy generation policy after {max_attempts} attempts; "
            f"rejections: {summary}"
        )

    def with_answer_mode(
        self,
        sample: GeneratedSpatialSample,
        answer_mode: AnswerMode | str,
        menu_coverage: MenuCoverage | str,
    ) -> GeneratedSpatialSample:
        """Render another answer contract for an already solved base problem."""
        policy = replace(
            sample.policy,
            answer_mode=AnswerMode(answer_mode),
            menu_coverage=MenuCoverage(menu_coverage),
        )
        resolution = resolve_answer(sample.analysis, policy.answer_mode)
        options = self._menu(policy, sample.analysis, resolution)
        menu_answer = encode_menu_answer(resolution, options)
        if not menu_answer.is_resolved:
            raise ValueError(menu_answer.error or "answer variant menu was unresolved")
        prompt = self._prompt(sample.problem, options, policy.answer_mode)
        self._verify_round_trip(
            sample.problem,
            options,
            prompt,
            policy,
            menu_answer,
        )
        trace = _render_trace(
            sample.certificate,
            resolution,
            menu_answer,
            policy,
        )
        return replace(
            sample,
            options=options,
            resolution=resolution,
            menu_answer=menu_answer,
            prompt=prompt,
            trace=trace,
            policy=policy,
        )

    def _generate_candidate(
        self,
        policy: GenerationPolicy,
        attempt: int,
    ) -> GeneratedSpatialSample:
        objects = tuple(sorted(self._random.sample(ENTITY_NAMES, policy.num_entities)))
        controlled_direction = self._uses_controlled_direction(policy)
        controlled_membership = self._uses_controlled_membership(policy)
        if controlled_direction:
            problem = self._controlled_direction_problem(policy, objects)
            coordinates: dict[str, tuple[int, int]] = {}
        elif controlled_membership:
            problem = self._controlled_membership_problem(policy, objects)
            coordinates = {}
        else:
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
        difficulty = measure_difficulty(problem, analysis, self._solver)
        if not _difficulty_matches(policy, difficulty):
            raise _RetryGeneration("difficulty controls did not match")
        if controlled_direction:
            assert isinstance(analysis, DirectionAnalysis)
            coordinates = analysis.coordinates or {}
        elif controlled_membership:
            assert isinstance(analysis, (WhichAnalysis, CountAnalysis))
            coordinates = next(iter(analysis.witnesses.values()), {})
        try:
            certificate = build_answer_certificate(problem, analysis, self._solver)
        except ProofConstructionError as exc:
            raise _RetryGeneration(
                f"proof-first certificate construction failed: {exc}"
            ) from exc

        resolution = resolve_answer(analysis, policy.answer_mode)
        options = self._menu(policy, analysis, resolution)
        menu_answer = encode_menu_answer(resolution, options)
        if not menu_answer.is_resolved:
            raise _RetryGeneration(menu_answer.error or "menu was unresolved")

        prompt = self._prompt(problem, options, policy.answer_mode)
        self._verify_round_trip(problem, options, prompt, policy, menu_answer)
        trace = _render_trace(
            certificate,
            resolution,
            menu_answer,
            policy,
        )
        return GeneratedSpatialSample(
            problem,
            options,
            analysis,
            resolution,
            menu_answer,
            certificate,
            prompt,
            trace,
            policy,
            self.seed,
            self._emitted,
            attempt,
            coordinates,
            difficulty,
            {},
        )

    @staticmethod
    def _uses_controlled_direction(policy: GenerationPolicy) -> bool:
        requested = policy.query_kind is QueryKind.DIRECTION and (
            policy.omit_direct_query_relation
            or policy.min_axis_depth > 1
            or policy.require_independent_axes
            or bool(policy.distractor_premises)
        )
        if not requested:
            return False
        try:
            x_depth, y_depth = SpatialGeneratorV2._controlled_depths(policy)
        except _RetryGeneration:
            return False
        proof_nodes = (
            x_depth + y_depth if policy.require_independent_axes else x_depth + 1
        )
        extras = policy.num_entities - proof_nodes
        filler = policy.num_premises - (
            x_depth + y_depth if policy.require_independent_axes else x_depth
        )
        filler_capacity = extras * (extras + 1) // 2
        return extras >= 0 and 0 <= filler <= filler_capacity

    def _controlled_direction_problem(
        self,
        policy: GenerationPolicy,
        objects: tuple[str, ...],
    ) -> SpatialProblem:
        direction = policy.target_direction or self._random.choice(list(Direction))
        x_depth, y_depth = self._controlled_depths(policy)
        shuffled = list(objects)
        self._random.shuffle(shuffled)
        reference, target = shuffled[:2]
        cursor = 2

        if policy.require_independent_axes:
            relations, cursor = self._independent_axis_relations(
                shuffled,
                reference,
                target,
                direction,
                x_depth,
                y_depth,
            )
        else:
            internal = shuffled[cursor : cursor + x_depth - 1]
            cursor += x_depth - 1
            nodes = [reference, *internal, target]
            relations = [
                _exact_relation(subject, direction, previous)
                for previous, subject in pairwise(nodes)
            ]

        extras = shuffled[cursor:]
        filler_count = policy.num_premises - len(relations)
        filler_pairs = list(combinations((reference, *extras), 2))
        if filler_count > len(filler_pairs):
            raise _RetryGeneration("not enough isolated pairs for distractor premises")
        relations.extend(
            _exact_relation(higher, Direction.NORTHEAST, lower)
            for lower, higher in filler_pairs[:filler_count]
        )
        self._random.shuffle(relations)
        return SpatialProblem(
            objects,
            And(tuple(relations)),
            DirectionQuery(target, reference),
        )

    @staticmethod
    def _uses_controlled_membership(policy: GenerationPolicy) -> bool:
        if (
            policy.query_kind not in {QueryKind.WHICH, QueryKind.COUNT}
            or policy.semantic_shape is not SemanticShape.UNIQUE
            or policy.min_membership_depth <= 1
        ):
            return False
        depth = policy.min_membership_depth
        proof_nodes = 2 * depth
        extras = policy.num_entities - proof_nodes
        filler = policy.num_premises - policy.num_entities
        return extras >= 0 and 0 <= filler <= extras * (extras - 1) // 2

    def _controlled_membership_problem(
        self,
        policy: GenerationPolicy,
        objects: tuple[str, ...],
    ) -> SpatialProblem:
        direction = policy.query_direction or self._random.choice(list(Direction))
        depth = policy.min_membership_depth
        shuffled = list(objects)
        self._random.shuffle(shuffled)
        reference, target = shuffled[:2]
        relations, cursor = self._independent_axis_relations(
            shuffled,
            reference,
            target,
            direction,
            depth,
            depth,
            avoid_target_prefix=True,
        )
        extras = shuffled[cursor:]
        nonmember_direction = next(
            candidate
            for candidate in (
                Direction.NORTHEAST,
                Direction.SOUTHEAST,
                Direction.SOUTHWEST,
                Direction.NORTHWEST,
            )
            if candidate is not direction
        )
        relations.extend(
            _exact_relation(candidate, nonmember_direction, reference)
            for candidate in extras
        )
        filler_count = policy.num_premises - len(relations)
        filler_pairs = list(combinations(extras, 2))
        relations.extend(
            _exact_relation(higher, nonmember_direction, lower)
            for lower, higher in filler_pairs[:filler_count]
        )
        self._random.shuffle(relations)
        direction_set = frozenset({direction})
        candidates = tuple(obj for obj in objects if obj != reference)
        query = (
            WhichQuery(direction_set, reference, candidates)
            if policy.query_kind is QueryKind.WHICH
            else CountQuery(direction_set, reference, candidates)
        )
        return SpatialProblem(objects, And(tuple(relations)), query)

    @staticmethod
    def _controlled_depths(policy: GenerationPolicy) -> tuple[int, int]:
        support = (
            policy.num_premises - policy.distractor_premises
            if policy.distractor_premises
            else None
        )
        if not policy.require_independent_axes:
            depth = support or policy.min_axis_depth
            if policy.max_axis_depth is not None and depth > policy.max_axis_depth:
                raise _RetryGeneration("proof and distractor depths are incompatible")
            return depth, depth

        if support is None:
            return policy.min_axis_depth, policy.min_axis_depth
        maximum = policy.max_axis_depth or support
        for x_depth in range(policy.min_axis_depth, maximum + 1):
            y_depth = support - x_depth
            if policy.min_axis_depth <= y_depth <= maximum:
                return x_depth, y_depth
        raise _RetryGeneration(
            "independent proof and distractor depths are incompatible"
        )

    def _independent_axis_relations(
        self,
        objects: list[str],
        reference: str,
        target: str,
        direction: Direction,
        x_depth: int,
        y_depth: int,
        *,
        avoid_target_prefix: bool = False,
    ) -> tuple[list[RelationConstraint], int]:
        cursor = 2
        x_internal = objects[cursor : cursor + x_depth - 1]
        cursor += x_depth - 1
        y_internal = objects[cursor : cursor + y_depth - 1]
        cursor += y_depth - 1
        relations = [
            *self._axis_path_relations(
                [reference, *x_internal, target],
                direction,
                axis=0,
                avoid_target_prefix=avoid_target_prefix,
            ),
            *self._axis_path_relations(
                [reference, *y_internal, target],
                direction,
                axis=1,
                avoid_target_prefix=avoid_target_prefix,
            ),
        ]
        return relations, cursor

    @staticmethod
    def _axis_path_relations(
        nodes: list[str],
        direction: Direction,
        *,
        axis: int,
        avoid_target_prefix: bool = False,
    ) -> list[RelationConstraint]:
        fixed_sign = direction_signs(direction)[axis]
        free_sign = direction_signs(direction)[1 - axis]
        if avoid_target_prefix:
            prefix_sign = -free_sign if free_sign else 1
            free_components = [
                *([prefix_sign] * (len(nodes) - 2)),
                -prefix_sign,
            ]
        else:
            free_components = [1, -1, *([free_sign] * (len(nodes) - 3))]
        signs = (
            [(fixed_sign, component) for component in free_components]
            if axis == 0
            else [(component, fixed_sign) for component in free_components]
        )
        return [
            _exact_relation(subject, _DIRECTION_BY_SIGNS[sign], previous)
            for (previous, subject), sign in zip(pairwise(nodes), signs)
        ]

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
            pairs = [
                (target, reference)
                for target in objects
                for reference in objects
                if target != reference
                and (
                    policy.target_direction is None
                    or direction_between(coordinates[target], coordinates[reference])
                    is policy.target_direction
                )
            ]
            if not pairs:
                raise _RetryGeneration("coordinate witness lacks target direction")
            target, reference = self._random.choice(pairs)
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
            if policy.query_kind is QueryKind.WHICH:
                direction = self._random.choice(unique_directions)
            elif policy.min_membership_depth > 1:
                positive_directions = [
                    candidate_direction
                    for candidate_direction, count in counts.items()
                    if 0 < count < len(candidates)
                ]
                if not positive_directions:
                    raise _RetryGeneration(
                        "coordinate witness has no nontrivial Count membership"
                    )
                direction = self._random.choice(positive_directions)
            else:
                direction = self._random.choice(list(Direction))
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
        forbidden: set[frozenset[str]] = set()
        controlled_membership = isinstance(query, (WhichQuery, CountQuery)) and (
            policy.omit_direct_query_relation or policy.min_membership_depth > 1
        )
        if (
            policy.semantic_shape is SemanticShape.UNIQUE
            and isinstance(query, (WhichQuery, CountQuery))
            and not controlled_membership
        ):
            pairs = [(candidate, query.reference) for candidate in query.candidates]
        elif (
            policy.semantic_shape is SemanticShape.UNIQUE
            and isinstance(query, DirectionQuery)
            and not policy.omit_direct_query_relation
            and policy.min_axis_depth == 1
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
            if isinstance(query, DirectionQuery) and policy.omit_direct_query_relation:
                forbidden.add(frozenset((query.target, query.reference)))
            elif controlled_membership:
                forbidden.update(
                    frozenset((candidate, query.reference))
                    for candidate in query.candidates
                    if direction_between(
                        coordinates[candidate], coordinates[query.reference]
                    )
                    in query.directions
                )
            pairs = self._spanning_pairs(objects, forbidden)

        seen = {frozenset(pair) for pair in pairs}
        remaining = [
            (first, second)
            for index, first in enumerate(objects)
            for second in objects[index + 1 :]
            if frozenset((first, second)) not in seen
            and frozenset((first, second)) not in forbidden
        ]
        self._random.shuffle(remaining)
        pairs.extend(remaining[: policy.num_premises - len(pairs)])
        atoms = tuple(
            _exact_relation(
                subject,
                direction_between(coordinates[subject], coordinates[reference]),
                reference,
            )
            for subject, reference in pairs
        )
        return And(atoms)

    def _spanning_pairs(
        self,
        objects: tuple[str, ...],
        forbidden: set[frozenset[str]],
    ) -> list[tuple[str, str]]:
        parent = {obj: obj for obj in objects}

        def root(obj: str) -> str:
            while parent[obj] != obj:
                parent[obj] = parent[parent[obj]]
                obj = parent[obj]
            return obj

        candidates = [
            (first, second)
            for index, first in enumerate(objects)
            for second in objects[index + 1 :]
            if frozenset((first, second)) not in forbidden
        ]
        self._random.shuffle(candidates)
        selected = []
        for first, second in candidates:
            first_root = root(first)
            second_root = root(second)
            if first_root == second_root:
                continue
            parent[second_root] = first_root
            selected.append((first, second))
            if len(selected) == len(objects) - 1:
                return selected
        raise _RetryGeneration("protected query pairs disconnect the premise graph")

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
        formulas = (
            problem.premise.operands
            if isinstance(problem.premise, And)
            else (problem.premise,)
        )
        premise_text = [f"{render_spatial_formula(formula)}." for formula in formulas]
        mentioned = {
            name for formula in formulas for name in spatial_formula_objects(formula)
        }
        premise_text.extend(
            f"{name} is in the map."
            for name in problem.objects
            if name not in mentioned
        )
        premises = " ".join(premise_text)
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
