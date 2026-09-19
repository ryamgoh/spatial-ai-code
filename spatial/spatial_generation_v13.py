"""Deep v13 synthetic generation module.

This is intentionally a narrow foundation.  It preserves the v6 question and
option laws, adds cardinal relations, and records structure measured by
``SpatialSolverV13`` after reparsing the final prompt.  Later v13 iterations
will add controlled proof-depth targets, distractors, and richer conflicts.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from spatial_graph import AxisGraph, ENTITY_NAMES
from spatial_solver import Grade
from spatial_solver_v13 import SolvedProblem, SpatialSolverV13


COMPOUNDS = ["Northeast", "Northwest", "Southeast", "Southwest"]
DIRECTIONS = ["North", "South", "East", "West", *COMPOUNDS]
SPECIAL = "Cannot be determined"
NONE = "None of the Options"


class RelationMode(str, Enum):
    DIAGONAL = "diagonal"
    CARDINAL = "cardinal"
    MIXED = "mixed"


class SemanticSubtype(str, Enum):
    DIR_1 = "dir-1"
    DIR_2 = "dir-2"
    DIR_UNDETERMINED = "dir-undetermined"
    DIR_CYCLE = "dir-cycle"
    DIR_INCOMPLETE = "dir-incomplete"
    DIR_OMIT = "dir-omit"
    WHICH_1 = "which-1"
    WHICH_2 = "which-2"
    WHICH_3 = "which-3"
    WHICH_4 = "which-4"
    WHICH_0 = "which-0"
    COUNT_1 = "count-1"
    COUNT_OMIT = "count-omit"


class Direction(str, Enum):
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"
    NORTHEAST = "Northeast"
    NORTHWEST = "Northwest"
    SOUTHEAST = "Southeast"
    SOUTHWEST = "Southwest"


SEMANTIC_SUBTYPES = tuple(subtype.value for subtype in SemanticSubtype)


@dataclass(frozen=True)
class DepthRange:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum < 1:
            raise ValueError("depth minimum must be positive")
        if self.maximum < self.minimum:
            raise ValueError("depth maximum must be greater than or equal to minimum")

    @classmethod
    def exact(cls, depth: int) -> "DepthRange":
        return cls(depth, depth)

    def contains(self, depth: int | None) -> bool:
        return depth is not None and self.minimum <= depth <= self.maximum

    def choose(self, rng: random.Random) -> int:
        return rng.randint(self.minimum, self.maximum)


class DistractorPolicy(str, Enum):
    NONE = "none"
    DISCONNECTED = "disconnected"
    QUERY_BRANCH = "query-branch"


@dataclass(frozen=True)
class DistractorSpec:
    policy: DistractorPolicy = DistractorPolicy.NONE
    count: int = 0

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("distractor count must be non-negative")
        if self.policy is DistractorPolicy.NONE and self.count != 0:
            raise ValueError("none distractor policy requires count 0")
        if self.policy is not DistractorPolicy.NONE and self.count < 1:
            raise ValueError("a distractor policy requires a positive count")


class CycleAxes(str, Enum):
    X = "x"
    Y = "y"
    BOTH = "both"


class CycleTopology(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"


class CyclePlacement(str, Enum):
    QUERY_CONNECTED = "query-connected"
    DISCONNECTED = "disconnected"


class WorldConsistency(str, Enum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True)
class CycleSpec:
    axes: CycleAxes
    topology: CycleTopology
    placement: CyclePlacement
    length: int
    world_consistency: WorldConsistency = WorldConsistency.INCONSISTENT

    def __post_init__(self) -> None:
        if self.topology is CycleTopology.DIRECT and self.length != 2:
            raise ValueError("direct cycle requires length 2")
        if self.topology is CycleTopology.INDIRECT and self.length < 3:
            raise ValueError("indirect cycle requires length of at least 3")


@dataclass(frozen=True)
class StructuralConstraints:
    require_independent_axes: bool = False
    x_depth: DepthRange | None = None
    y_depth: DepthRange | None = None
    distractors: DistractorSpec = field(default_factory=DistractorSpec)


@dataclass(frozen=True)
class Relation:
    subject: str
    direction: Direction
    reference: str

    @property
    def x_component(self) -> str | None:
        return _components(self.direction.value)[0]

    @property
    def y_component(self) -> str | None:
        return _components(self.direction.value)[1]

    def render(self) -> str:
        return (
            f"The {self.subject} is to the {self.direction.value} "
            f"of the {self.reference}."
        )


@dataclass(frozen=True)
class Scene:
    entities: tuple[str, ...]
    relations: tuple[Relation, ...]


@dataclass(frozen=True)
class _SubtypePolicy:
    question_type: int
    target_num_answers: int | None


_SUBTYPE_POLICIES = {
    SemanticSubtype.DIR_1: _SubtypePolicy(0, 1),
    SemanticSubtype.DIR_2: _SubtypePolicy(0, 2),
    SemanticSubtype.DIR_UNDETERMINED: _SubtypePolicy(0, 0),
    SemanticSubtype.DIR_CYCLE: _SubtypePolicy(0, 0),
    SemanticSubtype.DIR_INCOMPLETE: _SubtypePolicy(0, 2),
    SemanticSubtype.DIR_OMIT: _SubtypePolicy(0, 1),
    SemanticSubtype.WHICH_1: _SubtypePolicy(1, 1),
    SemanticSubtype.WHICH_2: _SubtypePolicy(1, 2),
    SemanticSubtype.WHICH_3: _SubtypePolicy(1, 3),
    SemanticSubtype.WHICH_4: _SubtypePolicy(1, 4),
    SemanticSubtype.WHICH_0: _SubtypePolicy(1, 0),
    SemanticSubtype.COUNT_1: _SubtypePolicy(2, None),
    SemanticSubtype.COUNT_OMIT: _SubtypePolicy(2, 0),
}


@dataclass(frozen=True)
class GenerationSpec:
    semantic_subtype: SemanticSubtype
    relation_mode: RelationMode
    constraints: StructuralConstraints = field(default_factory=StructuralConstraints)
    num_entities: int = 8
    num_relations: int = 10
    max_attempts: int = 500
    system_prompt: str | None = None
    cycle: CycleSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_subtype, SemanticSubtype):
            raise TypeError("semantic_subtype must be a SemanticSubtype")
        if not isinstance(self.relation_mode, RelationMode):
            raise TypeError("relation_mode must be a RelationMode")
        if not 2 <= self.num_entities <= len(ENTITY_NAMES):
            raise ValueError(f"num_entities must be in [2, {len(ENTITY_NAMES)}]")
        if self.num_relations < 1:
            raise ValueError("num_relations must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.cycle is not None and self.semantic_subtype is SemanticSubtype.DIR_CYCLE:
            raise ValueError(
                "use a base semantic subtype with CycleSpec; dir-cycle is the "
                "compatibility subtype"
            )
        if self.cycle is not None and (
            self.constraints.x_depth
            or self.constraints.y_depth
            or self.constraints.distractors.policy is not DistractorPolicy.NONE
        ):
            raise ValueError(
                "CycleSpec is currently separate from depth/distractor constraints"
            )
        if self.cycle is not None:
            cycle_nodes = self.cycle.length + (
                1
                if self.cycle.world_consistency is WorldConsistency.CONSISTENT
                else 0
            )
            fresh_cycle_entities = cycle_nodes - (
                1
                if self.cycle.placement is CyclePlacement.QUERY_CONNECTED
                else 0
            )
            cycle_relations = _cycle_relation_count(
                self.relation_mode, self.cycle
            )
            if self.num_entities < 8 + fresh_cycle_entities:
                raise ValueError(
                    "num_entities cannot fit an eight-entity base question "
                    "plus the requested cycle/control"
                )
            if self.num_relations < 10 + cycle_relations:
                raise ValueError(
                    "num_relations cannot fit a ten-relation base question "
                    "plus the requested cycle/control"
                )
        if (
            self.constraints.require_independent_axes
            and self.semantic_subtype is not SemanticSubtype.DIR_1
        ):
            raise ValueError("independent axes are currently supported only for dir-1")
        if (self.constraints.x_depth or self.constraints.y_depth) and (
            self.semantic_subtype is not SemanticSubtype.DIR_1
        ):
            raise ValueError("proof depth constraints currently require dir-1")
        if (self.constraints.x_depth or self.constraints.y_depth) and not (
            self.constraints.require_independent_axes
        ):
            raise ValueError(
                "proof depth constraints currently require independent axes"
            )
        if bool(self.constraints.x_depth) != bool(self.constraints.y_depth):
            raise ValueError(
                "x_depth and y_depth must be specified together for dir-1"
            )
        distractors = self.constraints.distractors
        if distractors.policy is not DistractorPolicy.NONE:
            if not (self.constraints.x_depth and self.constraints.y_depth):
                raise ValueError("distractors currently require proof depth constraints")
            if not self.constraints.require_independent_axes:
                raise ValueError("distractors currently require independent axes")
            if (
                self.constraints.x_depth.minimum != self.constraints.x_depth.maximum
                or self.constraints.y_depth.minimum != self.constraints.y_depth.maximum
            ):
                raise ValueError("distractors currently require exact proof depths")
        if (self.constraints.x_depth or self.constraints.y_depth) and (
            self.relation_mode is RelationMode.DIAGONAL
        ):
            raise ValueError(
                "proof depth constraints currently require cardinal or mixed relations"
            )
        required_entities = 2
        required_relations = 1
        if self.constraints.x_depth and self.constraints.y_depth:
            required_entities = (
                self.constraints.x_depth.maximum
                + self.constraints.y_depth.maximum
            )
            required_relations = (
                self.constraints.x_depth.maximum
                + self.constraints.y_depth.maximum
            )
        elif self.constraints.x_depth:
            required_entities = self.constraints.x_depth.maximum + 1
            required_relations = self.constraints.x_depth.maximum + 1
        elif self.constraints.y_depth:
            required_entities = self.constraints.y_depth.maximum + 1
            required_relations = self.constraints.y_depth.maximum + 1
        if (
            distractors.policy is DistractorPolicy.NONE
            and self.relation_mode is RelationMode.MIXED
            and (
            self.constraints.x_depth or self.constraints.y_depth
            )
        ):
            required_entities += 2
            required_relations += 1
        elif distractors.policy is DistractorPolicy.DISCONNECTED:
            required_entities += distractors.count + 1
            required_relations += distractors.count
        elif distractors.policy is DistractorPolicy.QUERY_BRANCH:
            required_entities += distractors.count
            required_relations += distractors.count
        if self.num_entities < required_entities:
            raise ValueError(
                f"num_entities={self.num_entities} cannot fit the minimum "
                f"proof skeleton ({required_entities})"
            )
        if self.num_relations < required_relations:
            raise ValueError(
                f"num_relations={self.num_relations} cannot fit the minimum "
                f"proof skeleton ({required_relations})"
            )
        if distractors.policy is not DistractorPolicy.NONE and (
            self.num_relations != required_relations
        ):
            raise ValueError(
                "distractor cells require num_relations to equal proof edges "
                "plus distractor count"
            )


@dataclass(frozen=True)
class GenerationCell:
    name: str
    spec: GenerationSpec
    count: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("generation cell name must not be empty")
        if self.count < 0:
            raise ValueError("generation cell count must be non-negative")


@dataclass(frozen=True)
class GeneratedExample:
    messages: tuple[dict[str, str], ...]
    oracle_option: str
    difficulty: dict[str, Any]
    base_semantic_subtype: str
    generator_version: str = "v13.4-global-consistency"

    def to_row(self, *, generation_cell: str | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "messages": [dict(message) for message in self.messages],
            "oracle_option": self.oracle_option,
            "base_semantic_subtype": self.base_semantic_subtype,
            "difficulty": dict(self.difficulty),
            "difficulty_schema_version": 3,
            "generator_version": self.generator_version,
        }
        if generation_cell is not None:
            row["generation_cell"] = generation_cell
        return row


class GenerationError(RuntimeError):
    def __init__(
        self, spec: GenerationSpec, attempts: int, rejections: Counter[str]
    ) -> None:
        self.spec = spec
        self.attempts = attempts
        self.rejections = dict(rejections)
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in rejections.most_common()
        ) or "none recorded"
        super().__init__(
            f"unable to generate {spec.relation_mode.value}/"
            f"{spec.semantic_subtype.value} after {attempts} attempts; "
            f"rejections: {reasons}"
        )


DEFAULT_SYSTEM_PROMPT = (
    "You are an advanced spatial reasoning agent. Process the spatial "
    "relations step-by-step. A cardinal relation updates only its named axis; "
    "a diagonal relation updates both axes. Build an X-axis order from West "
    "to East and a Y-axis order from South to North. Use A < B to mean A is "
    "West/South of B on that axis. A two-axis direction is proven only when "
    "both components are derived.\n\n"
    "Three Type-0 letter rules:\n"
    "1. If exactly one axis is derived and both matching compounds are listed, "
    "return both letters.\n"
    "2. If exactly one axis is derived and only one matching compound is "
    "listed, select Cannot be determined.\n"
    "3. If both orders hold for a pair on an axis, treat that axis as "
    "a globally inconsistent world. Any contradiction anywhere in the X or "
    "Y graph overrides every question type: select Cannot be determined and "
    "do not continue with local query reasoning.\n"
    "If neither axis is derived, select Cannot be determined. If a proven "
    "answer is absent, select None of the Options. End with Answer: followed "
    "by the exact letter set."
)


def _components(direction: str) -> tuple[str | None, str | None]:
    x = "East" if "east" in direction.lower() else "West" if "west" in direction.lower() else None
    y = "North" if "north" in direction.lower() else "South" if "south" in direction.lower() else None
    return x, y


def _relation_for_pair(
    a: str, b: str, coords: dict[str, tuple[int, int]], kind: str, rng: random.Random
) -> Relation:
    ax, ay = coords[a]
    bx, by = coords[b]
    east_west = "East" if ax > bx else "West"
    north_south = "North" if ay > by else "South"
    if kind == "diagonal":
        direction = f"{north_south}{east_west.lower()}"
    elif rng.choice(("x", "y")) == "x":
        direction = east_west
    else:
        direction = north_south
    return Relation(a, Direction(direction), b)


def _explicit_relation(a: str, direction: str, b: str) -> Relation:
    return Relation(a, Direction(direction), b)


def _make_scene(
    rng: random.Random, num_entities: int, num_sentences: int, relation_mode: str
) -> Scene:
    if relation_mode not in {"diagonal", "cardinal", "mixed"}:
        raise ValueError("relation_mode must be diagonal, cardinal, or mixed")
    if not 2 <= num_entities <= len(ENTITY_NAMES):
        raise ValueError(f"num_entities must be in [2, {len(ENTITY_NAMES)}]")
    entities = rng.sample(ENTITY_NAMES, num_entities)
    coords: dict[str, tuple[int, int]] = {}
    for entity in entities:
        while True:
            candidate = (rng.randint(0, 1000), rng.randint(0, 1000))
            if all(candidate[0] != x and candidate[1] != y for x, y in coords.values()):
                coords[entity] = candidate
                break
    pairs = list(itertools.combinations(entities, 2))
    rng.shuffle(pairs)
    pairs = pairs[: min(num_sentences, len(pairs))]
    relations: list[Relation] = []
    for index, (a, b) in enumerate(pairs):
        if relation_mode == "mixed":
            # Guarantee both families whenever at least two statements exist.
            kind = "cardinal" if index == 0 else "diagonal" if index == 1 else rng.choice(("cardinal", "diagonal"))
        else:
            kind = relation_mode
        relations.append(_relation_for_pair(a, b, coords, kind, rng))
    return Scene(tuple(entities), tuple(relations))


def _make_depth_scene(
    spec: GenerationSpec, rng: random.Random
) -> tuple[Scene, str, str]:
    """Build disjoint X/Y proof chains, with safe filler off the query paths."""
    assert spec.constraints.x_depth is not None
    assert spec.constraints.y_depth is not None
    x_depth = spec.constraints.x_depth.choose(rng)
    y_depth = spec.constraints.y_depth.choose(rng)
    entities = list(rng.sample(ENTITY_NAMES, spec.num_entities))
    reference, target = entities[0], entities[1]
    cursor = 2
    x_internal = entities[cursor : cursor + x_depth - 1]
    cursor += x_depth - 1
    y_internal = entities[cursor : cursor + y_depth - 1]
    cursor += y_depth - 1
    extras = entities[cursor:]

    x_direction = rng.choice((Direction.EAST, Direction.WEST))
    y_direction = rng.choice((Direction.NORTH, Direction.SOUTH))
    relations: list[Relation] = []
    x_nodes = [reference, *x_internal, target]
    y_nodes = [reference, *y_internal, target]
    relations.extend(
        Relation(subject=right, direction=x_direction, reference=left)
        for left, right in zip(x_nodes, x_nodes[1:])
    )
    relations.extend(
        Relation(subject=right, direction=y_direction, reference=left)
        for left, right in zip(y_nodes, y_nodes[1:])
    )

    distractors = spec.constraints.distractors
    if distractors.policy is DistractorPolicy.DISCONNECTED:
        distractor_nodes = extras[: distractors.count + 1]
        for index, (left, right) in enumerate(
            zip(distractor_nodes, distractor_nodes[1:])
        ):
            direction = (
                Direction.NORTHEAST
                if spec.relation_mode is RelationMode.MIXED and index == 0
                else rng.choice(
                    (Direction.EAST, Direction.WEST, Direction.NORTH, Direction.SOUTH)
                )
            )
            relations.append(Relation(right, direction, left))
    elif distractors.policy is DistractorPolicy.QUERY_BRANCH:
        distractor_nodes = extras[: distractors.count]
        for index, node in enumerate(distractor_nodes):
            direction = (
                Direction.NORTHEAST
                if spec.relation_mode is RelationMode.MIXED and index == 0
                else rng.choice(
                    (Direction.EAST, Direction.WEST, Direction.NORTH, Direction.SOUTH)
                )
            )
            relations.append(Relation(node, direction, reference))
    elif spec.relation_mode is RelationMode.MIXED:
        # Mixed depth controls need diagonal evidence, but it stays in an
        # irrelevant component so it cannot collapse either requested path.
        relations.append(Relation(extras[1], Direction.NORTHEAST, extras[0]))

    # Cardinal filler is confined to the opposite axis's internal nodes and
    # unrelated entities. It increases prompt size without introducing a path
    # from the query reference to target on the protected axis.
    filler: list[Relation] = []
    if distractors.policy is DistractorPolicy.NONE:
        for left, right in itertools.combinations([*y_internal, *extras], 2):
            filler.append(Relation(right, Direction.EAST, left))
        for left, right in itertools.combinations([*x_internal, *extras], 2):
            filler.append(Relation(right, Direction.NORTH, left))
    rng.shuffle(filler)
    used = {(r.subject, r.direction, r.reference) for r in relations}
    for relation in filler:
        key = (relation.subject, relation.direction, relation.reference)
        if key in used:
            continue
        relations.append(relation)
        used.add(key)
        if len(relations) >= spec.num_relations:
            break
    if len(relations) < spec.num_relations:
        raise ValueError(
            f"num_relations={spec.num_relations} cannot be filled without "
            "touching the protected proof paths"
        )
    rng.shuffle(relations)
    return Scene(tuple(entities), tuple(relations)), target, reference


def _cycle_relation_count(mode: RelationMode, cycle: CycleSpec) -> int:
    if cycle.axes is CycleAxes.BOTH and mode is RelationMode.CARDINAL:
        return cycle.length * 2
    return cycle.length


def _cycle_relations(
    nodes: Sequence[str], mode: RelationMode, cycle: CycleSpec
) -> list[Relation]:
    pairs = (
        list(zip(nodes, [*nodes[1:], nodes[0]]))
        if cycle.world_consistency is WorldConsistency.INCONSISTENT
        else list(zip(nodes, nodes[1:]))
    )
    if cycle.axes is CycleAxes.X:
        return [Relation(source, Direction.EAST, target) for source, target in pairs]
    if cycle.axes is CycleAxes.Y:
        return [Relation(source, Direction.NORTH, target) for source, target in pairs]
    if mode is RelationMode.CARDINAL:
        return [
            *[Relation(source, Direction.EAST, target) for source, target in pairs],
            *[Relation(source, Direction.NORTH, target) for source, target in pairs],
        ]
    return [
        Relation(source, Direction.NORTHEAST, target)
        for source, target in pairs
    ]


def _inject_cycle(
    prompt: str,
    base_objects: Sequence[str],
    anchor: str,
    mode: RelationMode,
    cycle: CycleSpec,
    rng: random.Random,
) -> str:
    fresh = [name for name in ENTITY_NAMES if name not in set(base_objects)]
    cycle_node_count = cycle.length + (
        1 if cycle.world_consistency is WorldConsistency.CONSISTENT else 0
    )
    needed = (
        cycle_node_count
        if cycle.placement is CyclePlacement.DISCONNECTED
        else cycle_node_count - 1
    )
    if len(fresh) < needed:
        raise ValueError("not enough fresh entities to construct the requested cycle")
    selected = rng.sample(fresh, needed)
    nodes = (
        selected
        if cycle.placement is CyclePlacement.DISCONNECTED
        else [anchor, *selected]
    )
    additions = _cycle_relations(nodes, mode, cycle)
    map_part, question_part = prompt.split("\n\nQuestion:", 1)
    # Every globally inconsistent question uses the same refusal option.
    question_part = question_part.replace(NONE, SPECIAL)
    return (
        map_part.rstrip()
        + " "
        + " ".join(relation.render() for relation in additions)
        + "\n\nQuestion:"
        + question_part
    )


def _graphs(entities: Sequence[str], relations: Sequence[Relation]) -> tuple[AxisGraph, AxisGraph]:
    x_graph, y_graph = AxisGraph(), AxisGraph()
    x_graph.nodes.update(entities)
    y_graph.nodes.update(entities)
    for relation in relations:
        a, b = relation.subject, relation.reference
        if relation.x_component == "East":
            x_graph.add_relation(b, a)
        elif relation.x_component == "West":
            x_graph.add_relation(a, b)
        if relation.y_component == "North":
            y_graph.add_relation(b, a)
        elif relation.y_component == "South":
            y_graph.add_relation(a, b)
    return x_graph, y_graph


def _status(closure: set[tuple[str, str]], target: str, reference: str) -> str | None:
    target_low = (target, reference) in closure
    target_high = (reference, target) in closure
    if target_low == target_high:
        return None
    return "low" if target_low else "high"


def _options_text(options: list[str]) -> str:
    return ", ".join(f"{letter}. {value}" for letter, value in zip("ABCDE", options, strict=True))


def _direction_prompt(
    entities: Sequence[str],
    relations: Sequence[Relation],
    x_graph: AxisGraph,
    y_graph: AxisGraph,
    target_num_answers: int | None,
    rng: random.Random,
    subtype: str | None = None,
    forced_pair: tuple[str, str] | None = None,
) -> str | None:
    x_closure, y_closure = x_graph.get_transitive_closure(), y_graph.get_transitive_closure()
    candidates: list[tuple[str, str]] = []
    for target, reference in itertools.permutations(entities, 2):
        x_known = _status(x_closure, target, reference) is not None
        y_known = _status(y_closure, target, reference) is not None
        answer_count = 1 if x_known and y_known else 2 if x_known or y_known else 0
        if target_num_answers is None or answer_count == target_num_answers:
            candidates.append((target, reference))
    if forced_pair is not None:
        if forced_pair not in candidates:
            return None
        target, reference = forced_pair
    elif not candidates:
        return None
    else:
        target, reference = rng.choice(candidates)
    x_state = _status(x_closure, target, reference)
    y_state = _status(y_closure, target, reference)
    x_comp = "East" if x_state == "high" else "West" if x_state == "low" else None
    y_comp = "North" if y_state == "high" else "South" if y_state == "low" else None
    xs = [x_comp] if x_comp else ["East", "West"]
    ys = [y_comp] if y_comp else ["North", "South"]
    possible = [f"{y}{x.lower()}" for y in ys for x in xs]
    if subtype == "dir-incomplete":
        if len(possible) != 2:
            return None
        kept = rng.choice(possible)
        pool = [direction for direction in DIRECTIONS if direction not in possible]
        options = [kept, *rng.sample(pool, 3), SPECIAL]
    elif subtype == "dir-omit":
        pool = [direction for direction in DIRECTIONS if direction not in possible]
        if len(pool) < 4:
            return None
        options = [*rng.sample(pool, 4), NONE]
    else:
        options = [*COMPOUNDS, SPECIAL]
    rng.shuffle(options)
    return (
        "Consider a map with multiple locations:\n\n"
        + " ".join(relation.render() for relation in relations)
        + f"\n\nQuestion: In which direction is the {target} relative to the {reference}? "
        + f"Available options: {_options_text(options)}"
    )


def _entities_in_direction(
    entities: Sequence[str],
    reference: str,
    direction: str,
    x_closure: set[tuple[str, str]],
    y_closure: set[tuple[str, str]],
) -> list[str]:
    need_x, need_y = _components(direction)
    found: list[str] = []
    for entity in entities:
        if entity == reference:
            continue
        x_status = _status(x_closure, entity, reference)
        y_status = _status(y_closure, entity, reference)
        x_ok = need_x is None or x_status == ("high" if need_x == "East" else "low")
        y_ok = need_y is None or y_status == ("high" if need_y == "North" else "low")
        if x_ok and y_ok:
            found.append(entity)
    return sorted(found)


def _which_prompt(
    entities: Sequence[str],
    relations: Sequence[Relation],
    x_graph: AxisGraph,
    y_graph: AxisGraph,
    target_num_answers: int | None,
    rng: random.Random,
) -> str | None:
    x_closure, y_closure = x_graph.get_transitive_closure(), y_graph.get_transitive_closure()
    desired = 1 if target_num_answers is None else target_num_answers
    candidates = []
    for reference in entities:
        for direction in DIRECTIONS:
            correct = _entities_in_direction(entities, reference, direction, x_closure, y_closure)
            incorrect = [entity for entity in entities if entity != reference and entity not in correct]
            if desired == 0 and len(incorrect) >= 4:
                candidates.append((reference, direction, correct, incorrect))
            elif desired > 0 and len(correct) >= desired and len(incorrect) >= 4 - desired:
                candidates.append((reference, direction, correct, incorrect))
    if not candidates:
        return None
    reference, direction, correct, incorrect = rng.choice(candidates)
    if desired == 0:
        choices = rng.sample(incorrect, 4)
    else:
        choices = rng.sample(correct, desired) + rng.sample(incorrect, 4 - desired)
        rng.shuffle(choices)
    options = choices + [NONE]
    return (
        "Consider a map with multiple locations:\n\n"
        + " ".join(relation.render() for relation in relations)
        + f"\n\nQuestion: Which object is in the {direction} of the {reference}? "
        + f"Available options: {_options_text(options)}"
    )


def _count_prompt(
    entities: Sequence[str],
    relations: Sequence[Relation],
    x_graph: AxisGraph,
    y_graph: AxisGraph,
    target_num_answers: int | None,
    rng: random.Random,
) -> str:
    x_closure, y_closure = x_graph.get_transitive_closure(), y_graph.get_transitive_closure()
    reference = rng.choice(entities)
    direction = rng.choice(DIRECTIONS)
    count = len(_entities_in_direction(entities, reference, direction, x_closure, y_closure))
    omit = target_num_answers == 0
    values: list[int] = [] if omit else [count]
    wrong = [value for value in range(len(entities) + 3) if value != count]
    rng.shuffle(wrong)
    values.extend(wrong[: 4 - len(values)])
    rng.shuffle(values)
    options = [str(value) for value in values] + [NONE]
    return (
        "Consider a map with multiple locations:\n\n"
        + " ".join(relation.render() for relation in relations)
        + f"\n\nQuestion: How many objects are in the {direction} of the {reference}? "
        + f"Available options: {_options_text(options)}"
    )


def _axis_conclusion(axis: str, relation: str, target: str, reference: str) -> str:
    labels = {
        ("X", "gt"): "East",
        ("X", "lt"): "West",
        ("Y", "gt"): "North",
        ("Y", "lt"): "South",
    }
    direction = labels.get((axis, relation))
    if direction:
        return f"{target} is {direction} of {reference}."
    return f"{target}'s {axis}-axis relation to {reference} is unknown."


def _query_proof_lines(solved: SolvedProblem) -> list[str]:
    structure = solved.structure
    if solved.grade.q_type in (1, 2):
        reference = structure["query_reference"]
        direction = structure["query_direction"]
        proven_entities = structure["proven_entities"]
        lines = [
            f"**Reference**: {reference}",
            f"**Direction Query**: {direction}",
        ]
        for proof in structure["proven_entity_proofs"]:
            for axis in structure["required_axes"]:
                path = proof[f"{axis}_path"]
                lines.append(
                    f"**{proof['entity']} {axis.upper()}-Proof**: "
                    f"{' < '.join(path)}"
                )
        rendered_entities = ", ".join(proven_entities) if proven_entities else "none"
        lines.append(f"**Proven Entities**: {rendered_entities}")
        if solved.grade.q_type == 2:
            lines.append(f"**Count**: {len(proven_entities)}")
        return lines
    if solved.grade.q_type != 0:
        return []
    target = structure["target"]
    reference = structure["reference"]
    lines = [f"**Target**: {target}", f"**Reference**: {reference}"]
    components: list[str] = []
    for axis in ("X", "Y"):
        key = axis.lower()
        path = structure[f"{key}_path"]
        conflict = structure[f"{key}_conflict"]
        relation = getattr(solved.grade, f"{key}_rel")
        if path:
            lines.append(f"**{axis}-Proof**: {' < '.join(path)}")
        elif conflict:
            lines.append(f"**{axis}-Proof**: none (contradictory ordering paths)")
        else:
            lines.append(f"**{axis}-Proof**: none (no ordering path)")
        conclusion = _axis_conclusion(axis, relation, target, reference)
        lines.append(f"**{axis}-Conclusion**: {conclusion}")
        if relation != "unknown":
            components.append(
                {
                    ("X", "gt"): "East",
                    ("X", "lt"): "West",
                    ("Y", "gt"): "North",
                    ("Y", "lt"): "South",
                }[(axis, relation)]
            )
    if len(components) == 2:
        x_component = components[0]
        y_component = components[1]
        compound = f"{y_component}{x_component.lower()}"
        lines.append(
            f"**Composition**: {x_component} + {y_component} = {compound}."
        )
    elif components:
        lines.append(
            f"**Composition**: only {components[0]} is proven; the other axis "
            "remains unknown."
        )
    else:
        lines.append("**Composition**: neither axis yields a unique compound.")
    return lines


def _consistency_lines(solved: SolvedProblem) -> list[str]:
    lines: list[str] = []
    for axis in solved.structure["cycle_axes"]:
        witness = list(reversed(solved.structure["cycle_witnesses"][axis]))
        lines.append(f"**{axis.upper()}-Cycle**: {' < '.join(witness)}")
    lines.append(
        "**Conclusion**: The complete map is inconsistent, so the requested "
        "answer cannot be determined reliably."
    )
    return lines


def _trace(solved: SolvedProblem) -> str:
    entities = solved.objects
    relations = tuple(
        Relation(subject=a, direction=Direction(direction.title()), reference=b)
        for a, direction, b in solved.relations
    )
    grade = solved.grade
    x_graph, y_graph = AxisGraph(), AxisGraph()
    x_graph.nodes.update(entities)
    y_graph.nodes.update(entities)
    active: set[str] = set()
    chunks = [
        "### Initialization\n"
        f"**Entities Detected**: {', '.join(sorted(entities))}\n"
        "**Initial X-State**: Empty\n"
        "**Initial Y-State**: Empty"
    ]
    for index, relation in enumerate(relations, 1):
        a, b = relation.subject, relation.reference
        active.update((a, b))
        extraction: list[str] = []
        if relation.x_component == "East":
            x_graph.add_relation(b, a)
            extraction.append(f"**X-Extraction**: {b} < {a}")
        elif relation.x_component == "West":
            x_graph.add_relation(a, b)
            extraction.append(f"**X-Extraction**: {a} < {b}")
        else:
            extraction.append("**X-Extraction**: none (cardinal Y-only relation)")
        if relation.y_component == "North":
            y_graph.add_relation(b, a)
            extraction.append(f"**Y-Extraction**: {b} < {a}")
        elif relation.y_component == "South":
            y_graph.add_relation(a, b)
            extraction.append(f"**Y-Extraction**: {a} < {b}")
        else:
            extraction.append("**Y-Extraction**: none (cardinal X-only relation)")
        chunks.append(
            f"### Step {index}\n**Sentence**: \"{relation.render()}\"\n"
            + "\n".join(extraction)
            + f"\n**X-State**: {x_graph.format_state(active)}"
            + f"\n**Y-State**: {y_graph.format_state(active)}"
        )
    if solved.structure["world_consistency"] == "inconsistent":
        verdicts = [
            "### Consistency Check",
            *_consistency_lines(solved),
            "**Options**:",
        ]
    else:
        verdicts = [
            "### Final Deduction",
            *_query_proof_lines(solved),
            "**Options**:",
        ]
    for letter, value, accepted, reason in grade.verdicts:
        verdicts.append(f"- {letter}. {value} — {reason}. {'In' if accepted else 'Out'}.")
    return "<think>\n" + "\n\n".join(chunks + ["\n".join(verdicts)]) + f"\n</think>\nAnswer: {grade.pretty}"


class SpatialGenerator:
    """Generate solver-verified spatial examples from one coherent spec."""

    def __init__(self, solver: SpatialSolverV13 | None = None) -> None:
        self.solver = solver or SpatialSolverV13()

    def generate(
        self, spec: GenerationSpec, rng: random.Random
    ) -> GeneratedExample:
        policy = _SUBTYPE_POLICIES[spec.semantic_subtype]
        question_type = policy.question_type
        target_num_answers = policy.target_num_answers
        subtype = spec.semantic_subtype.value
        relation_mode = spec.relation_mode.value
        rejections: Counter[str] = Counter()
        for _ in range(spec.max_attempts):
            forced_pair = None
            if spec.constraints.x_depth and spec.constraints.y_depth:
                scene, target, reference = _make_depth_scene(spec, rng)
                forced_pair = (target, reference)
            else:
                cycle_relation_count = (
                    _cycle_relation_count(spec.relation_mode, spec.cycle)
                    if spec.cycle
                    else 0
                )
                fresh_cycle_entities = (
                    (
                        spec.cycle.length
                        + (
                            1
                            if spec.cycle.world_consistency
                            is WorldConsistency.CONSISTENT
                            else 0
                        )
                    )
                    - (
                        1
                        if spec.cycle.placement
                        is CyclePlacement.QUERY_CONNECTED
                        else 0
                    )
                    if spec.cycle else 0
                )
                base_num_entities = spec.num_entities - fresh_cycle_entities
                base_num_relations = spec.num_relations - cycle_relation_count
                if base_num_entities < 2 or base_num_relations < 1:
                    rejections["cycle_does_not_fit"] += 1
                    continue
                scene = _make_scene(
                    rng, base_num_entities, base_num_relations, relation_mode
                )
            entities = list(scene.entities)
            relations = list(scene.relations)
            if spec.semantic_subtype is SemanticSubtype.DIR_CYCLE:
                target, reference = entities[0], entities[1]
                if spec.relation_mode is RelationMode.CARDINAL:
                    relations.extend(
                        [
                            _explicit_relation(target, "East", reference),
                            _explicit_relation(reference, "East", target),
                            _explicit_relation(target, "North", reference),
                            _explicit_relation(reference, "North", target),
                        ]
                    )
                else:
                    relations.extend(
                        [
                            _explicit_relation(target, "Northeast", reference),
                            _explicit_relation(reference, "Northeast", target),
                        ]
                    )
            x_graph, y_graph = _graphs(entities, relations)
            if question_type == 0:
                if spec.semantic_subtype is SemanticSubtype.DIR_CYCLE:
                    options = [*COMPOUNDS, SPECIAL]
                    rng.shuffle(options)
                    user_prompt = (
                        "Consider a map with multiple locations:\n\n"
                        + " ".join(relation.render() for relation in relations)
                        + f"\n\nQuestion: In which direction is the {target} relative to the {reference}? "
                        + f"Available options: {_options_text(options)}"
                    )
                else:
                    user_prompt = _direction_prompt(
                        entities, relations, x_graph, y_graph,
                        target_num_answers, rng, subtype, forced_pair,
                    )
            elif question_type == 1:
                user_prompt = _which_prompt(
                    entities, relations, x_graph, y_graph,
                    target_num_answers, rng,
                )
            else:
                user_prompt = _count_prompt(
                    entities, relations, x_graph, y_graph,
                    target_num_answers, rng,
                )
            if not user_prompt:
                rejections["no_qualifying_query"] += 1
                continue
            base_solved = self.solver.solve_and_analyze(user_prompt)
            if spec.cycle is not None:
                if base_solved.structure["world_consistency"] != "consistent":
                    rejections["base_world_inconsistent"] += 1
                    continue
                if base_solved.structure["semantic_subtype"] != subtype:
                    rejections["wrong_base_semantic_subtype"] += 1
                    continue
            if spec.cycle is not None:
                anchor = (
                    base_solved.structure["reference"]
                    if question_type == 0
                    else base_solved.structure["query_reference"]
                )
                user_prompt = _inject_cycle(
                    user_prompt,
                    base_solved.objects,
                    anchor,
                    spec.relation_mode,
                    spec.cycle,
                    rng,
                )
            solved = self.solver.solve_and_analyze(user_prompt)
            grade = solved.grade
            difficulty = solved.structure
            if not grade.accept or len(grade.options) != 5:
                rejections["solver_rejected"] += 1
                continue
            expected_subtype = (
                {0: "dir-cycle", 1: "which-cycle", 2: "count-cycle"}[
                    question_type
                ]
                if spec.cycle
                and spec.cycle.world_consistency is WorldConsistency.INCONSISTENT
                else subtype
            )
            if difficulty["semantic_subtype"] != expected_subtype:
                rejections["wrong_semantic_subtype"] += 1
                continue
            expected_mix = {
                RelationMode.DIAGONAL: "diagonal-only",
                RelationMode.CARDINAL: "cardinal-only",
                RelationMode.MIXED: "mixed",
            }[spec.relation_mode]
            if difficulty["relation_mix"] != expected_mix:
                rejections["wrong_relation_mode"] += 1
                continue
            if spec.cycle is not None:
                expected_axes = (
                    []
                    if spec.cycle.world_consistency is WorldConsistency.CONSISTENT
                    else ["x", "y"]
                    if spec.cycle.axes is CycleAxes.BOTH
                    else [spec.cycle.axes.value]
                )
                if difficulty["cycle_axes"] != expected_axes:
                    rejections["wrong_cycle_axes"] += 1
                    continue
                if spec.cycle.world_consistency is WorldConsistency.INCONSISTENT:
                    if difficulty["cycle_topology"] != spec.cycle.topology.value:
                        rejections["wrong_cycle_topology"] += 1
                        continue
                    if difficulty["cycle_placement"] != spec.cycle.placement.value:
                        rejections["wrong_cycle_placement"] += 1
                        continue
                    if any(
                        length != spec.cycle.length
                        for length in difficulty["cycle_lengths"].values()
                    ):
                        rejections["wrong_cycle_length"] += 1
                        continue
            if (
                spec.constraints.require_independent_axes
                and difficulty["axes_independent"] is not True
            ):
                rejections["axes_not_independent"] += 1
                continue
            if (
                spec.constraints.x_depth
                and not spec.constraints.x_depth.contains(difficulty["x_depth"])
            ):
                rejections["x_depth_out_of_range"] += 1
                continue
            if (
                spec.constraints.y_depth
                and not spec.constraints.y_depth.contains(difficulty["y_depth"])
            ):
                rejections["y_depth_out_of_range"] += 1
                continue
            distractors = spec.constraints.distractors
            if distractors.policy is not DistractorPolicy.NONE:
                if difficulty["num_distractor_relations"] != distractors.count:
                    rejections["wrong_distractor_count"] += 1
                    continue
                policy_key = {
                    DistractorPolicy.DISCONNECTED: "num_disconnected_distractors",
                    DistractorPolicy.QUERY_BRANCH: "num_query_branch_distractors",
                }[distractors.policy]
                if difficulty[policy_key] != distractors.count:
                    rejections["wrong_distractor_topology"] += 1
                    continue
            return GeneratedExample(
                messages=(
                    {
                        "role": "system",
                        "content": spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "assistant",
                        "content": _trace(solved),
                    },
                ),
                oracle_option=grade.raw,
                difficulty=difficulty,
                base_semantic_subtype=subtype,
            )
        raise GenerationError(spec, spec.max_attempts, rejections)


def generate_dataset(
    cells: Sequence[GenerationCell],
    *,
    output_file: str | Path,
    test_fraction: float,
    seed: int,
    generator: SpatialGenerator | None = None,
) -> tuple[Path, Path]:
    """Generate explicit cells and split each cell independently."""
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    names = [cell.name for cell in cells]
    if len(names) != len(set(names)):
        raise ValueError("generation cell names must be unique")
    engine = generator or SpatialGenerator()
    seed_rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for cell in cells:
        rows = [
            engine.generate(cell.spec, random.Random(seed_rng.randrange(2**63))).to_row(
                generation_cell=cell.name
            )
            for _ in range(cell.count)
        ]
        seed_rng.shuffle(rows)
        test_count = int(len(rows) * test_fraction)
        split_at = len(rows) - test_count
        train_rows.extend(rows[:split_at])
        test_rows.extend(rows[split_at:])
    seed_rng.shuffle(train_rows)
    seed_rng.shuffle(test_rows)
    base = Path(output_file)
    train_path = base.with_name(base.stem + "_train.jsonl")
    test_path = base.with_name(base.stem + "_test.jsonl")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, subset in ((train_path, train_rows), (test_path, test_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_path, test_path
