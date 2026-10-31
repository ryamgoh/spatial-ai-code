"""Explicit controlled construction parameters for a fixed coverage experiment."""

import random
from dataclasses import dataclass, replace
from functools import lru_cache

from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_pilot import Story
from spatial.v2.entailment_controlled_prototype import (
    CASES,
    KINDS,
)
from spatial.v2.entailment_controlled_prototype import (
    _core as original_core,
)
from spatial.v2.entailment_names import NAMES, rename_structure, story_name_map
from spatial.v2.entailment_scene_protocol import all_scene_problem
from spatial.v2.solver import And, Direction, direction_constraint

VERSION = "all-controlled-coverage-v2"
CAP = 968


@dataclass(frozen=True)
class Slot:
    span: int
    direction: Direction
    kind: str
    case: str
    count_value: int = 1
    variant: int = 0
    compact: bool = False
    extra_size: int = 0

    @property
    def seed(self):
        return 2033103100 + self.span * 8 + tuple(Direction).index(self.direction)

    @property
    def family(self):
        # Seed labels are conservative variant groups, not canonical families.
        return f"{VERSION}/seed-{self.seed}"

    @property
    def identifier(self):
        return (
            f"{VERSION}-s{self.span}-{self.direction.value}-{self.kind}-{self.case}"
            f"-c{self.count_value}-v{self.variant}-compact{int(self.compact)}"
            f"-z{self.extra_size}"
        )

    @property
    def requested_depth(self):
        return None if self.case in ("no-entailed", "inconsistent") else self.span

    @property
    def size_group(self):
        return (self.span, self.direction.value, self.compact, self.extra_size)


@lru_cache(maxsize=1)
def slots():
    result = []
    for span in range(1, 11):
        for i, direction in enumerate(Direction):
            case = "unlisted" if (span + i) % 2 else "listed"
            for kind in ("direction", "which"):
                result.append(Slot(span, direction, kind, case))
            result.append(Slot(span, direction, "which", "multiple"))
            for count in range(5):
                result.append(Slot(span, direction, "count", case, count))
        for variant in (0, 1):
            direction = tuple(Direction)[(span + 3 * variant) % 8]
            for kind in KINDS:
                for case in ("no-entailed", "inconsistent"):
                    result.append(Slot(span, direction, kind, case, variant=variant))
    for i, direction in enumerate(Direction):
        for extra in (0, 3):
            for kind in KINDS:
                for case in (*CASES, *(("multiple",) if kind == "which" else ())):
                    result.append(
                        Slot(
                            1,
                            direction,
                            kind,
                            case,
                            count_value=i % 5,
                            variant=(i + extra) % 2,
                            compact=True,
                            extra_size=extra,
                        )
                    )
    if len(result) != CAP or len({s.identifier for s in result}) != CAP:
        raise ValueError("fixed coverage inventory changed")
    return tuple(result)


def _opposite(direction):
    return tuple(Direction)[(tuple(Direction).index(direction) + 4) % 8]


def _compact_core(slot):
    direction, opposite = slot.direction, _opposite(slot.direction)
    objects = ("O", *(f"C{i}" for i in range(5)))
    atoms = []

    def add(a, b, relation):
        atoms.append(direction_constraint(a, b, relation))

    if slot.case == "no-entailed":
        if slot.variant == 0:
            diagonal = tuple(Direction)[
                (tuple(Direction).index(direction) // 2) * 2 + 1
            ]
            add("C0", "C1", diagonal)
            add("O", "C1", diagonal)
            for node in objects[3:]:
                add(node, "O", opposite)
            add("C2", "C3", opposite)
        else:
            # One component can translate relative to the reference component.
            add("C0", "C1", direction)
            add("C1", "C2", direction)
            add("C0", "C2", direction)
            add("C3", "O", opposite)
            add("C4", "O", opposite)
            add("C4", "C3", opposite)
    elif slot.case == "inconsistent":
        if slot.variant == 0:
            add("C0", "O", direction)
            add("O", "C0", direction)
            for node in objects[2:]:
                add(node, "O", opposite)
        else:
            add("C0", "O", direction)
            add("C1", "C0", direction)
            add("O", "C1", direction)
            for node in objects[3:]:
                add(node, "O", opposite)
    else:
        matches = (
            2 + slot.span % 3
            if slot.case == "multiple"
            else slot.count_value
            if slot.kind == "count"
            else 1
        )
        for i in range(5):
            add(f"C{i}", "O", direction if i < matches else opposite)
        # Link two objects already on the same side, without changing membership.
        a, b = (0, 1) if matches >= 2 else (3, 4)
        add(f"C{a}", f"C{b}", direction if matches >= 2 else opposite)
    return Story(
        slot.identifier,
        slot.family,
        VERSION,
        objects,
        And(tuple(atoms)),
        "C0",
        "O",
        objects[1:],
        direction,
        (),
    )


def _core(slot):
    if slot.compact:
        return _compact_core(slot)
    if slot.case not in ("no-entailed", "inconsistent") or slot.variant == 0:
        return replace(original_core(slot), recipe=VERSION)
    # Start with the same excluded support; change the E mechanism only.
    core = original_core(slot)
    removed = (
        {"Unknown", "Anchor"} if slot.case == "no-entailed" else {"CycleA", "CycleB"}
    )
    atoms = [
        a
        for a in core.premise.operands
        if a.subject not in removed and a.reference not in removed
    ]
    objects = [o for o in core.objects if o not in removed]
    if slot.case == "no-entailed":
        objects.extend(("Unknown", "FreeAnchor"))
        atoms.append(direction_constraint("Unknown", "FreeAnchor", slot.direction))
        target = "Unknown"
    else:
        # A strict cycle with variable length, rather than a fixed local pair.
        nodes = ["O", *(f"Conflict{i}" for i in range(1 + slot.span % 4))]
        objects.extend(nodes[1:])
        atoms.extend(
            direction_constraint(a, b, slot.direction)
            for a, b in zip(nodes, [*nodes[1:], nodes[0]])
        )
        target = "C0"
    return replace(
        core,
        objects=tuple(objects),
        premise=And(tuple(atoms)),
        target=target,
        recipe=VERSION,
    )


@lru_cache(maxsize=128)
def _deep_size(span, direction):
    peers = [
        _core(Slot(span, direction, kind, case, count, variant))
        for kind in KINDS
        for case in (*CASES, *(("multiple",) if kind == "which" else ()))
        for count in (range(5) if kind == "count" else (1,))
        for variant in (0, 1)
    ]
    objects = max(len(s.objects) for s in peers) + 4
    premises = objects + max(len(s.premise.operands) - len(s.objects) for s in peers)
    return objects, premises


def size_target(slot):
    return (
        (6 + slot.extra_size, 6 + slot.extra_size)
        if slot.compact
        else _deep_size(slot.span, slot.direction)
    )


def construct(slot):
    if (
        slot.kind not in KINDS
        or slot.case not in (*CASES, "multiple")
        or slot.case == "multiple"
        and slot.kind != "which"
        or not 1 <= slot.span <= 10
        or slot.direction not in tuple(Direction)
        or not 0 <= slot.count_value <= 4
        or slot.variant not in (0, 1)
        or slot.extra_size not in (0, 3)
        or slot.compact
        and slot.span != 1
        or not slot.compact
        and slot.extra_size != 0
    ):
        raise ValueError("unsupported controlled coverage parameters")
    core = _core(slot)
    object_target, premise_target = size_target(slot)
    if object_target > len(NAMES):
        raise ValueError("ordinary-name vocabulary exceeded")
    rng = random.Random(stable_int(slot.identifier))
    opposite = _opposite(slot.direction)
    contexts = [f"Context{i}" for i in range(object_target - len(core.objects))]
    atoms = list(core.premise.operands)
    edges = set()
    lengths = {"O": 0}
    for i, node in enumerate(contexts):
        parents = [p for p in ["O", *contexts[:i]] if lengths[p] < slot.span]
        parent = rng.choice(parents)
        atoms.append(direction_constraint(node, parent, opposite))
        edges.add((node, parent))
        lengths[node] = lengths[parent] + 1
    pairs = [
        (node, parent)
        for i, node in enumerate(contexts)
        for parent in ["O", *contexts[:i]]
        if (node, parent) not in edges
    ]
    rng.shuffle(pairs)
    missing = premise_target - len(atoms)
    if not 0 <= missing <= len(pairs):
        raise ValueError("size matching has insufficient acyclic context edges")
    atoms.extend(direction_constraint(a, b, opposite) for a, b in pairs[:missing])
    atoms = [
        direction_constraint(a.reference, a.subject, _opposite(next(iter(a.allowed))))
        if rng.getrandbits(1)
        else a
        for a in atoms
    ]
    rng.shuffle(atoms)
    story = replace(core, objects=(*core.objects, *contexts), premise=And(tuple(atoms)))
    story = rename_structure(story, story_name_map(story.objects, slot.identifier))
    return story, all_scene_problem(story.problems()[KINDS.index(slot.kind)])
