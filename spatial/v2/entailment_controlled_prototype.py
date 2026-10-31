"""Bounded all-controlled feasibility recipes, not a production sampling policy."""

import random
from dataclasses import dataclass, replace

from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_pilot import Story
from spatial.v2.composition_scene_sampler import _support
from spatial.v2.entailment_names import NAMES, rename_structure, story_name_map
from spatial.v2.entailment_scene_protocol import all_scene_problem
from spatial.v2.solver import And, Direction, direction_constraint

VERSION = "all-controlled-prototype-v1"
KINDS = ("direction", "which", "count")
CASES = ("listed", "unlisted", "no-entailed", "inconsistent")


@dataclass(frozen=True)
class Slot:
    span: int
    extra_size: int
    kind: str
    case: str

    @property
    def seed(self):
        return 2032103100 + self.span

    @property
    def direction(self):
        return tuple(Direction)[(self.span - 1) % 8]

    @property
    def count_value(self):
        return (self.span - 1) % 5

    @property
    def family(self):
        # All size, query, and answer variants must stay in the same group.
        return f"{VERSION}/seed-{self.seed}"

    @property
    def identifier(self):
        return f"{VERSION}-s{self.span}-z{self.extra_size}-{self.kind}-{self.case}"

    @property
    def requested_depth(self):
        return None if self.case in ("no-entailed", "inconsistent") else self.span


def slots():
    return tuple(
        Slot(span, extra, kind, case)
        for span in range(1, 11)
        for extra in (0, 3)
        for kind in KINDS
        for case in (*CASES, *(("multiple",) if kind == "which" else ()))
    )


def _core(slot):
    direction, depth = slot.direction, slot.span
    directions = tuple(Direction)
    index = directions.index(direction)
    opposite = directions[(index + 4) % 8]
    matches = (
        0
        if slot.case in ("no-entailed", "inconsistent")
        else 2 + slot.span % 3
        if slot.case == "multiple"
        else slot.count_value
        if slot.kind == "count"
        else 1
    )
    candidates = tuple(f"C{i}" for i in range(5))
    target_direction = direction if matches else opposite
    exclude = depth > 1 and matches == 0 and index % 2 == 1
    links, support = _support(
        depth,
        direction if exclude else target_direction,
        "homogeneous" if depth == 1 else "axis-split",
        exclude=exclude,
    )
    if depth > 1 and matches > 0 and index % 2 == 0:
        # Keep the equality-path intermediate outside the all-scene answer set.
        support = [
            direction_constraint(
                a.subject,
                a.reference,
                opposite if a.subject == "AxisLink0" else direction,
            )
            if a.subject == "AxisLink0" or a.reference == "AxisLink0"
            else a
            for a in support
        ]
    atoms = list(support)
    for candidate in candidates[1:]:
        if candidate in candidates[:matches]:
            atoms.extend(
                replace(a, subject=candidate) for a in support if a.subject == "C0"
            )
        else:
            atoms.append(direction_constraint(candidate, "O", opposite))
    objects = ["O", *candidates, *links]
    target = "C0"
    if slot.case == "no-entailed":
        # The two endpoints above a shared diagonal anchor are unordered.
        # Every old object is excluded; the fresh target can qualify or not.
        diagonal = directions[(index // 2) * 2 + 1]
        objects.extend(("Unknown", "Anchor"))
        atoms.extend(
            (
                direction_constraint("Unknown", "Anchor", diagonal),
                direction_constraint("O", "Anchor", diagonal),
            )
        )
        target = "Unknown"
    elif slot.case == "inconsistent":
        objects.extend(("CycleA", "CycleB"))
        atoms.extend(
            (
                direction_constraint("CycleA", "O", direction),
                direction_constraint("CycleB", "CycleA", direction),
                direction_constraint("O", "CycleB", direction),
            )
        )
    return Story(
        slot.identifier,
        slot.family,
        VERSION,
        tuple(objects),
        And(tuple(atoms)),
        target,
        "O",
        candidates,
        direction,
        (),
    )


def size_target(slot):
    peers = [
        _core(replace(slot, kind=kind, case=case))
        for kind in KINDS
        for case in (*CASES, *(("multiple",) if kind == "which" else ()))
    ]
    objects = max(len(s.objects) for s in peers) + 5 + slot.extra_size
    premises = (
        objects + max(len(s.premise.operands) - len(s.objects) for s in peers) + 2
    )
    return objects, premises


def construct(slot):
    if slot not in slots():
        raise ValueError("slot is outside the fixed prototype manifest")
    core = _core(slot)
    object_target, premise_target = size_target(slot)
    if object_target > len(NAMES):
        raise ValueError("prototype exceeds the ordinary-name vocabulary")
    rng = random.Random(stable_int(f"{slot.identifier}/presentation"))
    opposite = tuple(Direction)[(tuple(Direction).index(slot.direction) + 4) % 8]
    contexts = [f"Context{i}" for i in range(object_target - len(core.objects))]
    atoms = list(core.premise.operands)
    existing = set()
    lengths = {"O": 0}
    for i, node in enumerate(contexts):
        parent = rng.choice([p for p in ["O", *contexts[:i]] if lengths[p] < slot.span])
        lengths[node] = lengths[parent] + 1
        atoms.append(direction_constraint(node, parent, opposite))
        existing.add((node, parent))
    # All edges follow the same acyclic order on an excluded side of O.
    # These objects participate in Which/Count; they are not ignored padding.
    pairs = [
        (node, parent)
        for i, node in enumerate(contexts)
        for parent in ["O", *contexts[:i]]
        if (node, parent) not in existing
    ]
    rng.shuffle(pairs)
    needed = premise_target - len(atoms)
    if not 0 <= needed <= len(pairs):
        raise ValueError("insufficient context edges for the matched scene size")
    atoms.extend(direction_constraint(a, b, opposite) for a, b in pairs[:needed])
    # Random inverse wording preserves exact spatial semantics.
    atoms = [
        direction_constraint(
            a.reference,
            a.subject,
            tuple(Direction)[(tuple(Direction).index(next(iter(a.allowed))) + 4) % 8],
        )
        if rng.getrandbits(1)
        else a
        for a in atoms
    ]
    rng.shuffle(atoms)
    story = replace(core, objects=(*core.objects, *contexts), premise=And(tuple(atoms)))
    names = story_name_map(story.objects, slot.identifier)
    story = rename_structure(story, names)
    problem = all_scene_problem(story.problems()[KINDS.index(slot.kind)])
    return story, problem


def check_semantics(slot, semantics):
    if slot.case == "inconsistent":
        expected = not semantics.consistent and not semantics.entailed
    elif not semantics.consistent:
        expected = False
    elif slot.case == "no-entailed":
        expected = not semantics.entailed
    elif slot.kind == "direction":
        expected = semantics.entailed == {slot.direction.value}
    elif slot.kind == "count":
        expected = semantics.entailed == {slot.count_value}
    else:
        expected = len(semantics.entailed) == (
            2 + slot.span % 3 if slot.case == "multiple" else 1
        )
    if not expected:
        raise ValueError("verified semantics differ from the intended prototype case")
