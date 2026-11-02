"""Controlled atomic scenes; allocation, menu selection and verification are separate."""

import random
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations

from spatial.v2.composition_contracts import UnsupportedEvidence, VerificationError
from spatial.v2.composition_depth import membership_support
from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_pilot import Story
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.difficulty import derive_axis_proof
from spatial.v2.entailment_contracts import analyze
from spatial.v2.entailment_dataset import measure_depth
from spatial.v2.entailment_names import NAMES, rename_structure, story_name_map
from spatial.v2.entailment_scene_protocol import all_scene_problem
from spatial.v2.solver import (
    And,
    Direction,
    direction_between,
    direction_constraint,
    direction_signs,
    membership_constraint,
)

VERSION = "controlled-entailment-generator-v1"
EXPLICIT_SIZE_VERSION = "explicit-size-checked-augmentation-v1"
_KINDS = ("direction", "which", "count")
_CASES = ("sole", "no-entailed", "multiple", "inconsistent")
_DIRECTIONS = tuple(Direction)


@dataclass(frozen=True)
class SceneRequest:
    """Semantic intent, not a certified label or an instruction to sample a cohort.

    ``span`` is requested positive support depth only when ``requested_depth``
    is non-null. The two E mechanisms use it as a construction parameter.
    ``size`` selects a tested recipe, not an arbitrary premise-count interval.
    """

    span: int
    direction: Direction
    kind: str
    case: str
    count_value: int = 1
    variant: int = 0
    size: str = "contextual"

    def __post_init__(self):
        if (
            type(self.span) is not int
            or not 1 <= self.span <= 10
            or not isinstance(self.direction, Direction)
            or self.kind not in _KINDS
            or self.case not in _CASES
            or self.case == "multiple"
            and self.kind != "which"
            or type(self.count_value) is not int
            or not 0 <= self.count_value <= 4
            or type(self.variant) is not int
            or self.variant not in (0, 1)
            or self.size not in ("compact", "compact-extra", "contextual")
            or self.size != "contextual"
            and self.span != 1
        ):
            raise ValueError("unsupported controlled scene request")

    @property
    def requested_depth(self):
        return None if self.case in ("no-entailed", "inconsistent") else self.span


def _opposite(direction):
    return _DIRECTIONS[(_DIRECTIONS.index(direction) + 4) % 8]


def _matches(request):
    if request.case in ("no-entailed", "inconsistent"):
        return 0
    if request.case == "multiple":
        return 2 + request.span % 3
    return request.count_value if request.kind == "count" else 1


def _support(request, matches):
    """Construct the long axis support and, when needed, its short companion."""
    depth, direction = request.span, request.direction
    index = _DIRECTIONS.index(direction)
    opposite = _opposite(direction)
    target_direction = direction if matches else opposite
    objects, atoms = [], []

    def path(relations, prefix):
        previous = "O"
        for i, relation in enumerate(relations):
            node = "C0" if i == len(relations) - 1 else f"{prefix}{i}"
            if node != "C0":
                objects.append(node)
            atoms.append(direction_constraint(node, previous, relation))
            previous = node

    if depth == 1:
        path([target_direction], "Positive")
        return objects, atoms
    diagonal = index % 2 == 1
    exclude = matches == 0 and diagonal
    support_direction = direction if exclude else target_direction
    turn = _DIRECTIONS.index(support_direction) - int(diagonal)
    short = (
        [Direction.SOUTHEAST, Direction.NORTHEAST]
        if diagonal
        else [Direction.NORTH, Direction.SOUTH]
    )
    long = (
        [Direction.NORTHWEST] * (depth - 1) + [Direction.NORTHEAST]
        if diagonal
        else [Direction.NORTHEAST] * (depth - 1) + [Direction.NORTHWEST]
    )
    if exclude:
        long = [Direction.SOUTHEAST] * (depth - 1) + [Direction.SOUTHWEST]
    for relations, prefix in ((long, "Positive"), (short, "AxisLink")):
        path(
            [_DIRECTIONS[(_DIRECTIONS.index(d) + turn) % 8] for d in relations], prefix
        )
    if matches > 0 and not diagonal:
        # The equality-support intermediate must not become an extra Which answer.
        atoms = [
            direction_constraint(
                a.subject,
                a.reference,
                opposite if a.subject == "AxisLink0" else direction,
            )
            if a.subject == "AxisLink0" or a.reference == "AxisLink0"
            else a
            for a in atoms
        ]
    return objects, atoms


def _compact(request):
    direction, opposite = request.direction, _opposite(request.direction)
    objects = ["O", *(f"C{i}" for i in range(5))]
    atoms = []

    def add(a, b, relation):
        atoms.append(direction_constraint(a, b, relation))

    if request.case == "no-entailed":
        if request.variant == 0:
            diagonal = _DIRECTIONS[(_DIRECTIONS.index(direction) // 2) * 2 + 1]
            add("C0", "C1", diagonal)
            add("O", "C1", diagonal)
            for node in objects[3:]:
                add(node, "O", opposite)
            add("C2", "C3", opposite)
        else:
            add("C0", "C1", direction)
            add("C1", "C2", direction)
            add("C0", "C2", direction)
            add("C3", "O", opposite)
            add("C4", "O", opposite)
            add("C4", "C3", opposite)
    elif request.case == "inconsistent":
        if request.variant == 0:
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
        matches = _matches(request)
        for i in range(5):
            add(f"C{i}", "O", direction if i < matches else opposite)
        a, b = (0, 1) if matches >= 2 else (3, 4)
        add(f"C{a}", f"C{b}", direction if matches >= 2 else opposite)
    return objects, atoms, "C0"


def _contextual(request):
    matches = _matches(request)
    links, support = _support(request, matches)
    atoms = list(support)
    candidates = [f"C{i}" for i in range(5)]
    for candidate in candidates[1:]:
        if candidate in candidates[:matches]:
            atoms.extend(
                replace(a, subject=candidate) for a in support if a.subject == "C0"
            )
        else:
            atoms.append(
                direction_constraint(candidate, "O", _opposite(request.direction))
            )
    objects = ["O", *candidates, *links]
    target = "C0"
    if request.case == "no-entailed":
        target = "Unknown"
        if request.variant == 0:
            diagonal = _DIRECTIONS[(_DIRECTIONS.index(request.direction) // 2) * 2 + 1]
            objects.extend(("Unknown", "Anchor"))
            atoms.extend(
                (
                    direction_constraint("Unknown", "Anchor", diagonal),
                    direction_constraint("O", "Anchor", diagonal),
                )
            )
        else:
            objects.extend(("Unknown", "FreeAnchor"))
            atoms.append(
                direction_constraint("Unknown", "FreeAnchor", request.direction)
            )
    elif request.case == "inconsistent":
        if request.variant == 0:
            objects.extend(("CycleA", "CycleB"))
            atoms.extend(
                (
                    direction_constraint("CycleA", "O", request.direction),
                    direction_constraint("CycleB", "CycleA", request.direction),
                    direction_constraint("O", "CycleB", request.direction),
                )
            )
        else:
            nodes = ["O", *(f"Conflict{i}" for i in range(1 + request.span % 4))]
            objects.extend(nodes[1:])
            atoms.extend(
                direction_constraint(a, b, request.direction)
                for a, b in zip(nodes, [*nodes[1:], nodes[0]])
            )
    return objects, atoms, target


def _contextual_core_sizes(span, direction):
    for kind in _KINDS:
        for case in _CASES:
            if case == "multiple" and kind != "which":
                continue
            for count in range(5) if kind == "count" else (1,):
                for variant in (0, 1):
                    request = SceneRequest(span, direction, kind, case, count, variant)
                    objects, atoms, _ = _contextual(request)
                    yield request, len(objects), len(atoms)


@lru_cache(maxsize=80)
def _contextual_size(span, direction):
    # Include every supported answer case in the common size budget.
    cores = list(_contextual_core_sizes(span, direction))
    objects = max(n for _, n, _ in cores) + 4
    premises = objects + max(a - n for _, n, a in cores)
    return objects, premises


@lru_cache(maxsize=240)
def _context_premise_capacity(span, direction, object_target):
    """Common feasible ceiling, including the sole-Which contingent variant."""
    capacities = []
    for request, objects, atoms in _contextual_core_sizes(span, direction):
        for contingent in (
            (False, True)
            if request.kind == "which" and request.case == "sole"
            else (False,)
        ):
            contexts = object_target - objects - 2 * contingent
            capacities.append(atoms + 2 * contingent + contexts * (contexts + 1) // 2)
    return min(capacities)


def _flexible_core(request, contingent):
    if request.case in ("no-entailed", "inconsistent"):
        return _compact(request)
    matches = (
        2
        if request.case == "multiple"
        else request.count_value
        if request.kind == "count"
        else 1
    )
    links, support = _support(request, matches)
    objects = ["O", *(f"C{i}" for i in range(5)), *links]
    atoms = list(support)
    for i in range(1, 5):
        if request.case == "multiple" and i == 1:
            atoms.extend(replace(a, subject="C1") for a in support if a.subject == "C0")
        else:
            atoms.append(
                direction_constraint(
                    f"C{i}",
                    "O",
                    request.direction if i < matches else _opposite(request.direction),
                )
            )
    if contingent:
        diagonal = _DIRECTIONS[(_DIRECTIONS.index(request.direction) // 2) * 2 + 1]
        atoms = [
            a for a in atoms if not (a.reference == "O" and a.subject in ("C1", "C2"))
        ]
        atoms.extend(
            (
                direction_constraint("C1", "C2", diagonal),
                direction_constraint("O", "C2", diagonal),
            )
        )
    return objects, atoms, "C0"


def _preserves_depth(problem, request):
    if request.case in ("no-entailed", "inconsistent"):
        return True
    if request.kind == "count":
        return (
            membership_support(problem, "C0", request.count_value > 0)["depth"]
            == request.span
        )
    members = ("C0", "C1") if request.case == "multiple" else ("C0",)
    return all(
        (proof := derive_axis_proof(problem, node, "O", request.direction)) is not None
        and max(len(proof.x.premise_indices), len(proof.y.premise_indices))
        == request.span
        for node in members
    )


def _augmentation_worlds(problem, semantics, request, contingent, solver):
    worlds = list(semantics.counterexamples.values())
    if contingent:
        claim = solver.assess(problem, membership_constraint("C1", problem.query))
        if claim.error:
            raise VerificationError(f"inconclusive contingent analysis: {claim.error}")
        if not claim.possible or claim.entailed:
            return None
        worlds.append(claim.witness)
    if (
        request.kind != "direction"
        and request.case in ("sole", "multiple")
        and not contingent
    ):
        try:
            measurement = measure_depth(problem, solver=solver)
        except UnsupportedEvidence:
            return None
        if measurement["depth"] != request.span:
            return None
        removed = measurement["removed_premise_index"]
        reduced = replace(
            problem,
            premise=And(
                tuple(
                    atom
                    for i, atom in enumerate(problem.premise.operands)
                    if i != removed
                )
            ),
        )
        after = analyze(reduced, solver=solver)
        # Keep a witness that this core edge remains necessary, not just long.
        changed = sorted(semantics.entailed - after.entailed)
        if not changed:
            raise VerificationError("depth dependency witness is missing")
        worlds.append(after.counterexamples[changed[0]])
    return worlds


def _checked_scene(request, identity, family, rng, size_pair, contingent, limit):
    n, p = size_pair
    objects, atoms, target = _flexible_core(request, contingent)
    if n < len(objects):
        raise ValueError("requested objects smaller than semantic core")
    sx, sy = direction_signs(request.direction)
    axis, sign = (0, sx) if sx else (1, sy)
    context = [f"Context{i}" for i in range(n - len(objects))]
    for node in context:
        xy = [rng.randint(-20, 20), rng.randint(-20, 20)]
        xy[axis] = -sign * rng.randint(1, 20)
        atoms.append(
            direction_constraint(node, "O", direction_between(tuple(xy), (0, 0)))
        )
    objects += context
    if len(atoms) > p:
        raise ValueError("requested premises smaller than semantic core")
    story = Story(
        identity,
        family,
        EXPLICIT_SIZE_VERSION,
        tuple(objects),
        And(tuple(atoms)),
        target,
        "O",
        tuple(f"C{i}" for i in range(5)),
        request.direction,
        (),
    )
    problem = all_scene_problem(story.problems()[_KINDS.index(request.kind)])
    solver = QuestionSolver()
    semantics = analyze(problem, solver=solver)
    if (
        semantics.consistent != (request.case != "inconsistent")
        or request.case == "sole"
        and len(semantics.entailed) != 1
        or request.case == "sole"
        and request.kind == "count"
        and semantics.entailed != {request.count_value}
        or request.case == "multiple"
        and len(semantics.entailed) != 2
        or request.case == "no-entailed"
        and semantics.entailed
        or not _preserves_depth(problem, request)
    ):
        raise VerificationError(
            "explicit-size core differs from requested semantics/depth"
        )
    worlds = _augmentation_worlds(problem, semantics, request, contingent, solver)
    if worlds is None:
        raise UnsupportedEvidence("explicit-size core lacks required evidence")
    if not worlds:
        # An inconsistent core stays inconsistent under added constraints.
        worlds = [{o: (i, rng.randrange(1000)) for i, o in enumerate(objects)}]
    existing = {frozenset((a.subject, a.reference)) for a in atoms}
    additions = []
    for a, b in combinations(objects, 2):
        if frozenset((a, b)) not in existing:
            directions = {direction_between(w[a], w[b]) for w in worlds}
            if len(directions) == 1:
                additions.append(direction_constraint(a, b, directions.pop()))
    rng.shuffle(additions)
    for atom in additions:
        if len(atoms) == p:
            break
        candidate = replace(problem, premise=And((*atoms, atom)))
        if _preserves_depth(candidate, request):
            atoms.append(atom)
            problem = candidate
    if len(atoms) < p:
        # Fixed witness layouts are sufficient, not necessary: allow fresh worlds.
        used = {frozenset((a.subject, a.reference)) for a in atoms}
        proposals = [
            direction_constraint(a, b, d)
            for a, b in combinations(objects, 2)
            if frozenset((a, b)) not in used
            for d in _DIRECTIONS
        ]
        rng.shuffle(proposals)
        for atom in proposals[:limit]:
            if len(atoms) == p:
                break
            key = frozenset((atom.subject, atom.reference))
            if key in used:
                continue
            candidate = replace(problem, premise=And((*atoms, atom)))
            if not _preserves_depth(candidate, request):
                continue
            actual = analyze(candidate, solver=solver)
            if (actual.consistent, actual.entailed) != (
                semantics.consistent,
                semantics.entailed,
            ):
                continue
            if (
                _augmentation_worlds(candidate, actual, request, contingent, solver)
                is None
            ):
                continue
            atoms.append(atom)
            used.add(key)
            problem = candidate
    if len(atoms) != p:
        raise ValueError(
            f"augmentation exhausted: requested {size_pair}, built {len(atoms)} premises"
        )
    return _present_scene(replace(story, premise=And(tuple(atoms))), rng, request.kind)


def build_scene(
    request,
    *,
    identity,
    family,
    structure_seed=None,
    contingent=False,
    varied_context=False,
    size_pair=None,
    augmentation_limit=512,
):
    """Return an unverified (Story, all-scene SpatialProblem), without file I/O.

    Identity seeds presentation; neither it nor ``family`` certifies uniqueness.
    Callers must run entailment/evidence/depth verification before using the result.
    This function chooses no menu, label, split, quota, retries, or training data.
    """
    if not isinstance(request, SceneRequest):
        raise TypeError("expected a SceneRequest")
    if any(
        not isinstance(value, str) or not value.strip() for value in (identity, family)
    ):
        raise ValueError("nonempty identity and family are required")
    if structure_seed is not None and (
        type(structure_seed) is not int or structure_seed < 0
    ):
        raise ValueError("structure_seed must be a nonnegative integer")
    if type(contingent) is not bool or (
        contingent and (request.kind != "which" or request.case != "sole")
    ):
        raise ValueError("contingent context is supported only for sole Which")
    if type(varied_context) is not bool or (
        varied_context and (request.size != "contextual" or structure_seed is None)
    ):
        raise ValueError("varied context requires a seeded contextual scene")
    if size_pair is not None:
        if (
            request.size != "contextual"
            or structure_seed is None
            or varied_context
            or not isinstance(size_pair, (tuple, list))
            or len(size_pair) != 2
            or any(type(v) is not int for v in size_pair)
            or not 6 <= size_pair[0] <= len(NAMES)
            or not 1 <= size_pair[1] <= size_pair[0] * (size_pair[0] - 1) // 2
            or type(augmentation_limit) is not int
            or augmentation_limit < 0
        ):
            raise ValueError("invalid explicit-size construction request")
        return _checked_scene(
            request,
            identity,
            family,
            random.Random(structure_seed),
            tuple(size_pair),
            contingent,
            augmentation_limit,
        )
    if request.size == "contextual":
        objects, atoms, target = _contextual(request)
        object_target, premise_target = _contextual_size(
            request.span, request.direction
        )
        if varied_context:
            # Separate size randomness from relation/presentation randomness.
            # The same size law applies to every semantic case at each level.
            size_rng = random.Random(stable_int(f"context-size-v1/{structure_seed}"))
            extra_objects = size_rng.randint(0, min(2, len(NAMES) - object_target))
            object_target += extra_objects
            premise_target += extra_objects
            spare = (
                _context_premise_capacity(
                    request.span, request.direction, object_target
                )
                - premise_target
            )
            if spare < 0:
                raise ValueError("context size cannot fit every semantic case")
            premise_target += size_rng.randint(min(2, spare), min(4, spare))
    else:
        objects, atoms, target = _compact(request)
        object_target = premise_target = 9 if request.size == "compact-extra" else 6
    if object_target > len(NAMES):
        raise ValueError("ordinary-name vocabulary exceeded")
    rng = random.Random(
        stable_int(identity) if structure_seed is None else structure_seed
    )
    opposite = _opposite(request.direction)
    if contingent:
        # A shared anchor leaves Unknown unordered relative to O. It is possible,
        # not entailed; the old positive certificate remains valid.
        diagonal = _DIRECTIONS[(_DIRECTIONS.index(request.direction) // 2) * 2 + 1]
        objects.extend(("Unknown", "Anchor"))
        atoms.extend(
            (
                direction_constraint("Unknown", "Anchor", diagonal),
                direction_constraint("O", "Anchor", diagonal),
            )
        )
        if len(objects) > object_target or len(atoms) > premise_target:
            raise ValueError("selected size cannot accommodate contingent context")
    contexts = [f"Context{i}" for i in range(object_target - len(objects))]
    positions = {"O": (0, 0)}
    if structure_seed is not None:
        signs = direction_signs(request.direction)
        axis = 0 if signs[0] else 1
        sign = signs[axis]
        for i, node in enumerate([*(f"C{i}" for i in range(1, 5)), *contexts], 1):
            point = [rng.randint(-12, 12), rng.randint(-12, 12)]
            point[axis] = -sign * i
            positions[node] = tuple(point)
        # Only direct exclusions are varied. Neither positive axis support nor
        # any cross-object core constraints are edited.
        if request.size == "contextual":
            atoms = [
                direction_constraint(
                    a.subject,
                    a.reference,
                    direction_between(positions[a.subject], positions["O"]),
                )
                if a.subject in positions
                and a.subject.startswith("C")
                and a.reference == "O"
                and a.allowed == frozenset((opposite,))
                else a
                for a in atoms
            ]

    def context_relation(node, parent):
        return (
            opposite
            if structure_seed is None
            else direction_between(positions[node], positions[parent])
        )

    edges, lengths = set(), {"O": 0}
    for i, node in enumerate(contexts):
        parents = [p for p in ["O", *contexts[:i]] if lengths[p] < request.span]
        parent = rng.choice(parents)
        atoms.append(direction_constraint(node, parent, context_relation(node, parent)))
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
    atoms.extend(
        direction_constraint(a, b, context_relation(a, b)) for a, b in pairs[:missing]
    )
    story = Story(
        identity,
        family,
        "controlled-entailment-varied-context-v1"
        if varied_context
        else VERSION
        if structure_seed is None and not contingent
        else "controlled-entailment-generator-v2",
        (*objects, *contexts),
        And(tuple(atoms)),
        target,
        "O",
        tuple(f"C{i}" for i in range(5)),
        request.direction,
        (),
    )
    return _present_scene(story, rng, request.kind)


def _present_scene(story, rng, kind):
    atoms = [
        direction_constraint(a.reference, a.subject, _opposite(next(iter(a.allowed))))
        if rng.getrandbits(1)
        else a
        for a in story.premise.operands
    ]
    rng.shuffle(atoms)
    story = replace(story, premise=And(tuple(atoms)))
    identity = story.story_id
    story = rename_structure(story, story_name_map(story.objects, identity))
    return story, all_scene_problem(story.problems()[_KINDS.index(kind)])
