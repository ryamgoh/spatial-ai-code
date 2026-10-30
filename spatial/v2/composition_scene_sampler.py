"""Seeded controlled scene families, not a distribution of all spatial problems."""

import random
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations

from spatial.v2.composition_feasibility import scramble, stable_int
from spatial.v2.composition_pilot import Story, story_key
from spatial.v2.solver import And, Direction, direction_constraint

VERSION = "sampled-forest-v1"
MAX_CORE_OBJECTS = 24
PROOF_PATTERNS = ("homogeneous", "mixed-chain", "axis-split")


@dataclass(frozen=True)
class Scaffold:
    candidates: int
    levels: tuple[int, ...]
    attachments: tuple[int, ...]
    extra_pairs: tuple[tuple[int, int], ...]


def sample_scaffold(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(seed)
    count = rng.randint(4, 8)
    pairs = list(combinations(range(1, count), 2))
    rng.shuffle(pairs)
    return Scaffold(
        count,
        tuple(rng.randint(1, 10) for _ in range(count - 1)),
        tuple(rng.getrandbits(32) for _ in range(count - 1)),
        tuple(sorted(pairs[: rng.randint(0, min(4, len(pairs)))])),
    )


def _support(depth, direction, pattern, *, exclude=False):
    """Construct endpoint supports; axis-split paths are jointly necessary."""
    directions = tuple(Direction)
    diagonal = directions.index(direction) % 2 == 1
    if pattern not in PROOF_PATTERNS:
        raise ValueError("unknown proof pattern")
    if pattern != "homogeneous" and depth < 2:
        raise ValueError("nonhomogeneous proofs require depth at least two")
    if pattern == "mixed-chain" and not diagonal:
        raise ValueError("mixed-chain requires a diagonal direction")
    objects, atoms = [], []

    def path(relations, prefix):
        previous = "O"
        for i, relation in enumerate(relations):
            node = "C0" if i == len(relations) - 1 else f"{prefix}{i}"
            if node != "C0":
                objects.append(node)
            atoms.append(direction_constraint(node, previous, relation))
            previous = node

    if pattern == "homogeneous":
        path([direction] * depth, "Positive")
    else:
        turn = directions.index(direction) - (1 if diagonal else 0)

        def rotate(relations):
            return [directions[(directions.index(d) + turn) % 8] for d in relations]

        if pattern == "mixed-chain":
            path(
                rotate(
                    [
                        Direction.EAST if i % 2 == 0 else Direction.NORTH
                        for i in range(depth)
                    ]
                ),
                "Positive",
            )
        else:
            # Short path fixes x, but leaves the sign of y open. Long path
            # fixes y, but leaves x open. Neither alone fixes the direction.
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
                # Preserve the query's horizontal sign; only the long vertical
                # path contradicts it, so a zero-count exclusion stays deep.
                long = [Direction.SOUTHEAST] * (depth - 1) + [Direction.SOUTHWEST]
            path(rotate(long), "Positive")
            path(rotate(short), "AxisLink")
    return objects, atoms


def _core(scaffold, depth, direction, matches=1, *, proof_pattern="homogeneous"):
    directions = tuple(Direction)
    opposite = directions[(directions.index(direction) + 4) % 8]
    candidates = tuple(f"C{i}" for i in range(scaffold.candidates))
    objects, atoms = ["O", *candidates], []
    target_direction = direction if matches else opposite

    def add(subject, reference, relation):
        atoms.append(direction_constraint(subject, reference, relation))

    exclude = (
        proof_pattern == "axis-split"
        and matches == 0
        and directions.index(direction) % 2 == 1
    )
    support_objects, support_atoms = _support(
        depth,
        direction if exclude else target_direction,
        proof_pattern,
        exclude=exclude,
    )
    objects.extend(support_objects)
    atoms.extend(support_atoms)
    for candidate in candidates[1:matches]:
        add(candidate, "O", direction)

    # A tree rooted at O, pointing opposite the query, fixes all nonmatches.
    # Only intermediate nodes can be reused as parents; candidates stay leaves.
    pool = [("O", 0)]
    # Two supporting paths add one core cycle; reserve its extra premise.
    core_limit = MAX_CORE_OBJECTS - (proof_pattern == "axis-split")
    ranks = {"O": 0}
    next_rank = 1
    for candidate, level, attachment in zip(
        candidates[1:], scaffold.levels, scaffold.attachments
    ):
        if candidate in candidates[:matches]:
            continue
        desired = max(1, (level * depth + 9) // 10)
        parents = [(node, length) for node, length in pool if length < desired]
        parent, length = parents[attachment % len(parents)]
        while length + 1 < desired and len(objects) < core_limit:
            node = f"Negative{len(pool)}"
            objects.append(node)
            add(node, parent, opposite)
            length += 1
            pool.append((node, length))
            ranks[node] = next_rank
            next_rank += 1
            parent = node
        add(candidate, parent, opposite)
        ranks[candidate] = next_rank
        next_rank += 1

    # The ranks are one compatible order, not metric constraints or supervision.
    # Candidate-to-candidate edges add consistent cycles without connecting to
    # the positive chain and introducing a shortcut to the nominated match.
    for a, b in scaffold.extra_pairs:
        if a < matches or b < matches:
            continue
        first, second = f"C{a}", f"C{b}"
        subject, reference = (
            (first, second) if ranks[first] > ranks[second] else (second, first)
        )
        add(subject, reference, opposite)
    return Story(
        "unassigned",
        "unassigned",
        VERSION,
        tuple(objects),
        And(tuple(atoms)),
        "C0",
        "O",
        candidates,
        direction,
        (),
    )


@lru_cache(maxsize=4096)
def family_id(scaffold):
    """Canonical anchor graph, independent of depth, direction and decoration."""
    anchor = _core(scaffold, 10, Direction.NORTH)
    return f"{VERSION}/{story_key(anchor.objects, anchor.premise)}"


def sample_scene(
    seed, depth, *, direction=None, variant=0, matches=1, proof_pattern="homogeneous"
):
    if type(depth) is not int or not 1 <= depth <= 10:
        raise ValueError("depth must be an integer from 1 through 10")
    if type(variant) is not int or variant < 0:
        raise ValueError("variant must be a nonnegative integer")
    scaffold = sample_scaffold(seed)
    if type(matches) is not int or not 0 <= matches <= scaffold.candidates:
        raise ValueError("matches must be an integer within the candidate domain")
    if direction is None:
        direction = tuple(Direction)[stable_int(f"{seed}/direction") % 8]
    if not isinstance(direction, Direction):
        raise TypeError("direction must be a Direction")
    core = _core(scaffold, depth, direction, matches, proof_pattern=proof_pattern)
    rng = random.Random(stable_int(f"{seed}/{depth}/{variant}/presentation"))
    objects, atoms = list(core.objects), list(core.premise.operands)
    distractors = []
    for i in range(rng.randint(1, 3)):
        node = f"Context{i}"
        objects.append(node)
        distractors.append(len(atoms))
        atoms.append(
            direction_constraint(
                node, rng.choice(core.objects), rng.choice(tuple(Direction))
            )
        )
    for i in range(rng.randrange(3)):
        a, b = f"Separate{i}A", f"Separate{i}B"
        objects.extend((a, b))
        distractors.append(len(atoms))
        atoms.append(direction_constraint(a, b, rng.choice(tuple(Direction))))
    return scramble(
        replace(
            core,
            story_id=f"{VERSION}-{seed}-d{depth}-{direction.value}-v{variant}-m{matches}"
            + (f"-{proof_pattern}" if proof_pattern != "homogeneous" else ""),
            recipe=VERSION
            if proof_pattern == "homogeneous"
            else f"{VERSION}/{proof_pattern}",
            source_family=family_id(scaffold),
            objects=tuple(objects),
            premise=And(tuple(atoms)),
            distractors=tuple(distractors),
        ),
        rng,
    )
