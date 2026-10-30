"""Bounded semantic constructions on sampled, shared forest lineages."""

import random
from dataclasses import replace

from spatial.v2.composition_feasibility import scramble, stable_int
from spatial.v2.composition_scene_sampler import _core, family_id, sample_scaffold
from spatial.v2.solver import And, Direction, direction_constraint

VERSION = "sampled-targeted-v1"
PATTERNS = (
    "singleton",
    "empty",
    "multiple",
    "contingent",
    "consistent-cycle",
    "inconsistent-cycle",
    "coincidence",
)


def sample_targeted(seed, pattern, *, span=2, direction=None, variant=0, matches=None):
    """Return one scheduled draw, never retry or claim a controlled depth."""
    if pattern not in PATTERNS:
        raise ValueError("unknown targeted pattern")
    if type(span) is not int or not 1 <= span <= 3:
        raise ValueError("span must be an integer from one through three")
    if pattern.endswith("-cycle") and span < 2:
        raise ValueError("cycle constructions require span at least two")
    if type(variant) is not int or variant < 0:
        raise ValueError("variant must be a nonnegative integer")
    scaffold = sample_scaffold(seed)
    if direction is None:
        direction = tuple(Direction)[stable_int(f"{seed}/direction") % 8]
    if not isinstance(direction, Direction):
        raise TypeError("direction must be a Direction")
    if matches is not None and (
        pattern != "multiple"
        or type(matches) is not int
        or not 2 <= matches <= min(4, scaffold.candidates)
    ):
        raise ValueError("matches is only supported for multiple, between two and four")
    count = (
        0
        if pattern == "empty"
        else (matches if matches is not None else 2 + seed % 3)
        if pattern == "multiple"
        else 2
        if pattern == "coincidence"
        else 1
    )
    core = _core(scaffold, span, direction, count)
    objects, atoms = list(core.objects), list(core.premise.operands)

    if pattern == "contingent":
        # Separate chains above a common anchor do not order their endpoints.
        # Construct diagonal constraints directly, then use quarter turns only.
        directions = tuple(Direction)
        turn = (directions.index(direction) // 2) * 2
        diagonal = directions[(1 + turn) % 8]
        objects.append("Anchor")
        for i in range(span):
            atom = atoms[i]
            atoms[i] = direction_constraint(
                atom.subject, "Anchor" if i == 0 else atom.reference, diagonal
            )
        atoms.append(direction_constraint("O", "Anchor", diagonal))
    elif pattern.endswith("-cycle"):
        relation = direction
        if pattern == "inconsistent-cycle":
            directions = tuple(Direction)
            relation = directions[(directions.index(direction) + 4) % 8]
        atoms.append(direction_constraint("C0", "O", relation))
    elif pattern == "coincidence":
        # Equal x via a shared North reference and equal y via a shared East
        # reference force C0/C1 to coincide, violating distinct named points.
        objects.extend(("AxisX", "AxisY"))
        for candidate in ("C0", "C1"):
            atoms.append(direction_constraint(candidate, "AxisX", Direction.NORTH))
            atoms.append(direction_constraint(candidate, "AxisY", Direction.EAST))

    rng = random.Random(stable_int(f"{VERSION}/{seed}/{span}/{variant}/context"))
    distractors = []
    # Fresh leaves cannot constrain their attachment's location.
    objects.append("Context")
    distractors.append(len(atoms))
    atoms.append(
        direction_constraint(
            "Context", rng.choice(core.objects), rng.choice(tuple(Direction))
        )
    )
    if rng.randrange(2):
        objects.extend(("SeparateA", "SeparateB"))
        distractors.append(len(atoms))
        atoms.append(
            direction_constraint("SeparateA", "SeparateB", rng.choice(tuple(Direction)))
        )
    if len(objects) > 32 or len(atoms) > 32:
        raise ValueError("targeted construction exceeded scene bounds")
    return scramble(
        replace(
            core,
            story_id=f"{VERSION}-{seed}-{pattern}-s{span}-{direction.value}-v{variant}-m{count}",
            # Share identity with controlled forests and all semantic/context
            # variants; a different generator name must not permit split leakage.
            source_family=family_id(scaffold),
            recipe=f"{VERSION}/{pattern}",
            objects=tuple(objects),
            premise=And(tuple(atoms)),
            distractors=tuple(distractors),
        ),
        rng,
    )
