"""Development-only all-scene cardinality constructions; no production routing."""

import random
from dataclasses import replace
from functools import lru_cache

from spatial.v2.composition_feasibility import scramble, stable_int
from spatial.v2.composition_pilot import story_key
from spatial.v2.composition_scene_sampler import _core, sample_scaffold
from spatial.v2.entailment_names import rename_structure, story_name_map
from spatial.v2.entailment_scene_protocol import all_scene_problem
from spatial.v2.solver import And, Direction, direction_constraint

VERSION = "all-scene-cardinality-probe-v1"


def _probe_core(seed, depth, direction, matches):
    pattern = "homogeneous" if depth == 1 else "axis-split"
    core = _core(
        sample_scaffold(seed),
        depth,
        direction,
        matches,
        proof_pattern=pattern,
        shared_selection_support=matches > 1,
    )
    directions = tuple(Direction)
    index = directions.index(direction)
    atoms = list(core.premise.operands)
    if depth > 1 and index % 2 == 0:
        # Equality on the query axis goes through an opposite-side object.
        # The old N-then-S support made its intermediate a shallow N answer.
        opposite = directions[(index + 4) % 8]
        atoms = [
            replace(
                atom,
                allowed=frozenset(
                    (opposite if atom.subject == "AxisLink0" else direction,)
                ),
            )
            if atom.reference == "AxisLink0" or atom.subject == "AxisLink0"
            else atom
            for atom in atoms
        ]
    return replace(core, premise=And(tuple(atoms)))


@lru_cache(maxsize=64)
def probe_family(seed):
    core = _probe_core(seed, 10, Direction.NORTH, 1)
    return f"{VERSION}/{story_key(core.objects, core.premise)}"


def probe_scene(seed, depth, direction, matches):
    if type(seed) is not int or not 0 <= seed < 64:
        raise ValueError("probe seed must be an integer in 0-63")
    if type(depth) is not int or not 1 <= depth <= 10:
        raise ValueError("probe depth must be an integer in 1-10")
    if type(matches) is not int or not 1 <= matches <= 4:
        raise ValueError("probe matches must be an integer in 1-4")
    if not isinstance(direction, Direction):
        raise TypeError("probe direction must be a Direction")
    core = _probe_core(seed, depth, direction, matches)
    opposite = tuple(Direction)[(tuple(Direction).index(direction) + 4) % 8]
    # Context facts remain present but are redundant for all-scene membership.
    # Removing the extra fact must leave the declared context object excluded.
    atoms = (
        *core.premise.operands,
        direction_constraint("Context", "O", opposite),
        direction_constraint("Context", "AxisLink0" if depth > 1 else "O", opposite),
    )
    # At depth one use one fact, not a duplicate pair.
    if depth == 1:
        atoms = atoms[:-1]
    story = replace(
        core,
        story_id=f"{VERSION}-s{seed}-d{depth}-{direction.value}-m{matches}",
        source_family=probe_family(seed),
        recipe=VERSION,
        objects=(*core.objects, "Context"),
        premise=And(atoms),
        distractors=(len(atoms) - 1,) if depth > 1 else (),
    )
    story = scramble(story, random.Random(stable_int(story.story_id)))
    names = story_name_map(story.objects, story.story_id)
    story = rename_structure(story, names)
    problem = all_scene_problem(story.problems()[1])
    return story, problem
