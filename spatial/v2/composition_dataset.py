"""Fixed random-plus-targeted composition candidate; no retry-to-quota loop."""

import argparse
import json
import random
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from spatial.v2.composition_pilot import (
    Story,
    construct_stories,
    run_pilot,
    write_pilot,
)
from spatial.v2.solver import And, Direction, direction_between, direction_constraint

SEED = 20261029
RANDOM_WORLD_STORIES = 100
RANDOM_ATOM_STORIES = 50


def rename_story(story, rng):
    labels = list(range(len(story.objects)))
    rng.shuffle(labels)
    names = {obj: f"Object{labels[i]}" for i, obj in enumerate(story.objects)}
    objects = list(names.values())
    rng.shuffle(objects)
    candidates = [names[obj] for obj in story.candidates]
    rng.shuffle(candidates)
    return replace(
        story,
        objects=tuple(objects),
        candidates=tuple(candidates),
        premise=And(
            tuple(
                replace(
                    atom, subject=names[atom.subject], reference=names[atom.reference]
                )
                for atom in story.premise.operands
            )
        ),
        target=names[story.target],
        reference=names[story.reference],
    )


def random_stories(
    seed=SEED, world_count=RANDOM_WORLD_STORIES, atom_count=RANDOM_ATOM_STORIES
):
    rng = random.Random(seed)
    result = []
    for recipe, count in (("random-world", world_count), ("random-atoms", atom_count)):
        for index in range(count):
            objects = tuple(f"Object{i}" for i in range(rng.randint(5, 8)))
            pairs = list(combinations(objects, 2))
            rng.shuffle(pairs)
            pairs = pairs[: rng.randint(2, min(len(pairs), len(objects) + 1))]
            # Coordinates choose valid qualitative relations only. They are not
            # additional constraints, distance assumptions, or supervised inputs.
            coordinates = {}
            while len(coordinates) < len(objects):
                point = (rng.randrange(5), rng.randrange(5))
                if point not in coordinates.values():
                    coordinates[objects[len(coordinates)]] = point
            atoms = []
            for a, b in pairs:
                if rng.random() < 0.5:
                    a, b = b, a
                direction = (
                    direction_between(coordinates[a], coordinates[b])
                    if recipe == "random-world"
                    else rng.choice(tuple(Direction))
                )
                atoms.append(direction_constraint(a, b, direction))
            reference = rng.choice(objects)
            candidates = tuple(obj for obj in objects if obj != reference)
            target = rng.choice(candidates)
            name = f"{recipe}-{index}"
            result.append(
                Story(
                    name,
                    name,
                    recipe,
                    objects,
                    And(tuple(atoms)),
                    target,
                    reference,
                    candidates,
                    rng.choice(tuple(Direction)),
                    (),
                )
            )
    return tuple(result)


def candidate_stories():
    generated = random_stories()
    rng = random.Random(SEED + 1)
    targeted = tuple(rename_story(s, rng) for s in construct_stories(SEED, 5))
    return (*generated, *targeted)


def build_candidate():
    stories = candidate_stories()
    pack = run_pilot(SEED, stories=stories, compact=True, stratified=True)
    pack["status"] = "study-candidate-unsealed"
    pack["config"] = {
        "seed": SEED,
        "world_stories": RANDOM_WORLD_STORIES,
        "unconstrained_atom_stories": RANDOM_ATOM_STORIES,
        "targeted_stories": 50,
        "attempt_budget": 200,
        "retry_to_quota": False,
        "split_policy": "source-recipe-stratified-families-v1",
        "compact_unique_direction": True,
        "object_count_random": [5, 8],
        "candidate_count_random": [4, 7],
        "menu_policy": "full-plus-one-partial-when-feasible; otherwise-one-menu",
    }
    pack["remaining_gates"] = [
        "Unsealed random-plus-targeted study candidate; no model training launched.",
        "All 33 semantic cells must be present in every split before preparation.",
        "Full paired agent inspection, pinned token admission and final integrity seal remain required.",
        "Count supports fixed memberships and pair conflicts only; unsupported cases are quarantined.",
        "Independent reference backend covers small Direction problems only.",
    ]
    return pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack = build_candidate()
    write_pilot(pack, args.output)
    print(json.dumps(pack["counts"], indent=2))
    if pack["counts"]["attempt_status"].get("verification-failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
