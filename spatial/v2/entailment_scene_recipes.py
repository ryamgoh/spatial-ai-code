"""All-scene controlled recipes and an explicit diagnostic fixed-slot mapping."""

import random
from collections import defaultdict
from dataclasses import replace

from spatial.v2.composition_feasibility import scramble, stable_int
from spatial.v2.composition_pilot import story_key
from spatial.v2.composition_scene_sampler import _core, family_id, sample_scaffold
from spatial.v2.solver import And, Direction, direction_constraint

PREVIOUS_VERSION = "all-scene-forest-v3"
VERSION = "all-scene-forest-v4"


def controlled_core(seed, depth, direction, matches):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(depth) is not int or not 1 <= depth <= 10:
        raise ValueError("depth must be an integer in 1-10")
    if type(matches) is not int or not 0 <= matches <= 4:
        raise ValueError("matches must be an integer in 0-4")
    if not isinstance(direction, Direction):
        raise TypeError("direction must be a Direction")
    scaffold = sample_scaffold(seed)
    core = _core(
        scaffold,
        depth,
        direction,
        matches,
        proof_pattern="homogeneous" if depth == 1 else "axis-split",
        shared_selection_support=matches > 1,
    )
    if depth > 1:
        original = _core(
            scaffold,
            depth,
            direction,
            matches,
            shared_selection_support=matches > 1,
        )
        endpoints = set(original.candidates[: max(1, matches)])

        def is_support(atom):
            return atom.subject in endpoints or atom.subject.startswith(
                ("Positive", "AxisLink")
            )

        # Keep the original negative forest intact. Rebuilding it with the
        # larger positive support consumes its budget and merges old families.
        atoms = tuple(a for a in core.premise.operands if is_support(a)) + tuple(
            a for a in original.premise.operands if not is_support(a)
        )
        objects = tuple(
            dict.fromkeys(name for a in atoms for name in (a.subject, a.reference))
        )
        core = replace(core, objects=objects, premise=And(atoms))
    directions = tuple(Direction)
    index = directions.index(direction)
    if depth > 1 and matches > 0 and index % 2 == 0:
        # The equality-support intermediate must not become a shallow answer.
        opposite = directions[(index + 4) % 8]
        core = replace(
            core,
            premise=And(
                tuple(
                    replace(
                        atom,
                        allowed=frozenset(
                            (opposite if atom.subject == "AxisLink0" else direction,)
                        ),
                    )
                    if atom.reference == "AxisLink0" or atom.subject == "AxisLink0"
                    else atom
                    for atom in core.premise.operands
                )
            ),
        )
    return core


def controlled_scene(seed, depth, direction, matches, *, story_id, source_family):
    """Keep fixed lineage separate from naming, depth, and cardinality variants."""
    core = controlled_core(seed, depth, direction, matches)
    opposite = tuple(Direction)[(tuple(Direction).index(direction) + 4) % 8]
    atoms = (*core.premise.operands, direction_constraint("Context", "O", opposite))
    objects = (*core.objects, "Context")
    distractors = ()
    if depth > 1:
        atoms += (direction_constraint("Context", "AxisLink0", opposite),)
        distractors = (len(atoms) - 1,)
    else:
        # Direct exclusions keep every membership at depth one. Extra acyclic
        # context links provide scene variety without changing the answer set.
        rng = random.Random(stable_int(f"{seed}/depth-one-context-v1"))
        contexts = ("Context", *(f"Context{i}" for i in range(1, rng.randint(1, 3))))
        objects = (*core.objects, *contexts)
        atoms = (*core.premise.operands,)
        redundant = []
        for i, node in enumerate(contexts):
            atoms += (direction_constraint(node, "O", opposite),)
            references = (*core.candidates, *contexts[:i])
            for reference in rng.sample(
                references, rng.randint(1, min(3, len(references)))
            ):
                redundant.append(len(atoms))
                atoms += (direction_constraint(node, reference, opposite),)
        distractors = tuple(redundant)
    if depth == 2:
        # Shallow support graphs can collapse across unrelated fixed seeds.
        # New context leaves vary the scene, but are directly excluded from T.
        rng = random.Random(stable_int(f"{seed}/depth-two-context-v1"))
        budget = max(0, min(4, 31 - len(objects), (32 - len(atoms)) // 2))
        count = rng.randint(min(2, budget), budget)
        contexts = []
        redundant = list(distractors)
        for i in range(count):
            node = f"Context{i + 1}"
            references = (*core.candidates, "Context", *contexts)
            atoms += (direction_constraint(node, "O", opposite),)
            limit = min(2, len(references), 32 - len(atoms) - 2 * (count - i - 1))
            for reference in rng.sample(references, rng.randint(1, limit)):
                redundant.append(len(atoms))
                atoms += (direction_constraint(node, reference, opposite),)
            contexts.append(node)
        objects = (*objects, *contexts)
        distractors = tuple(redundant)
    story = replace(
        core,
        story_id=story_id,
        source_family=source_family,
        recipe=f"{VERSION if depth == 2 else PREVIOUS_VERSION}/"
        f"{'homogeneous' if depth == 1 else 'axis-split'}",
        objects=objects,
        premise=And(atoms),
        distractors=distractors,
    )
    # Keep all unaffected depths byte-for-byte stable.
    return scramble(story, random.Random(stable_int(f"all-scene-forest-v2/{story_id}")))


def map_fixed_slots(entries, policy, *, selection_matches=None):
    """Replace only existing forest slots; do not add seeds, slots, or views.

    Selection cardinalities are explicit diagnostic inputs, not a frozen balance
    policy. Count retains each slot's original target. Original and repaired
    anchor equivalences are unioned so a repair cannot inflate family counts.
    """
    selection_matches = {} if selection_matches is None else selection_matches
    selected_ids = {
        e.story.story_id
        for e in entries
        if "which" in e.query_kinds and e.depth is not None
    }
    if set(selection_matches) - selected_ids:
        raise ValueError("Selection mapping contains a nonexistent fixed slot")
    parents, splits = {}, defaultdict(set)

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    slots = []
    anchor_cache = {}
    for entry in entries:
        batch_index, slot_index = (
            int(entry.story.story_id.split("-")[1][1:]),
            int(entry.story.story_id.rsplit("s", 1)[1]),
        )
        batch = policy["source_batches"][batch_index]
        if batch["generator"] != "sampled-forest":
            slots.append((entry, None))
            continue
        if (
            entry.split != batch["split"]
            or entry.depth != batch["depth"]
            or entry.query_kinds != tuple(batch["query_kinds"])
            or slot_index not in batch.get("indices", range(batch["count"]))
        ):
            raise ValueError("entry does not match its frozen source slot")
        seeds = batch.get("seeds", range(batch["seed"], batch["seed"] + batch["count"]))
        seed = seeds[slot_index]
        original_family = family_id(sample_scaffold(seed))
        if entry.story.source_family != original_family:
            raise ValueError("fixed source seed and lineage disagree")
        matches = (
            batch["matches"]
            if "count" in entry.query_kinds
            else selection_matches.get(entry.story.story_id, 1)
        )
        if "count" in entry.query_kinds and entry.story.story_id in selection_matches:
            raise ValueError("Selection overrides cannot change Count slot targets")
        direction = (
            Direction(batch["direction"])
            if "direction" in batch
            else tuple(Direction)[stable_int(f"{seed}/direction") % 8]
        )
        if seed not in anchor_cache:
            anchor = controlled_core(seed, 10, Direction.NORTH, 1)
            anchor_cache[seed] = story_key(anchor.objects, anchor.premise)
        anchor_id = f"{PREVIOUS_VERSION}/{anchor_cache[seed]}"
        union(original_family, anchor_id)
        splits[original_family].add(entry.split)
        slots.append(
            (
                entry,
                {
                    "story_id": entry.story.story_id,
                    "seed": seed,
                    "depth": entry.depth,
                    "split": entry.split,
                    "query_kinds": list(entry.query_kinds),
                    "matches": matches,
                    "direction": direction.value,
                    "original_family": original_family,
                    "repair_anchor": anchor_id,
                },
            )
        )
    grouped_splits = defaultdict(set)
    grouped_originals = defaultdict(set)
    for family, roles in splits.items():
        grouped_splits[root(family)].update(roles)
        grouped_originals[root(family)].add(family)
    if any(len(roles) > 1 for roles in grouped_splits.values()):
        raise ValueError("related repaired construction family crosses splits")
    mapped, manifest = [], []
    for entry, slot in slots:
        if slot is None:
            mapped.append(entry)
            continue
        family = min(grouped_originals[root(slot["original_family"])])
        story = controlled_scene(
            slot["seed"],
            slot["depth"],
            Direction(slot["direction"]),
            slot["matches"],
            story_id=entry.story.story_id,
            source_family=family,
        )
        mapped.append(replace(entry, story=story))
        manifest.append({**slot, "family": family, "recipe": story.recipe})
    return tuple(mapped), manifest
