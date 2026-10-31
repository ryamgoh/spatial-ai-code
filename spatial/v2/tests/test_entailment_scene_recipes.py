"""All-scene Count targets and fixed-slot reuse must retain their meaning."""

from dataclasses import replace

import pytest

from spatial.v2.composition_pilot import story_key
from spatial.v2.composition_production import Entry
from spatial.v2.composition_scene_sampler import _core, sample_scaffold, sample_scene
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_dataset import (
    measure_depth,
    uniqueness_failure_depth,
    verify_transformations,
)
from spatial.v2.entailment_evidence import build_evidence
from spatial.v2.entailment_scene_protocol import all_scene_problem
from spatial.v2.entailment_scene_recipes import (
    controlled_core,
    controlled_scene,
    map_fixed_slots,
)
from spatial.v2.solver import Direction


@pytest.mark.parametrize("depth", (1, 2, 8, 10))
@pytest.mark.parametrize("direction", tuple(Direction))
@pytest.mark.parametrize("matches", range(5))
def test_all_scene_count_and_selection(depth, direction, matches):
    story = controlled_scene(
        2026111094,
        depth,
        direction,
        matches,
        story_id="fixed-test",
        source_family="original-family",
    )
    solver = QuestionSolver()
    selection = all_scene_problem(story.problems()[1])
    count = all_scene_problem(story.problems()[2])
    for problem in (selection, count):
        evidence = build_evidence(problem, solver=solver)
        assert evidence.semantics.consistent
        if problem is count:
            assert evidence.semantics.entailed == frozenset((matches,))
        else:
            assert len(evidence.semantics.entailed) == matches
            assert uniqueness_failure_depth(evidence) == (
                depth if matches > 1 else None
            )
        if problem is count or matches > 0:
            assert measure_depth(problem, solver=solver)["depth"] == depth
        verify_transformations(
            story, problem, evidence.semantics, solver, scene_scope=True
        )
    assert story.source_family == "original-family"
    assert len(story.objects) <= 32
    assert len(story.premise.operands) <= 32


def fixtures():
    batches = [
        {
            "generator": "sampled-forest",
            "seed": 2026111094,
            "count": 1,
            "split": "train",
            "depth": depth,
            "direction": "North",
            "matches": matches,
            "query_kinds": kinds,
        }
        for depth, matches, kinds in (
            (2, 1, ["direction", "which"]),
            (8, 1, ["direction", "which"]),
            (8, 0, ["count"]),
        )
    ]
    entries = tuple(
        Entry(
            replace(
                sample_scene(
                    b["seed"],
                    b["depth"],
                    direction=Direction.NORTH,
                    matches=b["matches"],
                ),
                story_id=f"production-b{i}-s0",
            ),
            b["split"],
            b["depth"],
            tuple(b["query_kinds"]),
        )
        for i, b in enumerate(batches)
    )
    return entries, {"source_batches": batches}


def test_mapping_preserves_slots_seeds_and_family_not_presentation():
    entries, policy = fixtures()
    mapped, manifest = map_fixed_slots(
        entries,
        policy,
        selection_matches={"production-b0-s0": 2, "production-b1-s0": 4},
    )
    assert len(mapped) == len(entries)
    assert [e.story.story_id for e in mapped] == [e.story.story_id for e in entries]
    assert [e.query_kinds for e in mapped] == [e.query_kinds for e in entries]
    assert {r["seed"] for r in manifest} == {2026111094}
    assert len({e.story.source_family for e in mapped}) == 1
    assert [r["matches"] for r in manifest] == [2, 4, 0]
    assert map_fixed_slots(entries, policy)[1][-1]["matches"] == 0
    assert entries[0].story.recipe == "sampled-forest-v1"


def test_mapping_rejects_new_slots_and_count_target_overrides():
    entries, policy = fixtures()
    for changes in ({"new-slot": 2}, {"production-b2-s0": 2}):
        with pytest.raises(ValueError):
            map_fixed_slots(entries, policy, selection_matches=changes)


def test_mapping_checks_original_lineage_and_related_split_isolation():
    entries, policy = fixtures()
    bad = replace(entries[0], story=replace(entries[0].story, source_family="wrong"))
    with pytest.raises(ValueError, match="lineage"):
        map_fixed_slots((bad, *entries[1:]), policy)
    policy["source_batches"][1]["split"] = "dev"
    entries = (entries[0], replace(entries[1], split="dev"), entries[2])
    with pytest.raises(ValueError, match="crosses splits"):
        map_fixed_slots(entries, policy)


def test_repair_does_not_truncate_the_original_negative_forest():
    anchors = []
    for seed in (2026112511, 2026113273):
        original = _core(sample_scaffold(seed), 10, Direction.NORTH, 1)
        repaired = controlled_core(seed, 10, Direction.NORTH, 1)
        original_negative = {
            a
            for a in original.premise.operands
            if a.subject != "C0" and not a.subject.startswith("Positive")
        }
        assert original_negative <= set(repaired.premise.operands)
        anchors.append(story_key(repaired.objects, repaired.premise))
    assert anchors[0] != anchors[1]


@pytest.mark.parametrize(
    "direction,seeds",
    (
        (Direction.WEST, (2026111095, 2026112960)),
        (Direction.NORTH, (2026111518, 2026111360)),
    ),
)
def test_depth_two_context_keeps_fixed_count_and_selection_slots_distinct(
    direction, seeds
):
    keys = [
        story_key(story.objects, story.premise)
        for seed in seeds
        for story in [
            controlled_scene(
                seed,
                2,
                direction,
                4,
                story_id=f"regression-{seed}",
                source_family=f"original-{seed}",
            )
        ]
    ]
    assert keys[0] != keys[1]


@pytest.mark.parametrize(
    "direction,seed",
    (
        (Direction.WEST, 2026111095),
        (Direction.WEST, 2026112960),
        (Direction.NORTH, 2026111518),
        (Direction.NORTH, 2026111360),
    ),
)
def test_repaired_collision_cases_keep_count_four_and_depth_two(direction, seed):
    story = controlled_scene(
        seed, 2, direction, 4, story_id=f"fixed-{seed}", source_family="fixed-family"
    )
    solver = QuestionSolver()
    for index in (1, 2):
        problem = all_scene_problem(story.problems()[index])
        evidence = build_evidence(problem, solver=solver)
        assert evidence.semantics.consistent
        assert (
            len(evidence.semantics.entailed) == 4
            if index == 1
            else evidence.semantics.entailed == frozenset((4,))
        )
        assert measure_depth(problem, solver=solver)["depth"] == 2
        if index == 1:
            assert uniqueness_failure_depth(evidence) == 2
        verify_transformations(
            story, problem, evidence.semantics, solver, scene_scope=True
        )


def test_depth_two_context_respects_scene_and_place_name_budgets():
    for seed in range(64):
        for matches in range(5):
            story = controlled_scene(
                seed,
                2,
                Direction.NORTHEAST,
                matches,
                story_id="budget-test",
                source_family="fixed-family",
            )
            assert len(story.objects) <= 31
            assert len(story.premise.operands) <= 32
            assert len(story.candidates) <= 8
