"""Targeted sampler software fixtures; seeds 0-31 are development lineages."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from spatial.v2.composition_balance import topology_key
from spatial.v2.composition_contracts import analyze
from spatial.v2.composition_feasibility import verify_transformations
from spatial.v2.composition_pilot import construct_stories, story_key
from spatial.v2.composition_production import (
    KINDS,
    Entry,
    admit,
    generate,
    preflight,
    schedule,
    source_for,
)
from spatial.v2.composition_scene_sampler import sample_scene
from spatial.v2.composition_targeted_sampler import PATTERNS, sample_targeted
from spatial.v2.composition_training import REVISION
from spatial.v2.solver import And, Direction, SpatialSolverV2

ROOT = Path(__file__).resolve().parents[3]
EMPTY_EXCLUSIONS = {"family_ids": [], "story_keys": []}


@pytest.fixture
def policy():
    value = json.loads(
        (
            ROOT / "experiments/spatialentail-composition-v1/production-config-v1.json"
        ).read_text()
    )
    value["paired_omission_percent"] = 100
    return value


@pytest.fixture(scope="module")
def tokenizer():
    snapshot = (
        Path.home()
        / f".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/{REVISION}"
    )
    if not snapshot.is_dir():
        pytest.skip("pinned tokenizer unavailable locally")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("direction", tuple(Direction))
def test_semantics_across_all_directions(pattern, direction):
    seed = tuple(Direction).index(direction)
    span = 2 + seed % 2
    story = sample_targeted(seed, pattern, direction=direction, span=span)
    assert story == sample_targeted(seed, pattern, direction=direction, span=span)
    assert 4 <= len(story.candidates) <= 8
    assert len(story.objects) <= 32 and len(story.premise.operands) <= 32
    solver = SpatialSolverV2(backend="z3")
    results = [
        analyze(p, solver=solver, include_witnesses=False) for p in story.problems()
    ]
    if pattern in ("inconsistent-cycle", "coincidence"):
        assert all(not s.consistent and not s.possible for s in results)
    elif pattern == "contingent":
        assert all(s.consistent for s in results)
        assert results[0].possible == {d.value for d in Direction}
        assert results[1].possible == {story.target} and not results[1].certain
        assert results[2].possible == {0, 1}
    else:
        matches = (
            0 if pattern == "empty" else 2 + seed % 3 if pattern == "multiple" else 1
        )
        assert all(s.consistent for s in results)
        expected_direction = tuple(Direction)[
            (tuple(Direction).index(direction) + (4 if matches == 0 else 0)) % 8
        ]
        assert results[0].possible == {expected_direction.value}
        assert results[1].possible == results[1].certain
        assert len(results[1].possible) == matches
        assert results[2].possible == {matches}
    # Verify full semantic invariance under names/order and individual/joint
    # context removal, including global consistency for contradiction cases.
    verify_transformations(story, story.problems()[2], results[2], solver)


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("kind", KINDS)
def test_real_readable_and_token_admission(policy, tokenizer, pattern, kind):
    story = sample_targeted(3, pattern, direction=Direction.NORTHEAST, span=2)
    entry = Entry(story, "dev", None, (kind,))
    result = admit(
        entry,
        story.problems()[KINDS.index(kind)],
        policy,
        tokenizer,
        SpatialSolverV2(backend="z3"),
    )
    assert result["status"] == "accepted", result.get("reason")
    assert result["source"] == "targeted"
    assert len(result["rows"]) in (3, 6)
    assert len(result["tokens"]) == len(result["rows"]) * 2
    assert all(t["admitted"] and t["target_tokens"] > 0 for t in result["tokens"])
    assert all(
        set(row["references"]) == {"natural", "symbolic"} for row in result["rows"]
    )
    assert all(row["family_id"] == story.source_family for row in result["rows"])


def test_family_shared_across_semantic_direction_span_and_context_variants():
    expected = sample_scene(7, 10).source_family
    for pattern in PATTERNS:
        for direction in (Direction.NORTH, Direction.NORTHEAST, Direction.SOUTHWEST):
            for span in (2, 3):
                for variant in (0, 1):
                    assert (
                        sample_targeted(
                            7, pattern, span=span, direction=direction, variant=variant
                        ).source_family
                        == expected
                    )
    assert (
        sample_targeted(7, "multiple", matches=2).source_family
        == sample_targeted(7, "multiple", matches=4).source_family
    )


def test_related_targeted_or_controlled_sources_cannot_cross_splits():
    story = sample_targeted(7, "contingent")
    base = Entry(story, "train", None, ("which",))
    variants = (
        sample_targeted(7, "inconsistent-cycle", span=3, variant=1),
        sample_scene(7, 3, direction=Direction.WEST),
    )
    for other in variants:
        with pytest.raises(ValueError, match="family crosses"):
            preflight([base, Entry(other, "dev", None, ("count",))], EMPTY_EXCLUSIONS)
    with pytest.raises(ValueError, match="development source"):
        preflight(
            [replace(base, split="dev")],
            {"family_ids": [story.source_family], "story_keys": []},
        )


def core_problem(story):
    problem = story.problems()[1]
    atoms = tuple(
        a for i, a in enumerate(story.premise.operands) if i not in story.distractors
    )
    objects = {o for atom in atoms for o in (atom.subject, atom.reference)}
    objects.update(story.candidates)
    objects.add(story.reference)
    return replace(problem, objects=tuple(sorted(objects)), premise=And(atoms))


def test_structure_variation_is_not_only_added_context():
    stories = [
        sample_targeted(seed, "contingent", span=3, direction=Direction.NORTH)
        for seed in range(32)
    ]
    cores = [core_problem(s) for s in stories]
    assert len({s.source_family for s in stories}) == 32
    assert len({topology_key(p) for p in cores}) >= 16
    legacy = {
        story_key(p.objects, p.premise)
        for s in construct_stories(20261029, 5)
        for p in [core_problem(s)]
    }
    assert all(story_key(p.objects, p.premise) not in legacy for p in cores)


def test_cycle_pair_differs_in_one_relation_not_its_family():
    consistent = sample_targeted(3, "consistent-cycle", direction=Direction.EAST)
    inconsistent = sample_targeted(3, "inconsistent-cycle", direction=Direction.EAST)
    assert consistent.source_family == inconsistent.source_family
    assert consistent.objects == inconsistent.objects
    assert consistent.distractors == inconsistent.distractors
    changes = [
        (a, b)
        for a, b in zip(consistent.premise.operands, inconsistent.premise.operands)
        if a != b
    ]
    assert len(changes) == 1
    assert changes[0][0].subject == changes[0][1].subject
    assert changes[0][0].reference == changes[0][1].reference


def test_schedule_routes_one_draw_per_requested_story(policy, tokenizer):
    policy["source_batches"] = [
        {
            "generator": "sampled-targeted",
            "seed": 3,
            "count": 2,
            "pattern": "contingent",
            "span": 2,
            "split": "dev",
            "query_kinds": ["which"],
        }
    ]
    entries = schedule(policy)
    assert len(entries) == 2 and entries == schedule(policy)
    assert all(e.depth is None and source_for(e) == "targeted" for e in entries)
    records = list(
        generate(
            entries, policy, EMPTY_EXCLUSIONS, tokenizer, SpatialSolverV2(backend="z3")
        )
    )
    assert all(r["status"] == "accepted" for r in records)
    assert all(r["source_stratum"] == "dev/which/targeted" for r in records)


def test_targeted_source_cannot_claim_controlled_depth(policy):
    policy["source_batches"] = [
        {
            "generator": "sampled-targeted",
            "seed": 3,
            "count": 1,
            "pattern": "singleton",
            "depth": 2,
            "split": "train",
            "query_kinds": ["direction"],
        }
    ]
    with pytest.raises(ValueError, match="cannot claim a depth"):
        schedule(policy)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"seed": -1},
        {"seed": True},
        {"pattern": "unknown"},
        {"span": 0},
        {"span": 4},
        {"span": True},
        {"variant": -1},
        {"variant": True},
        {"direction": "North"},
        {"matches": 2},
        {"pattern": "multiple", "matches": 1},
        {"pattern": "multiple", "matches": 5},
        {"pattern": "multiple", "matches": True},
        {"pattern": "consistent-cycle", "span": 1},
    ),
)
def test_invalid_requests_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        sample_targeted(**{"seed": 3, "pattern": "singleton", **kwargs})


@pytest.mark.parametrize("matches", (2, 3, 4))
def test_multiple_match_counts(matches):
    story = sample_targeted(3, "multiple", matches=matches)
    semantics = analyze(story.problems()[2], include_witnesses=False)
    assert semantics.possible == {matches}
