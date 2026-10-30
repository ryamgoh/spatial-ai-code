"""Software fixtures for sampled scenes; seeds 0-31 are development sources."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from spatial.v2.composition_balance import topology_key
from spatial.v2.composition_contracts import VerificationError, analyze
from spatial.v2.composition_depth import measure_depth
from spatial.v2.composition_evidence import build_evidence, verify_evidence
from spatial.v2.composition_feasibility import verify_transformations
from spatial.v2.composition_production import Entry, admit, preflight, schedule
from spatial.v2.composition_scene_sampler import (
    _core,
    family_id,
    sample_scaffold,
    sample_scene,
)
from spatial.v2.composition_training import REVISION
from spatial.v2.solver import Direction, SpatialSolverV2

EMPTY_EXCLUSIONS = {"family_ids": [], "story_keys": []}
DIRECTIONS = tuple(Direction)


@pytest.fixture(scope="module")
def policy():
    root = Path(__file__).resolve().parents[3]
    policy = json.loads(
        (
            root / "experiments/spatialentail-composition-v1/production-config-v1.json"
        ).read_text()
    )
    # These fixtures test primary depth support; paired omission is covered by
    # the production tests, where escape views cannot fill depth quotas.
    policy["paired_omission_percent"] = 0
    return policy


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


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("depth", range(1, 11))
def test_full_semantics_depth_and_context_removal(depth, direction):
    seed = (depth + DIRECTIONS.index(direction)) % 32
    story = sample_scene(seed, depth, direction=direction)
    assert story == sample_scene(seed, depth, direction=direction)
    assert 4 <= len(story.candidates) <= 8
    assert len(story.objects) <= 31
    assert len(story.premise.operands) <= 32
    assert story.distractors
    solver = SpatialSolverV2(backend="z3")
    for index, problem in enumerate(story.problems()):
        semantics = analyze(problem, solver=solver)
        assert semantics.consistent
        expected = ({direction.value}, {story.target}, {1})[index]
        assert semantics.possible == expected
        assert semantics.certain == expected
        measured = measure_depth(problem, solver=solver)
        assert measured["depth"] == depth
        if index:
            assert any(a["answer_changed"] for a in measured["ablations"])
        verify_transformations(story, problem, semantics, solver)


def test_family_identity_ignores_direction_depth_and_decoration():
    families = {
        sample_scene(7, depth, direction=direction, variant=variant).source_family
        for depth in (1, 8, 9, 10)
        for direction in DIRECTIONS
        for variant in (0, 1)
    }
    assert len(families) == 1
    assert sample_scene(7, 8) != sample_scene(7, 8, variant=1)


def test_core_graphs_vary_without_counting_context_or_renamings():
    scaffolds = [sample_scaffold(seed) for seed in range(32)]
    cores = [_core(s, 5, Direction.NORTH) for s in scaffolds]
    assert len({family_id(s) for s in scaffolds}) >= 16
    assert len({topology_key(s.problems()[1]) for s in cores}) >= 16
    assert len({len(s.candidates) for s in cores}) >= 4
    assert any(len(s.premise.operands) > len(s.objects) - 1 for s in cores)
    assert any(
        sum(atom.reference == obj for atom in s.premise.operands) > 1
        for s in cores
        for obj in s.objects
        if obj != s.reference
    )


def test_equal_scaffolds_have_equal_ids_without_seed_identity():
    scaffold = sample_scaffold(3)
    assert family_id(scaffold) == family_id(replace(scaffold))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"seed": -1, "depth": 2},
        {"seed": True, "depth": 2},
        {"seed": 0, "depth": 0},
        {"seed": 0, "depth": 11},
        {"seed": 0, "depth": True},
        {"seed": 0, "depth": 2, "variant": -1},
        {"seed": 0, "depth": 2, "direction": "Northeast"},
    ),
)
def test_invalid_parameters_fail(kwargs):
    with pytest.raises((ValueError, TypeError)):
        sample_scene(**kwargs)


def test_production_schedule_preserves_families_and_rejects_cross_role_variants(policy):
    policy = {
        **policy,
        "source_batches": [
            {
                "generator": "sampled-forest",
                "seed": 7,
                "count": 1,
                "depth": depth,
                "split": role,
                "direction": direction.value,
                "variant": variant,
                "matches": 1,
                "query_kinds": ["direction", "which", "count"],
            }
            for depth, role, direction, variant in (
                (2, "train", Direction.NORTH, 0),
                (9, "extrapolation-test", Direction.SOUTHWEST, 1),
            )
        ],
    }
    entries = schedule(policy)
    assert entries == schedule(policy)
    assert len({e.story.source_family for e in entries}) == 1
    with pytest.raises(ValueError, match="family crosses"):
        preflight(entries, EMPTY_EXCLUSIONS)


def test_development_family_exclusion_applies_to_new_sampler():
    story = sample_scene(5, 9)
    with pytest.raises(ValueError, match="development source"):
        preflight(
            [Entry(story, "extrapolation-test", 9, ("which",))],
            {"family_ids": [story.source_family], "story_keys": []},
        )


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("kind,index", (("direction", 0), ("which", 1), ("count", 2)))
def test_all_direction_depth_ten_readable_admission(
    policy, tokenizer, direction, kind, index
):
    story = sample_scene(DIRECTIONS.index(direction), 10, direction=direction)
    entry = Entry(story, "extrapolation-test", 10, (kind,))
    result = admit(
        entry, story.problems()[index], policy, tokenizer, SpatialSolverV2(backend="z3")
    )
    assert result["status"] == "accepted", result.get("reason")
    assert all(row["substantive_depth_support"] for row in result["rows"])
    assert len(result["rows"]) == 3
    assert len(result["tokens"]) == 6
    assert all(t["admitted"] for t in result["tokens"])


@pytest.mark.parametrize("index", range(3))
def test_label_checks_skip_map_construction_but_evidence_requires_maps(
    index, monkeypatch
):
    story = sample_scene(0, 2)
    problem = story.problems()[index]
    solver = SpatialSolverV2(backend="z3")
    evidence = build_evidence(problem, solver=solver)
    expected = evidence.semantics
    assert set(expected.witnesses) == expected.possible

    def reject_map_construction(*args, **kwargs):
        pytest.fail("label-only verification must not construct canonical maps")

    monkeypatch.setattr(
        solver._engine, "_canonical_coordinates", reject_map_construction
    )
    labels = analyze(problem, solver=solver, include_witnesses=False)
    assert (labels.consistent, labels.possible, labels.certain) == (
        expected.consistent,
        expected.possible,
        expected.certain,
    )
    assert labels.witnesses == {}
    assert measure_depth(problem, solver=solver)["depth"] == 2
    verify_transformations(story, problem, labels, solver)
    verify_evidence(evidence, solver=solver)
    with pytest.raises(VerificationError, match="stored witnesses"):
        verify_evidence(replace(evidence, semantics=labels), solver=solver)


@pytest.mark.parametrize("matches", range(5))
@pytest.mark.parametrize("depth", (1, 8, 10))
def test_fixed_count_variants_preserve_depth_and_family(matches, depth):
    story = sample_scene(3, depth, direction=Direction.NORTHEAST, matches=matches)
    assert story.source_family == sample_scene(3, 2).source_family
    solver = SpatialSolverV2(backend="z3")
    selection = analyze(story.problems()[1], solver=solver, include_witnesses=False)
    count = analyze(story.problems()[2], solver=solver, include_witnesses=False)
    assert len(selection.certain) == matches
    assert selection.certain == selection.possible
    assert count.possible == count.certain == {matches}
    assert measure_depth(story.problems()[2], solver=solver)["depth"] == depth
    verify_transformations(story, story.problems()[2], count, solver)


@pytest.mark.parametrize("matches", (0, 2, 4))
def test_fixed_count_variants_pass_readable_admission(policy, tokenizer, matches):
    story = sample_scene(3, 10, direction=Direction.NORTHEAST, matches=matches)
    entry = Entry(story, "extrapolation-test", 10, ("count",))
    result = admit(
        entry, story.problems()[2], policy, tokenizer, SpatialSolverV2(backend="z3")
    )
    assert result["status"] == "accepted", result.get("reason")
    assert all(row["substantive_depth_support"] for row in result["rows"])


@pytest.mark.parametrize("matches", (-1, 9, True))
def test_invalid_fixed_count_rejected(matches):
    with pytest.raises(ValueError, match="matches"):
        sample_scene(3, 2, matches=matches)
