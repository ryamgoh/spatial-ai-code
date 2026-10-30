"""Label-blind allocation checks using existing development scaffolds only."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from spatial.v2 import composition_allocation as allocation
from spatial.v2 import composition_production as production
from spatial.v2.composition_pilot import renamed_problem, story_key
from spatial.v2.composition_scene_sampler import sample_scene
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import Direction

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/spatialentail-composition-v1/production-config-v1.json"
EMPTY = {"family_ids": [], "story_keys": []}


@pytest.fixture
def policy():
    return json.loads(CONFIG.read_text())


def batch(**changes):
    return {
        "generator": "sampled-forest",
        "seed": 3,
        "count": 2,
        "seeds": [3, 7],
        "split": "train",
        "query_kinds": ["count"],
        "depth": 2,
        "matches": 0,
        "proof_pattern": "axis-split",
        "direction": "Northeast",
        **changes,
    }


def entries():
    first = production.Entry(
        replace(sample_scene(3, 2), story_id="production-b0-s0"),
        "train",
        2,
        ("direction",),
    )
    second = production.Entry(
        replace(sample_scene(7, 2), story_id="production-b1-s0"),
        "dev",
        2,
        ("direction",),
    )
    return first, second


def test_explicit_seed_list_preserves_original_family_and_draw(policy):
    policy["source_batches"] = [batch()]
    scheduled = production.schedule(policy)
    assert len(scheduled) == 2
    for entry in scheduled:
        index = int(entry.story.story_id.rsplit("-s", 1)[1])
        original = sample_scene(
            (3, 7)[index],
            2,
            matches=0,
            proof_pattern="axis-split",
            direction=Direction.NORTHEAST,
        )
        assert replace(entry.story, story_id=original.story_id) == original


@pytest.mark.parametrize("seeds", ([3], [3, 3], [-1, 3], [True, 3], "3,7"))
def test_invalid_seed_lists_fail_before_sampling(policy, seeds):
    policy["source_batches"] = [batch(seeds=seeds)]
    with pytest.raises(ValueError, match="batch seeds"):
        production.schedule(policy)


def test_non_sampled_generator_cannot_use_explicit_seeds(policy):
    policy["source_batches"] = [batch(generator="random-world", depth=None)]
    with pytest.raises(ValueError, match="only supported"):
        production.schedule(policy)


@pytest.mark.parametrize("indices", ([2], [-1], [0, 0], [1, 0], [False], "0"))
def test_invalid_structural_indices_fail(policy, indices):
    policy["source_batches"] = [batch(indices=indices)]
    with pytest.raises(ValueError, match="batch indices"):
        production.schedule(policy)


def test_structural_index_filter_keeps_ids_and_order_without_replacement(policy):
    policy["source_batches"] = [batch()]
    before = production.schedule(policy)
    policy["source_batches"][0]["indices"] = [1]
    after = production.schedule(policy)
    assert after == tuple(e for e in before if e.story.story_id.endswith("-s1"))
    policy["source_batches"][0]["indices"] = []
    assert production.schedule(policy) == ()


def test_numeric_approval_does_not_authorize_generation(policy):
    assert policy["diversity_checks"]["status"] == "approved"
    assert not policy["approval"]["production_freeze"]
    assert not policy["approval"]["training_authorized"]
    assert not any("await approval" in b for b in production.plan(policy)["blockers"])


def test_fixed_pool_assignment_is_deterministic_and_groups_seed_collisions(
    policy, monkeypatch
):
    monkeypatch.setattr(allocation, "sample_scaffold", lambda seed: seed // 2)
    monkeypatch.setattr(allocation, "family_id", lambda scaffold: f"family-{scaffold}")
    first = allocation.allocate_pool(policy, EMPTY, start=0, size=4096)
    assert first == allocation.allocate_pool(policy, EMPTY, start=0, size=4096)
    seeds, report = first
    assert report["distinct_families"] == 2048
    assert all(seed % 2 == 0 for group in seeds.values() for seed in group)
    assert len({seed for group in seeds.values() for seed in group}) == 2048
    assert set(report["family_roles"].values()) == set(production.ROLES)


def test_fixed_pool_excludes_all_development_lineages(policy, monkeypatch):
    monkeypatch.setattr(allocation, "sample_scaffold", lambda seed: seed)
    monkeypatch.setattr(allocation, "family_id", lambda seed: f"family-{seed}")
    excluded = {"family_ids": ["family-1", "family-4"], "story_keys": []}
    seeds, report = allocation.allocate_pool(policy, excluded, start=0, size=4096)
    assert all(1 not in group and 4 not in group for group in seeds.values())
    assert report["excluded_development_families"] == ["family-1", "family-4"]


def test_small_fixed_pool_fails_instead_of_drawing_more(policy, monkeypatch):
    calls = []
    monkeypatch.setattr(
        allocation, "sample_scaffold", lambda seed: calls.append(seed) or seed
    )
    monkeypatch.setattr(allocation, "family_id", str)
    with pytest.raises(ValueError, match="no refill"):
        allocation.allocate_pool(policy, EMPTY, start=0, size=3)
    assert calls == [0, 1, 2]


def test_source_plan_exactly_triples_approved_quotas_without_sampling(
    policy, monkeypatch
):
    monkeypatch.setattr(allocation, "sample_scaffold", lambda _: pytest.fail("sampled"))
    pools = {role: list(range(1000)) for role in production.ROLES}
    batches = allocation.source_batches(policy, pools)
    counts = {}
    count_values = {}
    for b in batches:
        for kind in b["query_kinds"]:
            depth = b.get("depth", "semantic")
            key = f"{b['split']}/{kind}/{depth}"
            counts[key] = counts.get(key, 0) + b["count"]
            if kind == "count" and depth != "semantic":
                ck = f"{key}/{b['matches']}"
                count_values[ck] = count_values.get(ck, 0) + b["count"]
    assert counts == {k: 3 * n for k, n in production.quotas(policy).items()}
    assert count_values == {
        k: 3 * n for k, n in production.count_quotas(policy).items()
    }
    assert sum(counts.values()) == 32400
    assert all(
        b.get("matches") is not None
        for b in batches
        if b["generator"] == "sampled-forest"
    )
    assert {
        b.get("pattern") for b in batches if b["generator"] == "sampled-targeted"
    } == set(allocation.TARGETED_WEIGHTS)


def test_proof_cycle_supports_all_directions_and_valid_patterns():
    assert len(allocation.proof_options(1)) == 8
    options = allocation.proof_options(8)
    assert len(options) == 20
    assert sum(p != "homogeneous" for p, _ in options) == 12
    assert {d for _, d in options} == {d.value for d in Direction}
    assert all(d not in production.CARDINALS for p, d in options if p == "mixed-chain")


def test_cross_split_scene_collision_quarantines_every_related_family():
    first, second = entries()
    renamed, _ = renamed_problem(first.story.problems()[0])
    second = replace(
        second,
        story=replace(
            second.story,
            objects=renamed.objects,
            premise=renamed.premise,
        ),
    )
    relative = replace(
        first,
        story=replace(
            sample_scene(3, 8),
            story_id="production-b2-s0",
        ),
        depth=8,
    )
    removed = allocation.structural_exclusions((first, second, relative), EMPTY)
    assert {r["story_id"] for r in removed} == {
        first.story.story_id,
        second.story.story_id,
        relative.story.story_id,
    }
    assert all("cross-split-story" in r["reasons"] for r in removed)


def test_development_story_collision_propagates_through_same_split_copy():
    first, second = entries()
    copy = replace(
        second,
        split="train",
        story=replace(
            second.story,
            objects=first.story.objects,
            premise=first.story.premise,
        ),
    )
    related = replace(
        copy,
        story=replace(
            sample_scene(7, 8),
            story_id="production-b2-s0",
        ),
        depth=8,
    )
    removed = allocation.structural_exclusions(
        (first, copy, related),
        {"family_ids": [first.story.source_family], "story_keys": []},
    )
    assert len(removed) == 3
    assert "related-to-quarantined-family" in removed[-1]["reasons"]


def test_unrelated_entries_remain_and_exclusion_does_not_mutate_batches(policy):
    policy["source_batches"] = [batch()]
    scheduled = production.schedule(policy)
    assert allocation.structural_exclusions(scheduled, EMPTY) == []
    original = deepcopy(policy["source_batches"])
    removed = [{"story_id": "production-b0-s0"}]
    filtered = allocation.apply_exclusions(original, scheduled, removed)
    assert original == policy["source_batches"]
    assert filtered[0]["indices"] == [1]
    assert len(production.schedule({**policy, "source_batches": filtered})) == 1


def test_development_inventory_includes_rejected_problem_and_records_hashes(
    tmp_path, monkeypatch
):
    folder = tmp_path / "old"
    folder.mkdir()
    first, _ = entries()
    artifact = folder / "attempts.jsonl"
    artifact.write_text(
        json.dumps(
            {
                "status": "unsupported",
                "family_id": "depth-extrapolation-family-2",
                "problem": tagged_dataclass_to_dict(first.story.problems()[0]),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(allocation, "sample_scaffold", lambda seed: seed)
    monkeypatch.setattr(allocation, "family_id", lambda seed: f"fixture-{seed}")
    monkeypatch.setattr(allocation, "development_scenes", lambda: (first.story,))
    inventory = allocation.development_inventory(tmp_path)
    assert (
        story_key(first.story.objects, first.story.premise) in inventory["story_keys"]
    )
    assert "controlled-depth/sample-2" in inventory["family_ids"]
    assert "fixture-31" in inventory["family_ids"]
    assert inventory["sources_sha256"][
        "old/attempts.jsonl"
    ] == production.artifact_digest(artifact)
    assert inventory["software_fixture_scope"]["reconstructed_sampler_scenes"] == 1


def test_unsaved_fixture_reconstruction_includes_exact_context_variants():
    stories = list(allocation.development_scenes())
    assert sample_scene(7, 9, direction=Direction.SOUTHWEST, variant=1) in stories
    assert (
        sample_scene(
            3,
            10,
            direction=Direction.NORTHEAST,
            matches=0,
            proof_pattern="axis-split",
        )
        in stories
    )
    assert (
        allocation.sample_targeted(7, "inconsistent-cycle", span=3, variant=1)
        in stories
    )
    allowed = {
        allocation.family_id(allocation.sample_scaffold(seed)) for seed in range(32)
    }
    assert {s.source_family for s in stories} <= allowed | {"unassigned"}


def test_substantive_floors_are_implied_by_existing_controlled_targets(policy):
    rules = allocation.substantive_minima(policy)
    assert len(rules) == 36
    assert all(r["minimum_bases"] > 0 for r in rules)
    assert all(r["match"]["menu_role"] == "primary" for r in rules)
    assert {r["minimum_bases"] for r in rules if r["match"]["split"] == "train"} == {
        1400
    }


def test_supply_report_counts_remaining_slots_and_value_shortfalls(policy):
    policy["source_batches"] = [batch(indices=[1])]
    report = allocation.supply_report(production.schedule(policy), policy)
    assert report["scheduled_bases"] == report["scheduled_stories"] == 1
    assert report["scheduled_by_stratum"] == {"train/count/2": 1}
    assert report["requested_count_value_supply"] == {"train/count/2/0": 1}
    assert report["nonhomogeneous_supply"] == {"train/count/2": 1}
    targets = production.count_quotas(policy)
    assert report["insufficient_count_value_supply"]["train/count/2/0"] == (
        targets["train/count/2/0"] - 1
    )
    assert (
        report["insufficient_count_value_supply"]["train/count/2/1"]
        == (targets["train/count/2/1"])
    )
    assert report["scene_bounds_failures"] == []


def test_supply_family_counts_ignore_related_variants(policy):
    policy["source_batches"] = [
        batch(count=1, seeds=[3]),
        batch(count=1, seeds=[3], matches=2),
    ]
    report = allocation.supply_report(production.schedule(policy), policy)
    assert report["scheduled_bases"] == 2
    assert report["families_by_stratum"] == {"train/count/2": 1}


def test_supply_report_keeps_bound_failures_visible(policy):
    policy["source_batches"] = [batch(count=1, seeds=[3])]
    policy["scene_limits"]["objects"] = 1
    report = allocation.supply_report(production.schedule(policy), policy)
    assert report["scene_bounds_failures"] == ["production-b0-s0"]
    assert report["scheduled_bases"] == 1


def test_preflight_cli_does_not_freeze_generate_or_load_tokenizer(
    tmp_path, policy, monkeypatch, capsys
):
    policy["source_batches"] = [batch()]
    policy["attempt_budget"] = 2
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps(EMPTY))
    policy["development_exclusions"] = {
        "path": exclusions.name,
        "sha256": production.artifact_digest(exclusions),
    }
    policy["coverage_minima"] = allocation.substantive_minima(policy)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(policy))
    monkeypatch.setattr(
        production, "SpatialSolverV2", lambda **_: pytest.fail("solver")
    )
    monkeypatch.setattr(production, "generate", lambda *a: pytest.fail("generation"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "production",
            "--config",
            str(config),
            "--preflight",
        ],
    )
    production.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "structural-preflight-passed"
    assert result["scheduled_bases"] == 2
    assert result["production_freeze"] is False
    assert result["solver_admission"] == "not-run"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "config.json",
        "exclusions.json",
    ]


def test_preflight_rejects_tampered_exclusions(tmp_path, policy, monkeypatch):
    policy["source_batches"] = [batch()]
    policy["coverage_minima"] = allocation.substantive_minima(policy)
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps(EMPTY))
    policy["development_exclusions"] = {"path": exclusions.name, "sha256": "0" * 64}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(policy))
    monkeypatch.setattr(
        "sys.argv", ["production", "--config", str(config), "--preflight"]
    )
    with pytest.raises(SystemExit) as exc:
        production.main()
    assert exc.value.code == 2
