"""Diagnostic balance checks must expose conflicts without changing policy."""

import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from spatial.v2.composition_production import Entry
from spatial.v2.entailment_production import all_scene_diagnostic_policy
from spatial.v2.entailment_scene_recipes import controlled_scene
from spatial.v2.solver import Direction

DIRECTORY = (
    Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_reserve_diagnostic", DIRECTORY / "check_repaired_fixed_reserve.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def test_quarter_multi_selection_conflicts_with_all_substantive_unique_extrapolation():
    policy = json.loads((DIRECTORY / "entailment-config-frozen-v1.json").read_text())
    original = deepcopy(policy)
    assert MODULE.selection_balance_conflicts(policy) == [
        {
            "split": "extrapolation-test",
            "selection_bases": 400,
            "proposed_multi_bases": 100,
            "maximum_unique_substantive_bases": 300,
            "unchanged_unique_substantive_minimum": 400,
        }
    ]
    assert policy == original


def test_decision_depth_is_not_mistaken_for_substantive_unique_output():
    policy = {
        "base_targets": dict.fromkeys(MODULE.historical.ROLES, 1200),
        "coverage_minima": [
            {
                "match": {
                    "split": "extrapolation-test",
                    "query_kind": "which",
                    "contract": "unique-entailed",
                    "menu_role": "primary",
                    "decision_depth_support": True,
                },
                "minimum_bases": 400,
            }
        ],
    }
    assert MODULE.selection_balance_conflicts(policy) == []


def test_approved_decision_depth_revision_preserves_frozen_policy_and_numbers():
    policy = json.loads((DIRECTORY / "entailment-config-frozen-v1.json").read_text())
    original = deepcopy(policy)
    revised = all_scene_diagnostic_policy(policy)
    assert policy == original
    assert revised["status"] == "diagnostic-not-frozen"
    assert revised["base_targets"] == policy["base_targets"]
    assert [r.get("minimum_bases") for r in revised["coverage_minima"]] == [
        r.get("minimum_bases") for r in policy["coverage_minima"]
    ]
    assert MODULE.selection_balance_conflicts(revised) == []
    rules = [
        r
        for r in revised["coverage_minima"]
        if r["match"].get("query_kind") == "which"
        and r["match"].get("contract") == "unique-entailed"
        and r["match"].get("menu_role") == "primary"
    ]
    assert len(rules) == 4
    assert all(r["match"]["decision_depth_support"] for r in rules)
    assert {r["match"]["split"]: r["minimum_bases"] for r in rules} == {
        "train": 1400,
        "dev": 280,
        "in-range-test": 560,
        "extrapolation-test": 400,
    }
    assert not any(
        r["match"].get("query_kind") == "which"
        and r["match"].get("contract") == "unique-entailed"
        and r["match"].get("substantive_depth_support")
        for r in revised["coverage_minima"]
    )
    for before, after in zip(policy["coverage_minima"], revised["coverage_minima"]):
        if (
            before["match"].get("query_kind") != "which"
            or before["match"].get("contract") != "unique-entailed"
            or not before["match"].get("substantive_depth_support")
        ):
            assert before == after
    assert all_scene_diagnostic_policy(revised) == revised


def test_reserve_audit_cannot_credit_an_over_limit_scene():
    story = controlled_scene(
        2026111094,
        10,
        Direction.NORTH,
        1,
        story_id="bounds-test",
        source_family="fixed-family",
    )
    result = MODULE.inspect(
        (
            Entry(story, "train", 10, ("count",)),
            {
                "seed": 2026111094,
                "matches": 1,
                "direction": "North",
                "scene_limits": {"objects": 1, "premises": 32, "candidates": 8},
            },
            "count",
            False,
        )
    )
    assert result["status"] == "failed"
    assert "scene limits exceeded" in result["reason"]


def test_overlap_rejection_never_moves_or_refills_a_fixed_slot():
    story = controlled_scene(
        2026111094,
        2,
        Direction.NORTH,
        1,
        story_id="train-slot",
        source_family="family-a",
    )
    second = replace(story, story_id="dev-slot", source_family="family-b")
    entries = (
        Entry(story, "train", 2, ("count",)),
        Entry(second, "dev", 2, ("count",)),
    )
    keys, rejected, eligible = MODULE.mapping_preflight(
        entries, {"family_ids": [], "story_keys": []}
    )
    assert keys["train-slot"] == keys["dev-slot"]
    assert rejected == {
        "train-slot": "canonical-scene-cross-split-overlap",
        "dev-slot": "canonical-scene-cross-split-overlap",
    }
    assert eligible == ()
    assert [entry.split for entry in entries] == ["train", "dev"]


def test_development_overlap_is_an_explicit_rejection():
    story = controlled_scene(
        2026111094,
        2,
        Direction.NORTH,
        1,
        story_id="dev-slot",
        source_family="family",
    )
    key = MODULE.historical.story_key(story.objects, story.premise)
    keys, rejected, eligible = MODULE.mapping_preflight(
        (Entry(story, "dev", 2, ("count",)),),
        {"family_ids": [], "story_keys": [key]},
    )
    assert keys == {"dev-slot": key}
    assert rejected == {"dev-slot": "development-exclusion-overlap"}
    assert eligible == ()
