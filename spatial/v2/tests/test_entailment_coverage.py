"""Coverage migration retains sizes and exposes structurally inapplicable gates."""

import json
from pathlib import Path

import pytest

from spatial.v2.composition_production import plan
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_coverage import derive


@pytest.fixture
def approved():
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "experiments/spatialentail-composition-v1/production-allocation-v1/execution-config-v1.json"
    )
    return json.loads(path.read_text())


def test_preserve_sizes_and_numeric_origins_without_freezing(approved):
    original = json.dumps(approved, sort_keys=True)
    result = derive(approved)
    assert sum(result["base_targets"].values()) == 10800
    assert result["base_targets"] == approved["base_targets"]
    assert result["source_mix_percent"] == approved["source_mix_percent"]
    assert result["depth_family_minima"] == approved["depth_family_minima"]
    assert result["controlled_count_weights"] == approved["controlled_count_weights"]
    assert result["frozen"] is False
    assert result["status"] == "draft-requires-coverage-review-not-executable"
    assert json.dumps(approved, sort_keys=True) == original


def test_listed_exhaustive_difference_only_required_for_selection(approved):
    result = derive(approved)
    differences = [
        rule
        for rule in result["rules"]
        if rule["match"].get("listed_exhaustive_differ")
    ]
    assert len(differences) == 3
    assert all(rule["match"]["query_kind"] == "which" for rule in differences)
    removed = [
        change
        for change in result["explicit_gate_changes"]
        if change["disposition"] == "structurally-inapplicable"
    ]
    assert len(removed) == 6
    assert {r["query_kind"] for r in removed} == {"direction", "count"}
    for category in ("contingent", "impossible"):
        rules = [
            r
            for r in result["rules"]
            if r["match"].get("has_" + category + "_negative")
        ]
        assert len(rules) == 9
        assert all(r["match"]["contract"] == Contract.LISTED.value for r in rules)


def test_draft_cannot_be_launched_by_old_production_runner(approved):
    with pytest.raises(ValueError, match="schema"):
        plan(derive(approved))
