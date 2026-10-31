"""Combined diagnostics use existing numerical quotas and a base-level target."""

import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from spatial.v2.composition_production import Entry
from spatial.v2.composition_scene_sampler import sample_scene

DIRECTORY = (
    Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
)
SPEC = importlib.util.spec_from_file_location(
    "combined_scene_diagnostic", DIRECTORY / "check_combined_all_scene.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def test_overall_900_multi_target_does_not_add_split_or_depth_quotas():
    frozen = json.loads((DIRECTORY / "entailment-config-frozen-v1.json").read_text())
    original = deepcopy(frozen)
    policy = MODULE.diagnostic_policy(frozen)
    assert frozen == original
    assert policy["diagnostic_multi_selection_bases"] == 900
    assert policy["base_targets"] == frozen["base_targets"]
    targets = MODULE.retention.quota_targets(policy)
    previous = MODULE.retention.quota_targets(frozen)
    assert targets == {
        **previous,
        ("selection-multi", "all-splits"): 900,
    }


def test_source_shortage_is_before_dedup_and_cannot_be_filled_by_controlled_data():
    frozen = json.loads((DIRECTORY / "entailment-config-frozen-v1.json").read_text())
    policy = MODULE.diagnostic_policy(frozen)
    records = [
        {"source_stratum": "train/which/random-world", "status": "accepted"},
        {"source_stratum": "train/which/random-world", "status": "unsupported"},
        *[
            {"source_stratum": "train/which/controlled", "status": "accepted"}
            for _ in range(1000)
        ],
    ]
    deficits = MODULE.source_shortfalls(records, policy)
    cell = next(r for r in deficits if r["cell"] == "train/which/random-world")
    assert cell["required"] == 300
    assert cell["verified_raw_supply"] == 1
    assert cell["shortfall"] == 299
    assert not any(r["cell"].endswith("/controlled") for r in deficits)


@pytest.mark.parametrize("target", (-1, 3601, True, 0.25))
def test_invalid_multi_base_targets_are_rejected(target):
    frozen = json.loads((DIRECTORY / "entailment-config-frozen-v1.json").read_text())
    policy = MODULE.diagnostic_policy(frozen)
    policy["diagnostic_multi_selection_bases"] = target
    with pytest.raises(ValueError, match="multi-Selection"):
        MODULE.retention.quota_targets(policy)


def test_unmentioned_scene_objects_are_rejected_without_stopping_the_run(monkeypatch):
    frozen = json.loads((DIRECTORY / "entailment-config-frozen-v1.json").read_text())
    monkeypatch.setattr(MODULE, "POLICY", MODULE.diagnostic_policy(frozen))
    story = sample_scene(3, 1)
    story = replace(story, objects=(*story.objects, "UnmentionedObject"))
    result = MODULE.admit_job((Entry(story, "train", None, ("which",)), "which"))
    assert result["status"] == "pre-admission-rejected"
    assert "every object to occur in a premise" in result["reason"]
    assert result["source_stratum"] == "train/which/targeted"
    assert not result["rows"]


def test_necessary_multi_bound_respects_substantive_not_decision_depth(monkeypatch):
    monkeypatch.setattr(MODULE.historical, "ROLES", ("train",))
    monkeypatch.setattr(MODULE.production, "common_policy", lambda p: p)
    monkeypatch.setattr(
        MODULE.historical,
        "source_quotas",
        lambda p: {"train/which/controlled": 8, "train/which/random-world": 4},
    )
    policy = {
        "diagnostic_multi_selection_bases": 4,
        "coverage_minima": [
            {
                "match": {
                    "split": "train",
                    "query_kind": "which",
                    "contract": "unique-entailed",
                    "menu_role": "primary",
                    "substantive_depth_support": True,
                },
                "minimum_bases": 8,
            }
        ],
    }
    records = [
        {
            "query_kind": "which",
            "status": "accepted",
            "source_stratum": "train/which/random-world",
            "rows": [
                {
                    "contract": "unique-entailed",
                    "menu_role": "primary",
                    "semantic_case": "multiple-entailed",
                    "entailed_count": 2,
                    "substantive_depth_support": False,
                    "decision_depth_support": True,
                }
            ],
        }
        for _ in range(2)
    ]
    result = MODULE.multi_selection_upper_bound(records, policy)
    assert result["overall_upper_bound"] == 2
    uncertain = {
        "query_kind": "which",
        "status": "verification-failed",
        "source_stratum": "train/which/random-world",
    }
    optimistic = MODULE.multi_selection_upper_bound([*records, uncertain], policy)
    assert optimistic["overall_upper_bound"] == 4
    assert optimistic["verification_failures_counted_optimistically"] == 1
    # The same numeric floor, but counting deep E decisions, removes this bound.
    match = policy["coverage_minima"][0]["match"]
    del match["substantive_depth_support"]
    match["decision_depth_support"] = True
    assert (
        MODULE.multi_selection_upper_bound(records, policy)["overall_upper_bound"] == 10
    )


def test_source_resume_requires_complete_fixed_inventory_and_verified_metadata(
    tmp_path,
):
    policy = {
        "source_batches": [
            {
                "generator": "random-world",
                "count": 1,
                "split": "train",
                "query_kinds": ["which"],
            }
        ]
    }
    path = tmp_path / "source.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="exact fixed inventory"):
        MODULE.load_verified_source_records(path, MODULE.digest(path), policy)
    record = {
        "base_id": "production-b0-s0-which",
        "split": "train",
        "query_kind": "which",
        "source_stratum": "train/which/random-world",
        "status": "accepted",
        "rows": [{}],
        "tokens": [{"admitted": True}],
    }
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="replay metadata"):
        MODULE.load_verified_source_records(path, MODULE.digest(path), policy)
    record["independent_audit"] = "passed"
    path.write_text(json.dumps(record) + "\n")
    assert MODULE.load_verified_source_records(path, MODULE.digest(path), policy) == [
        record
    ]
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.load_verified_source_records(path, "wrong-hash", policy)


@pytest.mark.parametrize(
    "change",
    (
        {},
        {"base_id": "wrong-slot"},
        {"query_key": "wrong-scene"},
        {"scope_policy": "old-scoped"},
        {"independent_audit": None},
        {"recipe": "sampled-forest-v1/axis-split"},
        {"rows": [{"prompt_version": MODULE.VERSION, "scope_policy": MODULE.SCOPE}]},
        {"tokens": [{"admitted": False}]},
    ),
)
def test_controlled_resume_only_accepts_current_verified_job_prefix(
    tmp_path, monkeypatch, change
):
    story = replace(sample_scene(3, 1), recipe=MODULE.RECIPE_VERSION + "/axis-split")
    entry = Entry(story, "train", 2, ("which",))
    monkeypatch.setattr(MODULE, "all_scene_problem", lambda p: p)
    monkeypatch.setattr(MODULE, "query_profile", lambda p: {"key": "current-scene"})
    record = {
        "base_id": f"{story.story_id}-which",
        "story_id": story.story_id,
        "family_id": story.source_family,
        "split": "train",
        "requested_depth": 2,
        "query_kind": "which",
        "stratum": MODULE.historical.stratum(entry, "which"),
        "source_stratum": "train/which/controlled",
        "query_key": "current-scene",
        "prompt_version": MODULE.VERSION,
        "scope_policy": MODULE.SCOPE,
        "status": "accepted",
        "independent_audit": "passed",
        "recipe": MODULE.RECIPE_VERSION + "/axis-split",
        "rows": [
            {
                "prompt_version": MODULE.VERSION,
                "scope_policy": MODULE.SCOPE,
                "decision_depth_support": True,
            }
        ],
        "tokens": [{"admitted": True}],
    }
    record.update(change)
    path = tmp_path / "controlled.jsonl"
    path.write_text(json.dumps(record) + "\n")
    jobs = [(entry, "which"), (entry, "count")]
    if change:
        with pytest.raises(ValueError, match="controlled"):
            MODULE.load_controlled_prefix(path, MODULE.digest(path), jobs)
    else:
        assert MODULE.load_controlled_prefix(path, MODULE.digest(path), jobs) == [record]
        with pytest.raises(ValueError, match="hash mismatch"):
            MODULE.load_controlled_prefix(path, "wrong-hash", jobs)
        with pytest.raises(ValueError, match="exceeds"):
            MODULE.load_controlled_prefix(path, MODULE.digest(path), [])


@pytest.mark.parametrize(
    "option",
    ("--source-records", "--source-sha256", "--controlled-records", "--controlled-sha256"),
)
def test_fresh_run_rejects_every_saved_result_option(monkeypatch, capsys, option):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_combined_all_scene.py",
            "--config",
            "unused.json",
            "--tokenizer",
            "unused-tokenizer",
            "--output",
            "unused-receipt.json",
            "--fresh",
            option,
            "saved-result",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        MODULE.main()
    assert exc.value.code == 2
    assert "--fresh forbids saved-result reuse" in capsys.readouterr().err


def test_supply_diagnostics_distinguish_missing_bases_and_families(monkeypatch):
    monkeypatch.setattr(
        MODULE.retention,
        "quota_targets",
        lambda policy: {("stratum", "cell"): 2, ("selection-multi", "all-splits"): 1},
    )
    monkeypatch.setattr(
        MODULE.retention,
        "requirements",
        lambda policy: [{"record_match": {"stratum": "cell"}, "minimum_families": 2}],
    )
    records = [
        {
            "stratum": "cell",
            "source_stratum": "source",
            "query_kind": "which",
            "family_id": "shared-family",
            "rows": [
                {
                    "menu_role": "primary",
                    "semantic_case": "multiple-entailed",
                    "entailed_count": 2,
                }
            ],
        }
    ]
    result = MODULE.retention_supply_shortfalls(records, {})
    assert result["quota_shortfalls"] == [
        {"kind": "stratum", "cell": "cell", "required": 2, "verified_supply": 1},
    ]
    assert result["coverage_shortfalls"][0]["verified_families"] == 1
    result = MODULE.retention_supply_shortfalls([*records, records[0]], {})
    assert not result["quota_shortfalls"]
    assert result["coverage_shortfalls"][0]["verified_families"] == 1
