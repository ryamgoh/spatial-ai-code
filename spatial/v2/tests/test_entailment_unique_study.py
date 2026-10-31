"""Unique-only projection preserves semantics and excludes retired contracts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from spatial.v2.composition_evaluation import score, select_epoch
from spatial.v2.composition_training import MODEL, REVISION, messages
from spatial.v2.entailment_contracts import VERSION, Contract
from spatial.v2.entailment_scene_protocol import SCOPE
from spatial.v2.entailment_scene_protocol import VERSION as PROMPT_VERSION
from spatial.v2.entailment_unique_study import (
    PAIR_FLAGS,
    STUDY_SCOPE,
    UNIQUE,
    project_record,
    study_policy,
)


def record():
    rows = [
        {
            "id": f"base-{menu}-{contract.value}",
            "base_id": "base",
            "bundle_id": f"base-{menu}",
            "interface": VERSION,
            "contract": contract.value,
            "prompt_version": PROMPT_VERSION,
            "scope_policy": SCOPE,
            "menu_role": menu,
            "answer": "A" if menu == "primary" else "F",
            "prompt": "unchanged premise paragraph",
            "references": {
                "natural": "unchanged natural",
                "symbolic": "unchanged symbolic",
            },
        }
        for menu in ("primary", "omission")
        for contract in Contract
    ]
    return {
        "base_id": "base",
        "status": "accepted",
        "rows": rows,
        "tokens": [
            {"id": row["id"], "style": style, "admitted": True}
            for row in rows
            for style in ("natural", "symbolic")
        ],
    }


def outputs(rows, correct=True):
    return [
        {
            "id": r["id"],
            "stage1": {
                "text": f"Answer: {r['answer'] if correct else 'B'}",
                "token_ids": [1],
                "input_tokens": 2,
            },
            "stage2": {"text": r["answer"], "token_ids": [1], "input_tokens": 3},
        }
        for r in rows
    ]


def test_projection_keeps_unique_content_and_both_styles_without_mutating_source():
    original = record()
    snapshot = deepcopy(original)
    result = project_record(original)
    assert original == snapshot
    assert len(result["rows"]) == 2
    assert len(result["tokens"]) == 4
    for row in result["rows"]:
        assert row["study_scope"] == STUDY_SCOPE
        expected = next(r for r in original["rows"] if r["id"] == row["id"])
        assert {k: v for k, v in row.items() if k != "study_scope"} == expected
    assert project_record(result) == result


@pytest.mark.parametrize(
    "fault", ("missing-style", "bad-gold", "scoped", "missing-primary", "duplicate")
)
def test_invalid_projections_fail_closed(fault):
    data = record()
    if fault == "missing-style":
        data["tokens"] = data["tokens"][1:]
    elif fault == "bad-gold":
        data["rows"][0]["answer"] = "AB"
    elif fault == "scoped":
        data["rows"][0]["scope_policy"] = None
    elif fault == "missing-primary":
        data["rows"] = [r for r in data["rows"] if r["menu_role"] != "primary"]
    else:
        data["rows"].append(deepcopy(data["rows"][-3]))
    with pytest.raises(ValueError):
        project_record(data)


def test_unique_score_is_not_divided_by_three_and_has_no_listed_metric():
    rows = project_record(record())["rows"]
    result = score(rows, outputs(rows))
    assert result["stage1"]["macro_accuracy"] == 1
    assert set(result["stage1"]["contracts"]) == {UNIQUE}
    assert "listed_option_classification" not in result["stage1"]
    assert score(rows, outputs(rows, False))["stage1"]["macro_accuracy"] == 0
    untagged = [{k: v for k, v in r.items() if k != "study_scope"} for r in rows]
    with pytest.raises(ValueError, match="all three"):
        score(untagged, outputs(untagged))
    mixed = deepcopy(rows)
    del mixed[0]["study_scope"]
    with pytest.raises(ValueError, match="study scopes"):
        score(mixed, outputs(mixed))


def test_unique_checkpoint_selection_uses_stage_one_and_earliest_tie():
    rows = project_record(record())["rows"]
    runs = [
        {
            "epoch": epoch,
            "split": "dev",
            "style": "natural",
            "manifest_sha256": "manifest",
            "model": MODEL,
            "revision": REVISION,
            "adapter": f"adapter-{epoch}",
            "adapter_sha256": f"hash-{epoch}",
            "outputs": outputs(rows, correct=epoch == 2),
        }
        for epoch in (1, 2)
    ]
    assert select_epoch(rows, runs, "natural", "manifest")["epoch"] == 2
    runs[0]["outputs"] = outputs(rows)
    selected = select_epoch(rows, runs, "natural", "manifest")
    assert selected["epoch"] == 1
    assert selected["rule"].startswith("Unique Stage-1")


def test_training_messages_reject_retired_contract_in_unique_study():
    row = project_record(record())["rows"][0]
    assert messages(row, "natural")[1]["content"] == row["prompt"]
    row["contract"] = Contract.LISTED.value
    with pytest.raises(ValueError, match="Unique-only"):
        messages(row, "natural")


def test_unique_policy_retires_only_contract_specific_requirements():
    root = Path(__file__).resolve().parents[3]
    frozen = json.loads(
        (
            root
            / "experiments/spatialentail-composition-v1/entailment-config-frozen-v1.json"
        ).read_text()
    )
    snapshot = deepcopy(frozen)
    policy = study_policy(frozen)
    assert frozen == snapshot
    assert policy["active_contracts"] == [UNIQUE]
    assert policy["base_targets"] == frozen["base_targets"]
    assert policy["source_batches"] == frozen["source_batches"]
    assert policy["depth_family_minima"] == frozen["depth_family_minima"]
    assert "diagnostic_multi_selection_bases" not in policy
    assert "approval" not in policy
    for rule in policy["coverage_minima"]:
        assert not PAIR_FLAGS.intersection(rule["match"])
        assert rule["match"].get("contract") in (None, UNIQUE)
    negative = [
        r
        for r in policy["coverage_minima"]
        if any(
            r["match"].get(f"has_{category}_negative")
            for category in ("contingent", "impossible")
        )
    ]
    assert len(negative) == 18
    assert all(r["match"]["contract"] == UNIQUE for r in negative)
