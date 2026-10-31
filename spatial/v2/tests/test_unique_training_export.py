"""Incomplete or duplicate selection receipts cannot authorize training exports."""

import importlib.util
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
)
SPEC = importlib.util.spec_from_file_location(
    "unique_training_export", DIRECTORY / "export_unique_training_selection.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def receipt():
    return {
        "status": "selection-validated-not-sealed",
        "selected_menus": [
            {"base_id": f"base-{i}", "row_id": f"row-{i}"} for i in range(6000)
        ],
    }


def test_completed_inventory_requires_exactly_6000_distinct_questions():
    assert len(MODULE.selected_inventory(receipt())) == 6000


@pytest.mark.parametrize("status", ("running", "joint-selection-infeasible", None))
def test_noncompleted_selection_is_rejected(status):
    value = receipt()
    value["status"] = status
    with pytest.raises(ValueError, match="completed"):
        MODULE.selected_inventory(value)


@pytest.mark.parametrize("field", ("base_id", "row_id"))
def test_duplicate_identity_is_rejected(field):
    value = receipt()
    value["selected_menus"][1][field] = value["selected_menus"][0][field]
    with pytest.raises(ValueError, match="distinct"):
        MODULE.selected_inventory(value)


def test_missing_question_is_rejected():
    value = receipt()
    value["selected_menus"].pop()
    with pytest.raises(ValueError, match="6,000"):
        MODULE.selected_inventory(value)


@pytest.fixture(scope="module")
def audited_example():
    from spatial.v2.tests.test_selected_explanation_audit import MODULE as audit
    from spatial.v2.tests.test_selected_explanation_audit import example

    return audit.replay(example.__wrapped__())


def test_exporting_f_keeps_only_its_two_style_records(audited_example):
    from copy import deepcopy

    before = deepcopy(audited_example)
    row = next(r for r in audited_example["rows"] if r["answer"] == "F")
    chosen = MODULE.selected_record(audited_example, row["id"])
    assert audited_example == before
    assert chosen["rows"] == [row]
    assert len(chosen["tokens"]) == 2
    assert {t["id"] for t in chosen["tokens"]} == {row["id"]}
    assert {t["style"] for t in chosen["tokens"]} == {"natural", "symbolic"}


def test_unverified_selected_explanation_cannot_be_exported(audited_example):
    from copy import deepcopy

    changed = deepcopy(audited_example)
    del changed["rows"][0]["selected_depth_audit"]
    with pytest.raises(ValueError, match="proof audit"):
        MODULE.selected_record(changed, changed["rows"][0]["id"])
