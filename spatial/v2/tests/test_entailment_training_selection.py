"""One-menu selection cannot borrow evidence or duplicate questions."""

from copy import deepcopy

import pytest

from spatial.v2 import entailment_training_selection as selection
from spatial.v2.entailment_unique_study import project_record
from spatial.v2.tests.test_entailment_unique_study import record


def candidate(identity):
    result = project_record(record())
    result.update(
        base_id=identity,
        query_key=identity,
        split="train",
        query_kind="direction",
        stratum="train/direction/1",
        source_stratum="train/direction/controlled",
        family_id=identity,
        independent_audit="passed",
    )
    for row in result["rows"]:
        old_id = row["id"]
        row.update(
            id=f"{identity}-{row['menu_role']}",
            base_id=identity,
            split="train",
            query_kind="direction",
            escape_reason=None if row["answer"] == "A" else "unique-unlisted",
            selected_depth_audit="passed",
        )
        for token in result["tokens"]:
            if token["id"] == old_id:
                token["id"] = row["id"]
    return result


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(selection.retention, "requirements", lambda p: [])
    monkeypatch.setattr(
        selection.retention,
        "quota_targets",
        lambda p: {
            ("stratum", "train/direction/1"): 2,
            ("source", "train/direction/controlled"): 2,
        },
    )
    return {"training_allocation": {"direction": {"A-D": 1, "unique-unlisted": 1}}}


def test_selection_uses_distinct_bases_and_keeps_originals(policy):
    candidates = [candidate("one"), candidate("two")]
    before = deepcopy(candidates)
    selected = selection.select(candidates, [], policy)
    assert len(selected) == 2
    assert len({r["base_id"] for r in selected}) == 2
    assert {r["rows"][0]["answer"] for r in selected} == {"A", "F"}
    assert candidates == before
    selection.validate(candidates, selected, [], policy)


def test_two_menus_on_one_base_cannot_fill_two_slots(policy):
    assert selection.select([candidate("one")], [], policy) is None


def test_duplicate_canonical_question_cannot_fill_two_slots(policy):
    candidates = [candidate("one"), candidate("two")]
    candidates[1]["query_key"] = "one"
    assert selection.select(candidates, [], policy) is None


def test_unreplayed_menu_is_not_selectable(policy):
    base = candidate("one")
    del base["rows"][1]["selected_depth_audit"]
    with pytest.raises(ValueError, match="replayed"):
        selection.select([base], [], policy)
