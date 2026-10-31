"""Coverage retention protects late rare cases without inventing new families."""

import pytest

from spatial.v2 import entailment_retention as retention
from spatial.v2.entailment_contracts import VERSION


def record(name, family, *, rare=False, status="accepted"):
    return {
        "interface": VERSION,
        "base_id": name,
        "family_id": family,
        "status": status,
        "stratum": "cell",
        "source_stratum": "source",
        "rows": [{"rare": rare}],
    }


@pytest.fixture
def constraints(monkeypatch):
    monkeypatch.setattr(
        retention,
        "quota_targets",
        lambda policy: {("stratum", "cell"): 2, ("source", "source"): 2},
    )
    monkeypatch.setattr(
        retention,
        "requirements",
        lambda policy: [
            {"match": {"rare": True}, "minimum_bases": 1},
            {"record_match": {"stratum": "cell"}, "minimum_families": 2},
        ],
    )


def test_selects_late_required_case_after_quota_could_have_filled(constraints):
    pool = [
        record("early-a", "F"),
        record("early-b", "F"),
        record("late", "G", rare=True),
    ]
    ids = retention.select(pool, {})
    assert ids == {"early-a", "late"}
    retention.validate(pool, [r for r in pool if r["base_id"] in ids], {})


def test_renamed_copies_do_not_satisfy_family_floor(constraints):
    pool = [record("a", "F"), record("renamed-a", "F", rare=True)]
    assert retention.select(pool, {}) is None
    with pytest.raises(ValueError, match="coverage"):
        retention.validate(pool, pool, {})


def test_gate_waits_for_required_support_then_stops(constraints):
    gate = retention.CompletionGate({})
    records = []
    for entry in (record("a", "F"), record("b", "F")):
        records.append(entry)
        assert not gate.satisfied(entry, records)
    assert gate.solver_calls == 0
    last = record("c", "G", rare=True)
    records.append(last)
    assert gate.satisfied(last, records)
    assert gate.selected_ids == {"a", "c"}


def test_verification_failure_blocks_success(constraints):
    gate = retention.CompletionGate({})
    records = []
    for entry in (
        record("a", "F"),
        record("bad", "H", status="verification-failed"),
        record("c", "G", rare=True),
    ):
        records.append(entry)
        assert not gate.satisfied(entry, records)
    assert gate.selected_ids is None
    assert gate.solver_calls == 0
    assert len(gate.diagnostic_ids()) == 2


def test_rejects_non_entailment_or_duplicate_candidates(constraints):
    entry = record("a", "F")
    with pytest.raises(ValueError, match="distinct"):
        retention.select([entry, entry], {})
    entry["interface"] = "spatialentail-composition-v1"
    with pytest.raises(ValueError, match="verified entailment"):
        retention.select([entry], {})


def test_multi_selection_is_base_cardinality_not_answer_length():
    entry = record("multi", "F")
    entry.update(query_kind="which")
    entry["rows"] = [
        {
            "menu_role": "primary",
            "semantic_case": "multiple-entailed",
            "entailed_count": 3,
            "answer": "E",
        }
    ]
    assert retention.multi_selection(entry)
    entry["rows"][0]["entailed_count"] = 5
    assert not retention.multi_selection(entry)
    entry["query_kind"] = "count"
    entry["rows"][0]["entailed_count"] = 3
    assert not retention.multi_selection(entry)


def test_exact_multi_selection_constraint_and_old_policy_compatibility(
    monkeypatch, constraints
):
    original = retention.quota_targets
    monkeypatch.setattr(
        retention,
        "quota_targets",
        lambda policy: {
            **original(policy),
            **({("selection-multi", "all-splits"): 1} if policy.get("multi") else {}),
        },
    )
    single = record("single", "F")
    multi = record("multi", "G", rare=True)
    multi.update(query_kind="which")
    multi["rows"][0].update(
        menu_role="primary",
        semantic_case="multiple-entailed",
        entailed_count=3,
    )
    assert retention.select([single, multi], {"multi": True}) == {"single", "multi"}
    assert retention.select([single, multi], {}) == {"single", "multi"}
    multi["rows"][0]["entailed_count"] = 5
    assert retention.select([single, multi], {"multi": True}) is None


def test_validator_still_rejects_unexpected_source_and_count_quota_keys(constraints):
    first = record("a", "F")
    second = record("b", "G", rare=True)
    second["controlled_count_value"] = 1
    with pytest.raises(ValueError, match="quotas"):
        retention.validate([first, second], [first, second], {})
