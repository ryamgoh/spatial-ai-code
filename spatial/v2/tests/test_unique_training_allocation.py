"""Necessary bounds must not count two menus as two training questions."""

import importlib.util
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/spatialentail-composition-v1/check_unique_training_allocation.py"
)
SPEC = importlib.util.spec_from_file_location("unique_allocation_check", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def base(*rows):
    return {
        "source_stratum": "train/which/controlled",
        "stratum": "train/which/1",
        "rows": list(rows),
    }


PRIMARY = {
    "answer": "A",
    "escape_reason": None,
    "menu_role": "primary",
    "substantive_depth_support": True,
}
OMISSION = {
    "answer": "F",
    "escape_reason": "unique-unlisted",
    "menu_role": "omission",
    "substantive_depth_support": False,
}
QUOTAS = {
    ("source", "train/which/controlled"): 10,
    ("stratum", "train/which/1"): 10,
}


def test_ad_and_f_union_counts_one_base_once():
    result = MODULE.supply_bound(
        [base(PRIMARY, OMISSION)], {"A-D", "unique-unlisted"}, QUOTAS
    )
    assert result["raw"] == result["upper_bound"] == 1


def test_omission_cannot_borrow_primary_coverage():
    record = base(PRIMARY, OMISSION)
    assert MODULE.eligible(record, {"unique-unlisted"})
    assert not MODULE.eligible(
        record, {"unique-unlisted"}, {"substantive_depth_support": True}
    )
    assert not MODULE.eligible(record, {"unique-unlisted"}, {"menu_role": "primary"})


def test_source_quota_caps_surplus_and_unknown_sources():
    records = [base(PRIMARY) for _ in range(3)]
    quotas = {**QUOTAS, ("source", "train/which/controlled"): 2}
    assert MODULE.supply_bound(records, {"A-D"}, quotas)["upper_bound"] == 2
    assert MODULE.supply_bound(records, {"A-D"}, {})["upper_bound"] == 0


@pytest.mark.parametrize(
    "reason", ("multiple-entailed", "no-entailed-answer", "inconsistent")
)
def test_e_subtypes_are_separate(reason):
    assert MODULE.category({"answer": "E", "escape_reason": reason}) == reason


def test_invalid_f_reason_is_rejected():
    with pytest.raises(ValueError, match="unexpected"):
        MODULE.category({"answer": "F", "escape_reason": "multiple-entailed"})


def test_targets_match_approved_counts():
    for kind, targets in MODULE.TARGETS.items():
        assert sum(targets.values()) == 2000
        assert targets["A-D"] == 1200
        assert targets["unique-unlisted"] == 200
        assert targets["no-entailed-answer"] == (225 if kind == "which" else 450)
        assert targets["inconsistent"] == (75 if kind == "which" else 150)
        assert targets.get("multiple-entailed", 0) == (300 if kind == "which" else 0)


def test_repair_policy_only_replaces_three_training_depth_predicates():
    import json

    from spatial.v2.entailment_unique_study import (
        study_policy,
        training_allocation_policy,
    )

    frozen = json.loads((PATH.parent / "entailment-config-frozen-v1.json").read_text())
    old, new = study_policy(frozen), training_allocation_policy(frozen)
    differences = [
        (a, b) for a, b in zip(old["coverage_minima"], new["coverage_minima"]) if a != b
    ]
    assert len(differences) == 3
    for before, after in differences:
        assert before["minimum_bases"] == after["minimum_bases"] == 1400
        assert after["match"]["split"] == "train"
        assert after["match"]["selected_explanation_depth_support"] is True
        assert "menu_role" not in after["match"]
    for field in (
        "source_batches",
        "depth_weights",
        "depth_family_minima",
        "controlled_count_weights",
        "source_mix_percent",
    ):
        assert old[field] == new[field]
    assert new["training_allocation"] == MODULE.TARGETS


@pytest.mark.parametrize("kind", ("direction", "count"))
def test_selected_f_proof_has_same_depth_as_displayed_answer(kind):
    from spatial.v2.entailment_contracts import Contract
    from spatial.v2.entailment_dataset import menus_for, selected_explanation_depth
    from spatial.v2.entailment_evidence import build_evidence
    from spatial.v2.entailment_scene_protocol import all_scene_problem
    from spatial.v2.tests.test_entailment_scene_protocol import scoped_problem

    evidence = build_evidence(all_scene_problem(scoped_problem(kind)))
    primary, omission = menus_for(evidence.semantics, 0, omission_percent=100)
    assert selected_explanation_depth(evidence, primary, Contract.UNIQUE) == 2
    assert selected_explanation_depth(evidence, omission, Contract.UNIQUE) == 2


def test_multi_e_does_not_credit_a_noncheapest_emitted_pair():
    from copy import deepcopy

    from spatial.v2.entailment_contracts import Contract
    from spatial.v2.entailment_dataset import selected_explanation_depth
    from spatial.v2.entailment_evidence import build_evidence
    from spatial.v2.entailment_scene_protocol import all_scene_problem
    from spatial.v2.tests.test_entailment_scene_protocol import scoped_problem

    evidence = build_evidence(all_scene_problem(scoped_problem("which")))
    # Actual first two domain entailments include the direct Clinic relation.
    assert (
        selected_explanation_depth(
            evidence, ("Library", "Museum", "Bank", "Cafe"), Contract.UNIQUE
        )
        == 2
    )
    # Isolate certificate-selection logic with a synthetic longer first member;
    # the remaining two members still form a cheaper depth-two certificate.
    altered = deepcopy(evidence)
    altered.positive_support["Clinic"]["paths"][0]["indices"] = [0, 1, 2]
    assert (
        selected_explanation_depth(
            altered, ("Library", "Museum", "Bank", "Cafe"), Contract.UNIQUE
        )
        is None
    )


def test_which_f_has_positive_proof_depth():
    from spatial.v2.entailment_contracts import Contract
    from spatial.v2.entailment_dataset import selected_explanation_depth
    from spatial.v2.entailment_evidence import build_evidence
    from spatial.v2.tests.test_pivot_evidence import BY_ID

    evidence = build_evidence(BY_ID["S6"].problem())
    assert (
        selected_explanation_depth(evidence, ("U", "V", "W", "T"), Contract.UNIQUE)
        is not None
    )


@pytest.mark.parametrize("name", ("D2", "I1"))
def test_ambiguous_and_inconsistent_e_have_no_positive_depth(name):
    from spatial.v2.entailment_contracts import Contract
    from spatial.v2.entailment_dataset import selected_explanation_depth
    from spatial.v2.entailment_evidence import build_evidence
    from spatial.v2.tests.test_pivot_evidence import BY_ID

    evidence = build_evidence(BY_ID[name].problem())
    assert (
        selected_explanation_depth(
            evidence, ("North", "East", "South", "West"), Contract.UNIQUE
        )
        is None
    )
