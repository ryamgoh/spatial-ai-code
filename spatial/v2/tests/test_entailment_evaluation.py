"""Versioned scoring and explicit Listed entailment/non-entailment diagnostics."""

import pytest

from spatial.v2.composition_evaluation import allowed_answers, score
from spatial.v2.entailment_contracts import VERSION, Contract


def rows():
    return [
        {
            "id": contract.value,
            "interface": VERSION,
            "contract": contract.value,
            "answer": "A",
            "semantic_case": "unique-entailed",
            "entailed_menu_letters": "A",
            "negative_menu_categories": {
                "B": "contingent",
                "C": "impossible",
                "D": "impossible",
            },
        }
        for contract in Contract
    ]


def outputs(listed_answer="A"):
    return [
        {
            "id": contract.value,
            "stage1": {
                "text": "Answer: "
                + (listed_answer if contract is Contract.LISTED else "A"),
                "token_ids": [1],
                "input_tokens": 10,
            },
            "stage2": {"text": "A", "token_ids": [1], "input_tokens": 20},
        }
        for contract in Contract
    ]


def test_entailment_contracts_have_distinct_versioned_scores():
    result = score(rows(), outputs())
    assert set(result["stage1"]["contracts"]) == {c.value for c in Contract}
    assert result["stage1"]["correct"] == 3
    assert result["stage2"]["correct"] == 3
    assert result["stage1"]["listed_option_classification"] == {
        "entailed": {"correct": 1, "total": 1},
        "contingent": {"correct": 1, "total": 1},
        "impossible": {"correct": 2, "total": 2},
    }


def test_selecting_contingent_option_is_a_false_positive():
    result = score(rows(), outputs("AB"))
    assert (
        result["stage1"]["listed_option_classification"]["contingent"]["correct"] == 0
    )
    assert result["stage1"]["listed_option_classification"]["entailed"]["correct"] == 1
    assert result["rescues"] == 1


@pytest.mark.parametrize("answer", ("invalid", "E"))
def test_invalid_or_inconsistent_response_gets_no_negative_credit(answer):
    result = score(rows(), outputs(answer))
    assert all(
        cell["correct"] == 0
        for cell in result["stage1"]["listed_option_classification"].values()
    )


def test_mixed_versions_are_rejected():
    data = rows()
    data[0]["contract"] = "single"
    with pytest.raises(ValueError, match="one semantic version"):
        score(data, outputs())
    data = rows()
    del data[0]["interface"]
    with pytest.raises(ValueError, match="versioned interface"):
        score(data, outputs())


def test_plain_scoring_requires_matched_known_protocol_and_scope():
    from spatial.v2.entailment_scene_protocol import SCOPE
    from spatial.v2.entailment_scene_protocol import VERSION as SCENE_VERSION

    data = rows()
    for row in data:
        row.update(prompt_version=SCENE_VERSION, scope_policy=SCOPE)
    assert score(data, outputs())["stage1"]["correct"] == 3
    data[0]["scope_policy"] = "unknown"
    with pytest.raises(ValueError, match="cannot mix"):
        score(data, outputs())
    for row in data:
        row["scope_policy"] = "unknown"
    with pytest.raises(ValueError, match="unknown evaluation"):
        score(data, outputs())


@pytest.mark.parametrize("contract", list(Contract))
def test_structured_answer_choices(contract):
    assert len(allowed_answers(contract.value)) == (
        6 if contract is Contract.UNIQUE else 17
    )
