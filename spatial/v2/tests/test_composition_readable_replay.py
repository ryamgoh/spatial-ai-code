"""Same-input replay must not change data or silently reuse old transcripts."""

from copy import deepcopy

import pytest

from spatial.v2.composition_contracts import encode, render_prompt
from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_readable_replay import assert_preserved, rerender_record
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import SpatialSolverV2
from spatial.v2.tests.test_pivot_evidence import BY_ID


@pytest.fixture
def source():
    problem = BY_ID["D2"].problem()
    evidence = build_evidence(problem)
    menu = ("West", "East", "South", "Northeast")
    rows = []
    for mode in AnswerMode:
        decision = encode(evidence.semantics, menu, mode)
        rows.append(
            {
                "id": mode.value,
                "prompt": render_prompt(problem, menu, mode),
                "menu": list(menu),
                "contract": mode.value,
                "answer": decision.letters,
                "semantic_case": decision.semantic_case,
                "escape_reason": decision.escape_reason,
                "depth": None,
                "split": "dev-pilot",
                "references": {"natural": "old natural", "symbolic": "old symbolic"},
            }
        )
    return {
        "status": "accepted",
        "problem": tagged_dataclass_to_dict(problem),
        "data": {
            "rows": rows,
            "evidence": tagged_dataclass_to_dict(evidence),
            "tokens": [],
            "measurement": {"depth": None, "reason": "underdetermined"},
        },
    }


def _tokens(monkeypatch, admitted=True):
    monkeypatch.setattr(
        "spatial.v2.composition_readable_replay.token_measurement",
        lambda tokenizer, row, style: {
            "id": row["id"],
            "style": style,
            "admitted": admitted,
        },
    )


def test_replay_preserves_original_and_adds_checked_pairs(source, monkeypatch):
    _tokens(monkeypatch)
    before = deepcopy(source)
    result = rerender_record(source, None, SpatialSolverV2(backend="z3"))
    assert source == before
    assert_preserved(source, result)
    assert result["data"]["evidence"] == source["data"]["evidence"]
    assert len(result["data"]["tokens"]) == 6
    assert result["status"] == "accepted"
    for row in result["data"]["rows"]:
        assert row["explanation_outline"]["units"]
        assert all(
            text.endswith(f"Answer: {row['answer']}")
            for text in row["references"].values()
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("answer", "F"),
        ("menu", ["North", "East", "South", "West"]),
        ("contract", "all-possible"),
        ("semantic_case", "unique"),
        ("escape_reason", "wrong"),
    ),
)
def test_replay_rejects_corrupted_saved_view(source, monkeypatch, field, value):
    _tokens(monkeypatch)
    source["data"]["rows"][0][field] = value
    with pytest.raises(ValueError):
        rerender_record(source, None, SpatialSolverV2(backend="z3"))


def test_token_failure_rejects_whole_base(source, monkeypatch):
    _tokens(monkeypatch, admitted=False)
    result = rerender_record(source, None, SpatialSolverV2(backend="z3"))
    assert result["status"] == "token-rejected"
    assert len(result["data"]["rows"]) == 3


def test_metadata_guard_rejects_changed_depth(source, monkeypatch):
    _tokens(monkeypatch)
    result = rerender_record(source, None, SpatialSolverV2(backend="z3"))
    result["data"]["rows"][0]["depth"] = 9
    with pytest.raises(ValueError, match="protected"):
        assert_preserved(source, result)
