from copy import deepcopy

import pytest

from spatial.v2 import composition_refinement as refinement
from spatial.v2.composition_contracts import render_prompt
from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_readable import build_outline, render_outline
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import SpatialSolverV2
from spatial.v2.tests.test_pivot_evidence import BY_ID


@pytest.fixture
def source():
    problem = BY_ID["D1"].problem()
    evidence = build_evidence(problem)
    menu = ("West", "Northeast", "South", "Northwest")
    outline = build_outline(evidence, menu, AnswerMode.SINGLE)
    outline["version"] = "readable-qualitative-evidence-v3"
    row = {
        "id": "test",
        "prompt": render_prompt(problem, menu, AnswerMode.SINGLE),
        "menu": list(menu),
        "contract": "single",
        "answer": "B",
        "semantic_case": "unique",
        "escape_reason": None,
        "depth": 2,
        "returned_count": 1,
        "substantive_depth_support": True,
        "explanation_outline": outline,
        "references": render_outline(outline),
    }
    return {
        "status": "accepted",
        "problem": tagged_dataclass_to_dict(problem),
        "evidence": tagged_dataclass_to_dict(evidence),
        "rows": [row],
        "tokens": [],
        "family_id": "preserve-family",
    }


def test_replay_preserves_inputs_and_metadata(source, monkeypatch):
    original = deepcopy(source)
    monkeypatch.setattr(refinement, "token_measurement", lambda *a: {"admitted": True})
    result = refinement.replay_record(source, None, SpatialSolverV2(backend="z3"))
    assert source == original
    assert result["family_id"] == source["family_id"]
    assert result["rows"][0]["prompt"] == source["rows"][0]["prompt"]
    assert result["rows"][0]["references"] == source["rows"][0]["references"]


@pytest.mark.parametrize("mutation", ("gold", "menu", "depth", "text", "semantics"))
def test_replay_rejects_changed_protected_data(source, monkeypatch, mutation):
    monkeypatch.setattr(refinement, "token_measurement", lambda *a: {"admitted": True})
    row = source["rows"][0]
    if mutation == "gold":
        row["answer"] = "E"
    elif mutation == "menu":
        row["menu"].reverse()
    elif mutation == "depth":
        row["depth"] = 10
    elif mutation == "text":
        row["references"]["natural"] += " altered"
    else:
        source["evidence"]["semantics"]["possible"] = ["South"]
    with pytest.raises(ValueError):
        refinement.replay_record(source, None, SpatialSolverV2(backend="z3"))


def test_replay_rejects_token_failure(source, monkeypatch):
    monkeypatch.setattr(refinement, "token_measurement", lambda *a: {"admitted": False})
    with pytest.raises(ValueError, match="token admission"):
        refinement.replay_record(source, None, SpatialSolverV2(backend="z3"))
