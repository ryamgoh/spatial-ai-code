"""Selected-depth audit binds certificates to the displayed reference text."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from spatial.v2.entailment_dataset import build_attempt
from spatial.v2.entailment_scene_protocol import SHORT_VERSION, all_scene_problem
from spatial.v2.tests.test_entailment_scene_protocol import scoped_problem

DIRECTORY = (
    Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
)
SPEC = importlib.util.spec_from_file_location(
    "selected_depth_audit", DIRECTORY / "audit_selected_explanation_depth.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def example():
    from spatial.v2.tests.test_entailment_dataset import tokenizer

    problem = all_scene_problem(scoped_problem("direction"))
    story = SimpleNamespace(
        story_id="selected-depth-fixture",
        source_family="selected-depth-fixture",
        recipe="fixture",
        objects=problem.objects,
        premise=problem.premise,
        candidates=tuple(o for o in problem.objects if o != "Park"),
        distractors=(),
    )
    result = build_attempt(
        story,
        problem,
        None,
        tokenizer.__wrapped__(),
        omission_percent=100,
        prompt_version=SHORT_VERSION,
    )
    assert result["status"] == "accepted"
    return result


def test_independent_depth_audit_includes_f_without_source_mutation(example):
    before = deepcopy(example)
    audited = MODULE.replay(example)
    assert example == before
    assert {row["answer"] for row in audited["rows"]} - set("ABCD") == {"F"}
    assert all(row["selected_depth_audit"] == "passed" for row in audited["rows"])
    assert all(row["selected_explanation_depth"] == 2 for row in audited["rows"])
    assert all(row["selected_explanation_depth_support"] for row in audited["rows"])


def test_changed_depth_is_rejected(example):
    value = deepcopy(example)
    value["rows"][1]["selected_explanation_depth"] = 7
    with pytest.raises(ValueError, match="depth mismatch"):
        MODULE.replay(value)


def test_correct_answer_with_altered_reference_is_rejected(example):
    value = deepcopy(example)
    value["rows"][1]["references"]["natural"] = "Unsupported assertion.\n\nAnswer: F"
    with pytest.raises(ValueError):
        MODULE.replay(value)
