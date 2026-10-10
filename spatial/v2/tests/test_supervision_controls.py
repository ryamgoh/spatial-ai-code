"""Sound process metadata and syntax-preserving evidence corruption."""

import json
from copy import deepcopy

import pytest

from spatial.v2.generation import (
    GenerationPolicy,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.solver import Direction
from spatial.v2.supervision_controls import SupervisionArm, apply_supervision_arm
from spatial.v2.symbolic_trace_codec import score_symbolic_training_trace
from spatial.v2.trace import TraceFormat


def _sample(kind=QueryKind.DIRECTION, shape=SemanticShape.UNIQUE):
    return SpatialGeneratorV2(seed=1724).generate(
        GenerationPolicy(
            query_kind=kind,
            answer_mode=AnswerMode.SINGLE,
            semantic_shape=shape,
            trace_format=TraceFormat.SYMBOLIC,
            num_entities=4,
            num_premises=4,
            query_direction=None if kind is QueryKind.DIRECTION else Direction.NORTH,
        )
    )


def test_answer_only_has_absent_process_instead_of_invalid_process():
    sample = _sample()
    row = sample.as_sft_row()
    answer = apply_supervision_arm(row, SupervisionArm.ANSWER_ONLY)
    assert answer["metadata"]["process_status"] == "absent"
    assert answer["metadata"]["expected_process_valid"] is None
    assert answer["messages"][-1]["content"] == "Answer: " + ", ".join(
        sorted(sample.menu_answer.letters)
    )
    assert row["metadata"]["expected_process_valid"] is True


@pytest.mark.parametrize(
    ("kind", "shape"),
    [
        (kind, shape)
        for kind in QueryKind
        for shape in SemanticShape
        if shape is not SemanticShape.NO_MATCH or kind is QueryKind.WHICH
    ],
)
def test_corruption_changes_evidence_but_preserves_schema_and_gold_decision(
    kind, shape
):
    sample = _sample(kind, shape)
    row = sample.as_sft_row()
    original = deepcopy(row)
    corrupt = apply_supervision_arm(
        row,
        SupervisionArm.CORRUPTED_TRACE,
        problem=sample.problem,
        expected=sample.menu_answer,
    )
    content = corrupt["messages"][-1]["content"]
    trace = content.split("<think>\n")[1].split("\n</think>")[0]
    score = score_symbolic_training_trace(sample.problem, trace, sample.menu_answer)
    assert not score.reasoning_valid
    assert score.decision_valid
    records = trace.splitlines()
    assert len(records) == 2
    assert records[1] == sample.trace.splitlines()[1]
    assert (
        json.loads(records[0])["schema"]
        == json.loads(sample.trace.splitlines()[0])["schema"]
    )
    assert (
        content.split("\n</think>\n")[1]
        == row["messages"][-1]["content"].split("\n</think>\n")[1]
    )
    assert corrupt["metadata"]["expected_process_valid"] is False
    assert corrupt["metadata"]["corruption"]["decision_preserved"]
    assert corrupt["metadata"]["corruption"]["replay_error"]
    assert row == original


def test_corruption_requires_problem_and_rejects_natural_scope():
    sample = _sample()
    with pytest.raises(ValueError, match="requires problem"):
        apply_supervision_arm(sample.as_sft_row(), SupervisionArm.CORRUPTED_TRACE)
    with pytest.raises(ValueError, match="only Symbolic"):
        apply_supervision_arm(
            sample.with_trace(TraceFormat.NATURAL).as_sft_row(),
            SupervisionArm.CORRUPTED_TRACE,
            problem=sample.problem,
            expected=sample.menu_answer,
        )


def test_removed_shuffled_arm_does_not_silently_map_to_semantic_corruption():
    with pytest.raises(ValueError):
        SupervisionArm("shuffled-trace")
