"""Public-contract tests for the first v13 synthetic generator."""

from __future__ import annotations

import re

import pytest

from generate_all_v13 import batch_generate, generate_sample
from spatial_solver_v13 import SpatialSolverV13


SOLVER = SpatialSolverV13()

V13_SUBTYPES = (
    "dir-1",
    "dir-2",
    "dir-undetermined",
    "dir-cycle",
    "dir-incomplete",
    "dir-omit",
    "which-1",
    "which-2",
    "which-3",
    "which-4",
    "which-0",
    "count-1",
    "count-omit",
)


def user_text(sample: dict) -> str:
    return next(message["content"] for message in sample["messages"] if message["role"] == "user")


def answer_text(sample: dict) -> str:
    assistant = next(message["content"] for message in sample["messages"] if message["role"] == "assistant")
    matches = re.findall(r"Answer:\s*([A-E](?:\s*,\s*[A-E])*)", assistant)
    assert matches
    return ",".join(part.strip() for part in matches[-1].split(","))


@pytest.mark.parametrize("relation_mode,expected_mix", [
    ("diagonal", "diagonal-only"),
    ("cardinal", "cardinal-only"),
    ("mixed", "mixed"),
])
def test_generated_type0_round_trips_through_v13_solver(
    relation_mode: str, expected_mix: str
) -> None:
    sample = generate_sample(
        seed=1301,
        relation_mode=relation_mode,
        question_type=0,
        target_num_answers=1,
        num_entities=8,
        num_sentences=10,
    )

    assert sample is not None
    text = user_text(sample)
    assert sample["oracle_option"] == SOLVER.solve(text) == answer_text(sample)
    assert sample["difficulty"] == SOLVER.analyze(text)
    assert sample["difficulty"]["relation_mix"] == expected_mix


@pytest.mark.parametrize("question_type,target_num_answers", [(1, 1), (2, None)])
def test_generated_non_direction_questions_round_trip(
    question_type: int, target_num_answers: int | None
) -> None:
    sample = generate_sample(
        seed=1307 + question_type,
        relation_mode="mixed",
        question_type=question_type,
        target_num_answers=target_num_answers,
        num_entities=8,
        num_sentences=10,
    )

    assert sample is not None
    text = user_text(sample)
    assert sample["oracle_option"] == SOLVER.solve(text) == answer_text(sample)
    assert sample["difficulty"]["question_type"] == question_type


@pytest.mark.parametrize("subtype", V13_SUBTYPES)
def test_every_semantic_subtype_round_trips_through_the_solver(subtype: str) -> None:
    sample = generate_sample(
        seed=1313,
        relation_mode="mixed",
        subtype=subtype,
        num_entities=10,
        num_sentences=15,
    )

    assert sample is not None, subtype
    text = user_text(sample)
    assert sample["oracle_option"] == SOLVER.solve(text) == answer_text(sample)
    assert sample["difficulty"] == SOLVER.analyze(text)
    assert sample["difficulty"]["semantic_subtype"] == subtype


def test_mixed_dir1_can_require_independent_axis_evidence() -> None:
    sample = generate_sample(
        seed=1321,
        relation_mode="mixed",
        subtype="dir-1",
        require_independent_axes=True,
        num_entities=10,
        num_sentences=15,
    )

    assert sample is not None
    difficulty = sample["difficulty"]
    assert difficulty["relation_mix"] == "mixed"
    assert difficulty["semantic_subtype"] == "dir-1"
    assert difficulty["x_depth"] is not None
    assert difficulty["y_depth"] is not None
    assert difficulty["axes_independent"] is True
    assert difficulty["shared_supporting_statements"] == []


def test_balanced_batch_crosses_modes_and_semantic_subtypes(tmp_path) -> None:
    train_path, test_path = batch_generate(
        str(tmp_path / "balanced.jsonl"),
        relation_modes=("diagonal", "cardinal", "mixed"),
        subtype_counts={subtype: 1 for subtype in V13_SUBTYPES},
        independent_mixed_dir1=1,
        test_split=0.0,
        seed=1331,
    )

    rows = [
        __import__("json").loads(line)
        for line in train_path.read_text().splitlines()
        if line.strip()
    ]
    assert not test_path.read_text()
    assert len(rows) == 3 * len(V13_SUBTYPES) + 1
    cells = {
        (row["difficulty"]["relation_mix"], row["difficulty"]["semantic_subtype"])
        for row in rows
    }
    assert len(cells) == 3 * len(V13_SUBTYPES)
    hard = [row for row in rows if row.get("generation_cell") == "mixed-dir-1-independent"]
    assert len(hard) == 1
    assert hard[0]["difficulty"]["axes_independent"] is True
