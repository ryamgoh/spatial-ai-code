"""Public-contract tests for the first v13 synthetic generator."""

from __future__ import annotations

import re

import pytest

from generate_all_v13 import generate_sample
from spatial_solver_v13 import SpatialSolverV13


SOLVER = SpatialSolverV13()


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
