"""Contracts for set-valued possibility skeletons."""

from __future__ import annotations

import pytest

from spatial.v2.generation import GenerationPolicy, SpatialGeneratorV2
from spatial.v2.grading import AnswerMode


def _policy(**overrides) -> GenerationPolicy:
    return GenerationPolicy(
        answer_mode=AnswerMode.ALL_POSSIBLE,
        semantic_shape="ambiguous",
        **overrides,
    )


@pytest.mark.parametrize("query_kind", ["direction", "which", "count"])
@pytest.mark.parametrize("skeleton", ["mp", "iff", "dje", "csr"])
@pytest.mark.parametrize("width", [2, 4])
def test_boolean_skeletons_plant_exactly_width_possible_answers(
    query_kind, skeleton, width
) -> None:
    sample = SpatialGeneratorV2(seed=9200).generate(
        _policy(
            query_kind=query_kind,
            possibility_skeleton=skeleton,
            possibility_width=width,
            num_entities=9,
            num_premises=10,
        ),
        max_attempts=40,
    )

    assert sample.generation_provenance.value == "proof-template-first"
    assert sample.menu_answer.status == "possibilities"
    assert len(sample.menu_answer.letters) == width


@pytest.mark.parametrize("query_kind", ["direction", "which", "count"])
@pytest.mark.parametrize("depth", [3, 5])
def test_depth_skeletons_leave_multiple_answers_open(query_kind, depth) -> None:
    sample = SpatialGeneratorV2(seed=9100).generate(
        _policy(
            query_kind=query_kind,
            possibility_skeleton="depth",
            possibility_depth=depth,
            num_entities=2 * depth + 3,
            num_premises=2 * depth + 4,
        ),
        max_attempts=40,
    )

    assert len(sample.menu_answer.letters) >= 2


def test_skeletons_reject_unique_semantics() -> None:
    with pytest.raises(ValueError, match="ambiguous semantics"):
        GenerationPolicy(
            query_kind="direction",
            semantic_shape="unique",
            possibility_skeleton="mp",
            num_entities=7,
            num_premises=8,
        )


@pytest.mark.parametrize("width", [2, 3])
def test_joint_count_skeleton_skips_intermediate_counts(width) -> None:
    sample = SpatialGeneratorV2(seed=9200).generate(
        _policy(
            query_kind="count",
            possibility_skeleton="jcr",
            possibility_width=width,
            num_entities=8,
            num_premises=9,
        ),
        max_attempts=40,
    )

    assert sample.analysis.possible_counts == (0, width)
    assert sample.difficulty["joint_count_dependency"]
