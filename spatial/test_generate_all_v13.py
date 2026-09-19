"""Public-contract tests for the first v13 synthetic generator."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from generate_all_v13 import app, batch_generate, generate_sample
from spatial_solver_v13 import SpatialSolverV13


SOLVER = SpatialSolverV13()

V13_SUBTYPES = (
    "dir-1",
    "dir-2",
    "dir-undetermined",
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


def test_batch_can_generate_an_arbitrary_subset_of_cells(tmp_path) -> None:
    train_path, _ = batch_generate(
        str(tmp_path / "subset.jsonl"),
        relation_modes=("cardinal", "mixed"),
        subtype_counts={"dir-incomplete": 2, "which-4": 3},
        test_split=0.0,
        seed=1337,
    )

    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert len(rows) == 10
    assert {row["generation_cell"] for row in rows} == {
        "cardinal-dir-incomplete",
        "cardinal-which-4",
        "mixed-dir-incomplete",
        "mixed-which-4",
    }


def test_cli_accepts_requested_relation_and_subtype_subsets(tmp_path) -> None:
    output = tmp_path / "cli-subset.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(output),
            "--relation-modes",
            "cardinal,mixed",
            "--subtypes",
            "dir-2,count-omit",
            "--samples-per-cell",
            "2",
            "--independent-mixed-dir1",
            "0",
            "--test-split",
            "0",
            "--seed",
            "13",
        ],
    )

    assert result.exit_code == 0, result.output
    train_path = tmp_path / "cli-subset_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert len(rows) == 8
    assert {row["generation_cell"] for row in rows} == {
        "cardinal-dir-2",
        "cardinal-count-omit",
        "mixed-dir-2",
        "mixed-count-omit",
    }


def test_cli_rejects_unknown_generation_dimensions(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(tmp_path / "bad.jsonl"),
            "--relation-modes",
            "mixed,teleport",
            "--subtypes",
            "dir-1,which-99",
        ],
    )

    assert result.exit_code != 0
    assert "unknown relation modes" in result.output or "unknown semantic subtypes" in result.output


def test_cli_generates_exact_and_ranged_depth_cells(tmp_path) -> None:
    output = tmp_path / "depths.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(output),
            "--subtypes",
            "",
            "--samples-per-cell",
            "0",
            "--independent-mixed-dir1",
            "0",
            "--depth-cells",
            "cardinal:3x4:2,mixed:2-3x3-4:2",
            "--test-split",
            "0",
            "--seed",
            "13",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line)
        for line in (tmp_path / "depths_train.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert {row["generation_cell"] for row in rows} == {
        "cardinal-dir-1-depth-x3-y4",
        "mixed-dir-1-depth-x2-3-y3-4",
    }
    for row in rows:
        difficulty = row["difficulty"]
        assert difficulty["axes_independent"] is True
        if row["generation_cell"].startswith("cardinal"):
            assert (difficulty["x_depth"], difficulty["y_depth"]) == (3, 4)
        else:
            assert 2 <= difficulty["x_depth"] <= 3
            assert 3 <= difficulty["y_depth"] <= 4


def test_cli_rejects_malformed_depth_cells(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(tmp_path / "bad-depth.jsonl"),
            "--depth-cells",
            "mixed:not-a-depth",
        ],
    )

    assert result.exit_code != 0
    assert "depth cell" in result.output.lower()


def test_cli_generates_disconnected_and_query_branch_distractor_cells(tmp_path) -> None:
    output = tmp_path / "distractors.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(output),
            "--subtypes",
            "",
            "--samples-per-cell",
            "0",
            "--independent-mixed-dir1",
            "0",
            "--distractor-cells",
            "cardinal:3x4:disconnected:3:2,mixed:3x4:query-branch:3:2",
            "--test-split",
            "0",
            "--seed",
            "13",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line)
        for line in (tmp_path / "distractors_train.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    for row in rows:
        difficulty = row["difficulty"]
        assert (difficulty["x_depth"], difficulty["y_depth"]) == (3, 4)
        assert difficulty["num_distractor_relations"] == 3
        if "disconnected" in row["generation_cell"]:
            assert difficulty["num_disconnected_distractors"] == 3
        else:
            assert difficulty["num_query_branch_distractors"] == 3


def test_cli_rejects_malformed_distractor_cells(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(tmp_path / "bad-distractor.jsonl"),
            "--distractor-cells",
            "mixed:3x4:random-noise:3:2",
        ],
    )

    assert result.exit_code != 0
    assert "distractor cell" in result.output.lower()


def test_cli_generates_global_cycle_cells_and_matched_controls(tmp_path) -> None:
    output = tmp_path / "cycles.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(output),
            "--subtypes",
            "",
            "--samples-per-cell",
            "0",
            "--independent-mixed-dir1",
            "0",
            "--cycle-cells",
            (
                "dir-1:mixed:x:direct:query-connected:2:2,"
                "which-2:mixed:y:indirect:disconnected:3:2,"
                "count-1:mixed:both:indirect:query-connected:4:2"
            ),
            "--cycle-controls",
            "--test-split",
            "0",
            "--seed",
            "13",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line)
        for line in (tmp_path / "cycles_train.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 12
    invalid = [row for row in rows if row["difficulty"]["world_consistency"] == "inconsistent"]
    controls = [row for row in rows if row["difficulty"]["world_consistency"] == "consistent"]
    assert len(invalid) == len(controls) == 6
    assert {row["difficulty"]["semantic_subtype"] for row in invalid} == {
        "dir-cycle",
        "which-cycle",
        "count-cycle",
    }
    assert {row["base_semantic_subtype"] for row in rows} == {
        "dir-1",
        "which-2",
        "count-1",
    }


def test_cli_rejects_malformed_cycle_cells(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(tmp_path / "bad-cycle.jsonl"),
            "--cycle-cells",
            "which-2:mixed:z:indirect:disconnected:3:2",
        ],
    )

    assert result.exit_code != 0
    assert "cycle cell" in result.output.lower()
