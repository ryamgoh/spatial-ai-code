"""Contracts for configurable V2 ablation matrices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from generate_matrix_v2 import app
from spatial_explanation_renderers_v2 import StateMode, TraceFormat
from spatial_generation_v2 import MenuCoverage, QueryKind, SemanticShape
from spatial_grading_v2 import AnswerMode
from spatial_matrix_v2 import generate_matrix, load_experiment_matrix
from spatial_solver_v2 import Direction
from typer.testing import CliRunner

MATRIX_YAML = """
version: 1
seed: 1801
test_split: 0.25
defaults:
  num_entities: 6
  num_premises: 7
  max_attempts_per_sample: 1000
variants:
  answers:
    - name: single
      mode: single
      menu_coverage: full
    - name: complete
      mode: all-possible
      menu_coverage: full
    - name: visible
      mode: visible-possible
      menu_coverage: full
  traces:
    - format: natural
      state_mode: delta
    - format: symbolic
      state_mode: delta
cells:
  - name: direction-depth-2
    count: 2
    query_kind: direction
    semantic_shape: unique
    depth: 2
    omit_direct_query_relation: true
    distractor_premises: 5
    target_directions: [North, South]
    answer_variants: [single, complete, visible]
"""

DUPLICATE_CELLS_YAML = """
cells:
  - name: duplicate
    count: 1
    query_kind: direction
    semantic_shape: unique
  - name: duplicate
    count: 1
    query_kind: count
    semantic_shape: unique
"""


def test_matrix_parser_builds_typed_cells_and_variants(tmp_path) -> None:
    path = tmp_path / "matrix.yaml"
    path.write_text(MATRIX_YAML)

    matrix = load_experiment_matrix(path)

    assert matrix.seed == 1801
    assert matrix.test_split == 0.25
    assert [(item.mode, item.menu_coverage) for item in matrix.answer_variants] == [
        (AnswerMode.SINGLE, MenuCoverage.FULL),
        (AnswerMode.ALL_POSSIBLE, MenuCoverage.FULL),
        (AnswerMode.VISIBLE_POSSIBLE, MenuCoverage.FULL),
    ]
    assert [item.name for item in matrix.answer_variants] == [
        "single",
        "complete",
        "visible",
    ]
    assert [(item.trace_format, item.state_mode) for item in matrix.trace_variants] == [
        (TraceFormat.NATURAL, StateMode.DELTA),
        (TraceFormat.SYMBOLIC, StateMode.DELTA),
    ]
    cell = matrix.cells[0]
    assert cell.name == "direction-depth-2"
    assert cell.count == 2
    assert cell.query_kind is QueryKind.DIRECTION
    assert cell.semantic_shape is SemanticShape.UNIQUE
    assert cell.depth == 2
    assert cell.target_directions == (Direction.NORTH, Direction.SOUTH)
    assert cell.answer_variants == ("single", "complete", "visible")


def test_checked_in_ablation_matrix_is_valid() -> None:
    path = Path(__file__).parents[1] / "experiments/spatial-v2-ablation.example.yaml"

    matrix = load_experiment_matrix(path)

    assert len(matrix.cells) == 7
    assert {cell.query_kind for cell in matrix.cells} == set(QueryKind)


@pytest.mark.parametrize(
    ("suffix", "message"),
    [
        ("\nunknown: value\n", "unknown matrix keys"),
        (
            DUPLICATE_CELLS_YAML,
            "duplicate cell names",
        ),
    ],
)
def test_matrix_parser_rejects_ambiguous_configuration(
    tmp_path,
    suffix: str,
    message: str,
) -> None:
    path = tmp_path / "bad.yaml"
    source = (
        MATRIX_YAML + suffix
        if "unknown" in suffix
        else MATRIX_YAML.split("cells:")[0] + suffix.lstrip()
    )
    path.write_text(source)

    with pytest.raises(ValueError, match=message):
        load_experiment_matrix(path)


def test_matrix_parser_rejects_unknown_cell_answer_variant(tmp_path) -> None:
    path = tmp_path / "unknown-variant.yaml"
    path.write_text(
        MATRIX_YAML.replace(
            "answer_variants: [single, complete, visible]",
            "answer_variants: [missing]",
        )
    )

    with pytest.raises(ValueError, match="unknown answer variants: missing"):
        load_experiment_matrix(path)


def test_matrix_names_cannot_escape_the_output_directory(tmp_path) -> None:
    path = tmp_path / "unsafe-name.yaml"
    path.write_text(MATRIX_YAML.replace("direction-depth-2", "../escape"))

    with pytest.raises(ValueError, match="cell name must start"):
        load_experiment_matrix(path)


def test_matrix_generates_exact_paired_variant_counts(tmp_path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(MATRIX_YAML)
    stale = tmp_path / "pilot_views" / "by_variant" / "stale_train.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n")

    with pytest.raises(FileExistsError, match="use replace"):
        generate_matrix(matrix_path, tmp_path / "pilot.jsonl")
    assert stale.exists()

    train_path, test_path, manifest_path = generate_matrix(
        matrix_path,
        tmp_path / "pilot.jsonl",
        replace=True,
    )

    rows = [
        json.loads(line)
        for path in (train_path, test_path)
        for line in path.read_text().splitlines()
    ]
    # 2 base problems x 2 target directions x 3 answer modes x 2 traces.
    assert len(rows) == 24
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["metadata"]["base_id"], []).append(row)
    assert len(grouped) == 4
    assert all(len(group) == 6 for group in grouped.values())
    assert all(
        {row["metadata"]["matrix_cell"] for row in group} == {"direction-depth-2"}
        for group in grouped.values()
    )
    assert all(
        {row["metadata"]["answer_mode"] for row in group}
        == {"single", "all-possible", "visible-possible"}
        for group in grouped.values()
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["matrix"]["requested_base_problems"] == 4
    assert manifest["matrix"]["generated_rows"] == 24
    assert manifest["matrix"]["cells"] == {"direction-depth-2": 4}
    assert not stale.exists()
    assert manifest["base_distributions"]["matrix_cell"] == {"direction-depth-2": 4}
    assert manifest["answer_distributions"]["matrix_answer_variant"] == {
        "complete": 8,
        "single": 8,
        "visible": 8,
    }
    views = manifest["views"]
    assert views["split_fingerprint"]["train"] != views["split_fingerprint"]["test"]
    variant = views["by_variant"]["single__natural__delta"]
    assert variant["train_rows"] == 3
    assert variant["test_rows"] == 1
    assert (tmp_path / variant["train"]).exists()
    assert (tmp_path / variant["test"]).exists()
    variant_rows = [
        json.loads(line)
        for split in ("train", "test")
        for line in (tmp_path / variant[split]).read_text().splitlines()
    ]
    assert {
        (
            row["metadata"]["matrix_answer_variant"],
            row["metadata"]["trace_format"],
            row["metadata"]["state_mode"],
        )
        for row in variant_rows
    } == {("single", "natural", "delta")}
    cell_variant = views["by_cell"]["direction-depth-2"]["single__natural__delta"]
    assert cell_variant["train_rows"] == 3
    assert cell_variant["test_rows"] == 1
    assert (tmp_path / cell_variant["train"]).exists()
    assert (tmp_path / cell_variant["test"]).exists()


def test_matrix_cli_writes_all_artifacts(tmp_path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(MATRIX_YAML)
    result = CliRunner().invoke(
        app,
        [str(matrix_path), "--out", str(tmp_path / "cli-pilot.jsonl")],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "cli-pilot_train.jsonl").exists()
    assert (tmp_path / "cli-pilot_test.jsonl").exists()
    assert (tmp_path / "cli-pilot_manifest.json").exists()


def test_matrix_pairs_meaningful_answer_modes_on_ambiguous_problem(tmp_path) -> None:
    matrix_path = tmp_path / "answer-modes.yaml"
    matrix_path.write_text(
        """
version: 1
seed: 1745
test_split: 0
variants:
  answers:
    - {name: single, mode: single, menu_coverage: full}
    - {name: complete, mode: all-possible, menu_coverage: full}
    - {name: visible, mode: visible-possible, menu_coverage: partial}
  traces:
    - {format: symbolic, state_mode: delta}
cells:
  - name: which-ambiguity-2
    count: 1
    query_kind: which
    semantic_shape: ambiguous
    ambiguity_size: 2
    num_entities: 6
    num_premises: 5
    answer_variants: [single, complete, visible]
"""
    )

    train_path, _, _ = generate_matrix(matrix_path, tmp_path / "answers.jsonl")

    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert len({row["metadata"]["base_id"] for row in rows}) == 1
    by_mode = {row["metadata"]["answer_mode"]: row for row in rows}
    assert by_mode["single"]["metadata"]["menu_status"] == "undetermined"
    assert len(by_mode["all-possible"]["metadata"]["oracle_letters"]) == 2
    assert len(by_mode["visible-possible"]["metadata"]["oracle_letters"]) == 1
