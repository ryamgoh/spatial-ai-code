"""Contracts for configurable V2 ablation matrices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spatial.v2.generate_matrix import app
from spatial.v2.generation import BooleanShape, MenuCoverage, QueryKind, SemanticShape
from spatial.v2.grading import AnswerMode
from spatial.v2.matrix import generate_matrix, load_experiment_matrix
from spatial.v2.solver import Direction
from spatial.v2.trace import TraceFormat

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
    - format: symbolic
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
    assert [item.trace_format for item in matrix.trace_variants] == [
        TraceFormat.NATURAL,
        TraceFormat.SYMBOLIC,
    ]
    cell = matrix.cells[0]
    assert cell.name == "direction-depth-2"
    assert cell.count == 2
    assert cell.query_kind is QueryKind.DIRECTION
    assert cell.semantic_shape is SemanticShape.UNIQUE
    assert cell.boolean_shape is BooleanShape.ATOMIC
    assert cell.depth == 2
    assert cell.target_directions == (Direction.NORTH, Direction.SOUTH)
    assert cell.answer_variants == ("single", "complete", "visible")


def test_matrix_parser_accepts_boolean_proof_shape(tmp_path) -> None:
    path = tmp_path / "boolean.yaml"
    path.write_text(
        MATRIX_YAML.replace(
            "depth: 2\n    omit_direct_query_relation: true\n    distractor_premises: 5",
            "boolean_shape: nested-case-split",
        )
    )

    matrix = load_experiment_matrix(path)

    assert matrix.cells[0].boolean_shape is BooleanShape.NESTED_CASE_SPLIT


def test_checked_in_ablation_matrix_is_valid() -> None:
    path = Path(__file__).parents[3] / "experiments/spatial-v2-ablation.example.yaml"

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

    train_path, dev_path, test_path, manifest_path = generate_matrix(
        matrix_path,
        tmp_path / "pilot.jsonl",
        replace=True,
    )

    assert dev_path.read_text() == ""
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
    assert manifest["matrix"]["cell_counts"] == {"direction-depth-2": 4}
    assert not stale.exists()
    assert manifest["base_distributions"]["matrix_cell"] == {"direction-depth-2": 4}
    assert manifest["answer_distributions"]["matrix_answer_variant"] == {
        "complete": 8,
        "single": 8,
        "visible": 8,
    }
    views = manifest["views"]
    assert views["split_fingerprint"]["train"] != views["split_fingerprint"]["test"]
    variant = views["by_variant"]["single__natural__checked-trace"]
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
        )
        for row in variant_rows
    } == {("single", "natural")}
    cell_variant = views["by_cell"]["direction-depth-2"][
        "single__natural__checked-trace"
    ]
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
    - {format: symbolic}
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

    train_path, _, _, _ = generate_matrix(matrix_path, tmp_path / "answers.jsonl")

    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert len({row["metadata"]["base_id"] for row in rows}) == 1
    by_mode = {row["metadata"]["answer_mode"]: row for row in rows}
    assert by_mode["single"]["metadata"]["menu_status"] == "undetermined"
    assert len(by_mode["all-possible"]["metadata"]["oracle_letters"]) == 2
    assert len(by_mode["visible-possible"]["metadata"]["oracle_letters"]) == 1


class FramingTokenizer:
    """Deterministic tokenizer double to test orchestration, not model lengths."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        return (
            "<chat>"
            + "".join(
                "<" + message["role"] + ">" + message["content"] for message in messages
            )
            + ("<assistant>" if add_generation_prompt else "</chat>")
        )

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(len(text) // 20 + 1))


def test_matrix_routes_controls_through_paired_context_admission(tmp_path):
    import yaml

    raw = yaml.safe_load(MATRIX_YAML)
    raw["dev_split"] = 0.25
    raw["variants"]["answers"] = raw["variants"]["answers"][:1]
    raw["variants"]["supervision_arms"] = [
        "checked-trace",
        "answer-only",
        "corrupted-trace",
    ]
    raw["cells"][0]["answer_variants"] = ["single"]
    raw["context_budget"] = {
        "tokenizer_name": "test-tokenizer",
        "train_max_tokens": 4096,
        "eval_max_tokens": 8192,
        "max_new_tokens": 4096,
    }
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(raw))
    train, dev, test, manifest = generate_matrix(
        path, tmp_path / "admitted.jsonl", tokenizer=FramingTokenizer()
    )
    payload = json.loads(manifest.read_text())
    assert payload["context_admission"]["accepted_groups"] == 4
    assert payload["context_admission"]["accepted_rows"] == 16
    rows = [
        json.loads(line)
        for split in (train, dev, test)
        for line in split.read_text().splitlines()
    ]
    assert all(
        row["metadata"]["evaluation_prompt"].startswith("<chat>") for row in rows
    )
    assert set(payload["views"]["by_variant"]) == {
        "single__natural__checked-trace",
        "single__symbolic__checked-trace",
        "single__natural__answer-only",
        "single__symbolic__corrupted-trace",
    }
    split_ids = [
        {
            json.loads(line)["metadata"]["base_id"]
            for line in split.read_text().splitlines()
        }
        for split in (train, dev, test)
    ]
    assert all(split_ids)
    assert all(
        not left & right
        for i, left in enumerate(split_ids)
        for right in split_ids[i + 1 :]
    )
    raw["context_budget"]["train_max_tokens"] = 1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(
        ValueError, match="cell direction-depth-2: paired context admission failed"
    ):
        generate_matrix(path, tmp_path / "rejected.jsonl", tokenizer=FramingTokenizer())
    assert not (tmp_path / "rejected_train.jsonl").exists()
    rejection = json.loads((tmp_path / "rejected_rejected.json").read_text())
    assert rejection["rejected_groups"] == 1
    assert rejection["rejection_reasons"]["train_context_exceeded"] == 1
    with pytest.raises(FileExistsError):
        generate_matrix(path, tmp_path / "rejected.jsonl", tokenizer=FramingTokenizer())


def test_failed_replacement_preserves_previous_validated_workload(tmp_path):
    import yaml

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(MATRIX_YAML)
    output = tmp_path / "preserved.jsonl"
    paths = generate_matrix(matrix, output)
    before = {path: path.read_bytes() for path in paths}
    raw = yaml.safe_load(MATRIX_YAML)
    raw["split_strategy"] = "holdout"
    raw["holdout_cells"] = {"dev": ["direction-depth-2"], "test": ["direction-depth-2"]}
    matrix.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="disjoint"):
        generate_matrix(matrix, output, replace=True)
    assert {path: path.read_bytes() for path in paths} == before


def test_workload_cli_path_persists_context_rejection(tmp_path, monkeypatch):
    from spatial.v2 import generate_all
    from spatial.v2.context_budget import ContextBudget

    monkeypatch.setattr(
        generate_all, "load_tokenizer", lambda _budget: FramingTokenizer()
    )
    spec = generate_all.WorkloadSpec(
        samples_per_cell=1,
        query_kinds=(QueryKind.DIRECTION,),
        semantic_shapes=(SemanticShape.UNIQUE,),
        test_split=0,
        context_budget=ContextBudget("test-tokenizer", 1, 8192, 4096),
    )
    with pytest.raises(ValueError, match="paired context admission failed"):
        generate_all.generate_workload(tmp_path / "workload.jsonl", spec)
    diagnostic = json.loads((tmp_path / "workload_rejected.json").read_text())
    assert diagnostic["rejected_groups"] == 1
    assert diagnostic["failed_rows"]
    assert not (tmp_path / "workload_train.jsonl").exists()
