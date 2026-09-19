"""Contract tests for the frozen v13 diagnostic matrix."""

from __future__ import annotations

from collections import Counter
import json

from generate_diagnostic_v13 import build_diagnostic_cells
from spatial_generation_v13 import generate_dataset


def test_default_diagnostic_matrix_is_balanced_and_has_expected_size() -> None:
    cells = build_diagnostic_cells()

    assert len(cells) == 233
    assert sum(cell.count for cell in cells) == 2256
    assert len({cell.name for cell in cells}) == len(cells)
    assert Counter(
        cell.name.split("-", 1)[0] for cell in cells
    ) == Counter({"semantic": 36, "depth": 29, "cycle": 168})


def test_depth_distractor_cells_have_truthful_policies_and_matched_hard_budgets(
    tmp_path,
) -> None:
    cells = [
        cell
        for cell in build_diagnostic_cells(
            semantic_per_cell=0, structural_per_cell=1, cycle_per_cell=0
        )
        if cell.count
    ]

    by_depth_mode: dict[tuple[str, str], list] = {}
    for cell in cells:
        _prefix, depth, mode, _policy = cell.name.split("-", 3)
        by_depth_mode.setdefault((depth, mode), []).append(cell)

    assert len(by_depth_mode) == 10
    for variants in by_depth_mode.values():
        hard = [cell for cell in variants if not cell.name.endswith("-none")]
        assert len(hard) == 2
        assert len({cell.spec.num_entities for cell in hard}) == 1
        assert len({cell.spec.num_relations for cell in hard}) == 1

    _, test_path = generate_dataset(
        cells, output_file=tmp_path / "diagnostic.jsonl", test_fraction=1.0, seed=1313
    )
    rows = [json.loads(line) for line in test_path.read_text().splitlines()]
    assert len(rows) == 29
    for row in rows:
        cell = row["generation_cell"]
        difficulty = row["difficulty"]
        if cell.endswith("-none"):
            assert difficulty["num_distractor_relations"] == 0
        elif cell.endswith("-disconnected"):
            assert difficulty["num_disconnected_distractors"] == 3
        else:
            assert cell.endswith("-query-branch")
            assert difficulty["num_query_branch_distractors"] == 3


def test_every_cycle_cell_has_a_budget_matched_open_chain_control() -> None:
    cells = {
        cell.name: cell
        for cell in build_diagnostic_cells()
        if cell.name.startswith("cycle-")
    }

    inconsistent = {name: cell for name, cell in cells.items() if not name.endswith("-control")}
    assert len(inconsistent) == 84
    for name, cycle in inconsistent.items():
        control = cells[name + "-control"]
        assert cycle.count == control.count
        assert cycle.spec.semantic_subtype == control.spec.semantic_subtype
        assert cycle.spec.relation_mode == control.spec.relation_mode
        assert cycle.spec.num_entities == control.spec.num_entities
        assert cycle.spec.num_relations == control.spec.num_relations


def test_diagonal_cycle_cells_only_request_both_axes() -> None:
    diagonal = [
        cell.name
        for cell in build_diagnostic_cells()
        if cell.name.startswith("cycle-") and "-diagonal-" in cell.name
    ]

    assert diagonal
    assert all("-both-" in name for name in diagonal)
