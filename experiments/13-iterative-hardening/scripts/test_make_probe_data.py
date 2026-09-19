from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


MODULE = Path(__file__).with_name("make_probe_data.py")
SPEC = importlib.util.spec_from_file_location("make_v13_probe", MODULE)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_probe_matrix_has_intended_400_row_training_mix() -> None:
    cells = PROBE.build_probe_cells(PROBE.ROWS_PER_CELL)

    assert len(cells) == 40
    assert sum(cell.count for cell in cells) == 520
    sections = Counter(cell.name.split("-", 2)[1] for cell in cells)
    assert sections == {
        "cardinal": 10,
        "mixed": 8,
        "dir": 6,
        "which": 6,
        "count": 4,
        "cycle": 6,
    }


def test_world_fingerprint_ignores_question_and_premise_order() -> None:
    first = (
        "Consider a map with multiple locations:\n\n"
        "The A is to the East of the B. The C is to the North of the D."
        "\n\nQuestion: first"
    )
    second = (
        "Consider a map with multiple locations:\n\n"
        "The C is to the North of the D. The A is to the East of the B."
        "\n\nQuestion: second"
    )

    assert PROBE.world_fingerprint(first) == PROBE.world_fingerprint(second)


def test_generate_probe_is_disjoint_and_holds_out_depth_five(tmp_path) -> None:
    diagnostic = tmp_path / "diagnostic.jsonl"
    diagnostic.write_text(json.dumps({
        "messages": [{"role": "user", "content": "held-out prompt"}]
    }) + "\n")

    train_path, val_path, manifest_path = PROBE.generate_probe(
        tmp_path / "probe.jsonl", diagnostic, seed=PROBE.SEED
    )
    train = PROBE.load_jsonl(train_path)
    val = PROBE.load_jsonl(val_path)
    manifest = json.loads(manifest_path.read_text())

    assert len(train) == 400
    assert len(val) == 120
    assert manifest["train_rows"] == 400
    assert manifest["probe_version"] == PROBE.PROBE_VERSION
    assert manifest["validation_rows"] == 120
    assert manifest["train_validation_world_overlap"] == 0
    assert manifest["train_diagnostic_world_overlap"] == 0
    assert set(manifest["train_cell_counts"].values()) == {10}
    assert set(manifest["validation_cell_counts"].values()) == {3}
    assert all(
        max(row["difficulty"].get("x_depth") or 0, row["difficulty"].get("y_depth") or 0) < 5
        for row in train + val
    )
    assert {row["difficulty_schema_version"] for row in train + val} == {4}
    PROBE.validate_probe(train_path, val_path, manifest_path, diagnostic)

    stale = json.loads(manifest_path.read_text())
    stale["probe_version"] = "old"
    manifest_path.write_text(json.dumps(stale) + "\n")
    with pytest.raises(ValueError, match="manifest version mismatch"):
        PROBE.validate_probe(train_path, val_path, manifest_path, diagnostic)
