from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).with_name("make_sft_1500_data.py")
SPEC = importlib.util.spec_from_file_location("make_v13_sft_1500", MODULE)
assert SPEC and SPEC.loader
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def test_1500_matrix_has_expected_categories_and_full_semantic_grid() -> None:
    cells = DATA.build_cells()

    assert len(cells) == 129
    assert sum(cell.count for cell in cells) == 1500
    assert sum(cell.count for cell in DATA.ordinary_cells()) == 940
    loops = DATA.loop_cells()
    assert sum(cell.count for cell in loops if cell.name.endswith("-closed")) == 280
    assert sum(cell.count for cell in loops if cell.name.endswith("-open-control")) == 280
    semantic = {
        (cell.spec.relation_mode.value, cell.spec.semantic_subtype.value)
        for cell in cells
        if cell.name.startswith("sft15-semantic-")
    }
    assert len(semantic) == 36


def test_1500_generation_is_disjoint_and_holds_out_depth_five(tmp_path) -> None:
    held_out = []
    for index in range(5):
        path = tmp_path / f"held-{index}.jsonl"
        path.write_text(json.dumps({
            "messages": [{"role": "user", "content": f"held-out {index}"}]
        }) + "\n")
        held_out.append(path)

    train_path, val_path, manifest_path = DATA.generate(
        tmp_path / "sft-1500.jsonl", held_out
    )
    train, val = DATA.V1.load_jsonl(train_path), DATA.V1.load_jsonl(val_path)
    manifest = json.loads(manifest_path.read_text())

    assert len(train) == 1500
    assert len(val) == 129
    assert manifest["ordinary_consistent_rows"] == 940
    assert manifest["closed_loop_rows"] == 280
    assert manifest["open_chain_rows"] == 280
    assert max(
        max(row["difficulty"].get("x_depth") or 0, row["difficulty"].get("y_depth") or 0)
        for row in train + val
    ) < 5
    DATA.validate(train_path, val_path, manifest_path, held_out)

    stale = json.loads(manifest_path.read_text())
    stale["version"] = "old"
    manifest_path.write_text(json.dumps(stale) + "\n")
    with pytest.raises(ValueError, match="manifest version mismatch"):
        DATA.validate(train_path, val_path, manifest_path, held_out)
