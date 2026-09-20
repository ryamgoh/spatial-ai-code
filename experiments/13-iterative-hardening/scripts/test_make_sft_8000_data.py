from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

MODULE = Path(__file__).with_name("make_sft_8000_data.py")
SPEC = importlib.util.spec_from_file_location("make_v13_sft_8000", MODULE)
assert SPEC and SPEC.loader
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def test_8000_extension_has_expected_failure_targeted_recipe() -> None:
    cells = DATA.build_addition_cells()
    categories = Counter()
    for cell in cells:
        if cell.name.endswith("-closed"):
            categories["closed"] += cell.count
        elif cell.name.endswith("-open-control"):
            categories["open"] += cell.count
        elif cell.name.startswith("sft8-large-"):
            categories["large-semantic"] += cell.count
        else:
            categories["interference"] += cell.count

    assert len(cells) == 80
    assert sum(cell.count for cell in cells) == 2000
    assert categories == Counter(
        {
            "interference": 1550,
            "open": 250,
            "closed": 150,
            "large-semantic": 50,
        }
    )
    assert all(
        max(
            cell.spec.constraints.x_depth.maximum
            if cell.spec.constraints.x_depth
            else 0,
            cell.spec.constraints.y_depth.maximum
            if cell.spec.constraints.y_depth
            else 0,
            cell.spec.cycle.length if cell.spec.cycle else 0,
        )
        < 10
        for cell in cells
    )


def test_8000_generation_is_exactly_nested_and_breakpoint_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep this contract test fast by substituting a tiny coherent recipe while
    # exercising the real nesting, hashes, and leakage validation paths.
    monkeypatch.setattr(DATA, "EXPECTED_TRAIN", 2)
    monkeypatch.setattr(DATA, "EXPECTED_ADDITIONS", 1)
    monkeypatch.setattr(DATA.SFT6, "EXPECTED_TRAIN", 1)
    cell = DATA.depth_cell(
        "sft8-test-cardinal",
        DATA.V1.RelationMode.CARDINAL,
        6,
        6,
        DATA.V1.DistractorPolicy.QUERY_BRANCH,
        1,
    )
    monkeypatch.setattr(DATA, "build_addition_cells", lambda: [cell])

    base_train = tmp_path / "base_train.jsonl"
    base_val = tmp_path / "base_val.jsonl"
    base_manifest = tmp_path / "base_manifest.json"
    breakpoint = tmp_path / "breakpoint.jsonl"
    breakpoint_manifest = tmp_path / "breakpoint-manifest.json"
    base_row = {
        "messages": [
            {
                "role": "user",
                "content": "The A is to the East of the B.\n\nQuestion: base",
            }
        ],
        "generation_cell": "base",
        "difficulty": {},
    }
    base_train.write_text(json.dumps(base_row) + "\n")
    base_val.write_text("")
    base_manifest.write_text("{}\n")
    breakpoint.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "The C is to the North of the D.\n\nQuestion: held",
                    }
                ]
            }
        )
        + "\n"
    )
    breakpoint_manifest.write_text("{}\n")
    held = []
    base_1500 = (tmp_path / "a", tmp_path / "b", tmp_path / "c")
    for path in base_1500:
        path.write_text("")
    monkeypatch.setattr(DATA.SFT6, "validate", lambda *_args: None)
    monkeypatch.setattr(DATA.BREAKPOINT, "validate", lambda *_args: None)
    monkeypatch.setattr(DATA, "EXPECTED_VAL", 0)

    train, val, manifest = DATA.generate(
        tmp_path / "sft-8000.jsonl",
        base_train,
        base_val,
        base_manifest,
        base_1500,
        held,
        breakpoint,
        breakpoint_manifest,
    )

    assert train.read_bytes().startswith(base_train.read_bytes())
    assert val.read_bytes() == base_val.read_bytes()
    assert json.loads(manifest.read_text())["nested_base_rows"] == 1
