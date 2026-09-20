from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).with_name("make_sft_6000_data.py")
SPEC = importlib.util.spec_from_file_location("make_v13_sft_6000", MODULE)
assert SPEC and SPEC.loader
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def make_1500(tmp_path: Path) -> tuple[Path, Path, Path, list[Path]]:
    held_out = []
    for index in range(5):
        path = tmp_path / f"held-{index}.jsonl"
        path.write_text(
            json.dumps({"messages": [{"role": "user", "content": f"held-out {index}"}]})
            + "\n"
        )
        held_out.append(path)
    train, val, manifest = DATA.SFT15.generate(tmp_path / "sft-1500.jsonl", held_out)
    return train, val, manifest, held_out


def test_6000_extension_preserves_fourfold_cell_recipe() -> None:
    cells = DATA.additional_cells()

    assert len(cells) == 129
    assert sum(cell.count for cell in cells) == 4500
    assert sum(DATA.expected_train_cells().values()) == 6000
    assert all(
        DATA.expected_train_cells()[cell.name] == cell.count * 4
        for cell in DATA.SFT15.build_cells()
    )


def test_6000_generation_is_exactly_nested_and_disjoint(tmp_path) -> None:
    base_train, base_val, base_manifest, held_out = make_1500(tmp_path)
    train, val, manifest_path = DATA.generate(
        tmp_path / "sft-6000.jsonl",
        base_train,
        base_val,
        base_manifest,
        held_out,
    )
    rows = DATA.V1.load_jsonl(train)
    manifest = json.loads(manifest_path.read_text())

    assert len(rows) == 6000
    assert train.read_bytes().startswith(base_train.read_bytes())
    assert val.read_bytes() == base_val.read_bytes()
    assert manifest["nested_base_rows"] == 1500
    assert manifest["new_rows"] == 4500
    assert manifest["ordinary_consistent_rows"] == 3760
    assert manifest["closed_loop_rows"] == 1120
    assert manifest["open_chain_rows"] == 1120
    DATA.validate(
        train, val, manifest_path, base_train, base_val, base_manifest, held_out
    )

    tampered = DATA.V1.load_jsonl(train)
    tampered[0], tampered[1] = tampered[1], tampered[0]
    train.write_text("".join(json.dumps(row) + "\n" for row in tampered))
    with pytest.raises(ValueError, match="exact 1.5K rows"):
        DATA.validate(
            train, val, manifest_path, base_train, base_val, base_manifest, held_out
        )
