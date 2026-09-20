from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

MODULE = Path(__file__).with_name("make_breakpoint_data.py")
SPEC = importlib.util.spec_from_file_location("make_v13_breakpoint", MODULE)
assert SPEC and SPEC.loader
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def test_breakpoint_matrix_has_expected_controlled_buckets() -> None:
    cells = DATA.build_cells()

    assert len(cells) == 186
    assert sum(cell.count for cell in cells) == 1224
    assert len(DATA.CHALLENGE_ENTITY_NAMES) > 30
    groups = Counter(
        "closed"
        if cell.name.endswith("-closed")
        else "open"
        if cell.name.endswith("-open-control")
        else "depth"
        for cell in cells
        for _ in range(cell.count)
    )
    assert groups == Counter({"depth": 504, "closed": 360, "open": 360})


def test_breakpoint_generation_is_unique_disjoint_and_structurally_true(
    tmp_path: Path,
) -> None:
    held_out = []
    for index in range(7):
        path = tmp_path / f"held-{index}.jsonl"
        path.write_text(
            json.dumps({"messages": [{"role": "user", "content": f"held-out {index}"}]})
            + "\n"
        )
        held_out.append(path)

    train, test, manifest = DATA.generate(tmp_path / "breakpoint.jsonl", held_out)

    assert train.read_text() == ""
    assert len(DATA.load_jsonl(test)) == 1224
    assert json.loads(manifest.read_text())["generation_cells"] == 186
    DATA.validate(test, manifest, held_out)
