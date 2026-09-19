from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


MODULE = Path(__file__).with_name("make_probe_v2_data.py")
SPEC = importlib.util.spec_from_file_location("make_v13_probe_v2", MODULE)
assert SPEC and SPEC.loader
V2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V2)


def test_probe_v2_matrix_changes_only_consistency_weight() -> None:
    train = V2.build_probe_v2_cells()
    val = V2.build_probe_v2_cells(validation=True)

    assert len(train) == len(val) == 55
    assert sum(cell.count for cell in train) == 400
    assert sum(cell.count for cell in val) == 110
    train_counts = Counter()
    for cell in train:
        name = cell.name
        group = (
            "closed" if name.endswith("-closed")
            else "open" if name.endswith("-open-control")
            else name.split("-", 3)[2]
        )
        train_counts[group] += cell.count
    assert train_counts["closed"] == 75
    assert train_counts["open"] == 75
    assert sum(value for key, value in train_counts.items() if key not in {"closed", "open"}) == 250


def test_generate_probe_v2_is_disjoint_from_v1_and_diagnostic(tmp_path) -> None:
    held_out = []
    for name, content in (
        ("diagnostic.jsonl", "diagnostic prompt"),
        ("v1-train.jsonl", "v1 train prompt"),
        ("v1-val.jsonl", "v1 val prompt"),
    ):
        path = tmp_path / name
        path.write_text(json.dumps({"messages": [{"role": "user", "content": content}]}) + "\n")
        held_out.append(path)

    train_path, val_path, manifest_path = V2.generate(
        tmp_path / "probe-v2.jsonl", *held_out
    )
    train, val = V2.V1.load_jsonl(train_path), V2.V1.load_jsonl(val_path)
    manifest = json.loads(manifest_path.read_text())

    assert len(train) == 400
    assert len(val) == 110
    assert manifest["probe_version"] == V2.PROBE_VERSION
    assert manifest["prompt_overlap"] == 0
    assert manifest["world_overlap"] == 0
    assert sum(row["difficulty"]["world_consistency"] == "inconsistent" for row in train) == 75
    assert sum(row["generation_cell"].endswith("-open-control") for row in train) == 75
    V2.validate(train_path, val_path, manifest_path, *held_out)

    stale = json.loads(manifest_path.read_text())
    stale["probe_version"] = "old"
    manifest_path.write_text(json.dumps(stale) + "\n")
    with pytest.raises(ValueError, match="manifest version mismatch"):
        V2.validate(train_path, val_path, manifest_path, *held_out)
