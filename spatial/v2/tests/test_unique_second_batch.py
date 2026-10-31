"""Second-batch approval does not extend or overwrite the first batch."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
)
SPEC = importlib.util.spec_from_file_location(
    "unique_second_batch", DIRECTORY / "run_unique_training_second_batch.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def test_second_caps_are_exact_and_have_no_count_queries():
    actual = {
        (b["query_kinds"][0], b["generator"]): b["count"] for b in MODULE.batches()
    }
    assert actual == {
        ("direction", "random-world"): 64,
        ("which", "random-world"): 1024,
        ("which", "random-atoms"): 512,
    }
    assert sum(actual.values()) == MODULE.CAP == 1600


def test_seeds_are_disjoint_from_both_preceding_inventories():
    frozen = json.loads((DIRECTORY / "entailment-config-frozen-v1.json").read_text())
    first = json.loads(
        (DIRECTORY / "UNIQUE-TRAINING-REPAIR-MANIFEST-v1.json").read_text()
    )
    seen = {
        seed
        for batch in [*frozen["source_batches"], *first["batches"]]
        for seed in batch.get(
            "seeds", range(batch["seed"], batch["seed"] + batch["count"])
        )
    }
    for batch in MODULE.batches():
        seeds = set(range(batch["seed"], batch["seed"] + batch["count"]))
        assert not seeds.intersection(seen)
        seen.update(seeds)


def test_existing_receipt_and_manifest_block_reruns(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="no rerun"):
        MODULE.run(tmp_path / "missing", path, tmp_path, 4)
    with pytest.raises(ValueError, match="never overwrite"):
        MODULE.prepare(path)
