"""The authorized supplementary batch is finite and replayable."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
)
SPEC = importlib.util.spec_from_file_location(
    "unique_repair_runner", DIRECTORY / "run_unique_training_repair.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def test_manifest_batch_has_exact_approved_source_query_caps():
    from collections import Counter

    batches = MODULE.batches()
    assert sum(b["count"] for b in batches) == 2048
    assert all(b["split"] == "train" and len(b["query_kinds"]) == 1 for b in batches)
    counts = Counter()
    for batch in batches:
        counts[batch["query_kinds"][0]] += batch["count"]
    assert counts == {"direction": 1024, "count": 384, "which": 640}
    assert batches == MODULE.batches()
    assert all(b.get("depth") is None for b in batches)


def test_new_seed_ranges_do_not_overlap_each_other_or_old_inventory():
    frozen = json.loads((DIRECTORY / "entailment-config-frozen-v1.json").read_text())
    old = {
        seed
        for b in frozen["source_batches"]
        for seed in b.get("seeds", range(b["seed"], b["seed"] + b["count"]))
    }
    seen = set()
    for batch in MODULE.batches():
        seeds = set(range(batch["seed"], batch["seed"] + batch["count"]))
        assert not old.intersection(seeds)
        assert not seen.intersection(seeds)
        seen.update(seeds)
    assert len(seen) == 2048


def test_existing_receipt_blocks_rerun_even_if_incomplete(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="no rerun"):
        MODULE.run(tmp_path / "absent-manifest", receipt, tmp_path / "tokenizer", 4)


def test_existing_manifest_is_never_replaced(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="never replace"):
        MODULE.prepare(manifest)
