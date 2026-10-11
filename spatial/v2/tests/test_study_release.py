from pathlib import Path

import pytest
import yaml

from spatial.v2.study.build import build_release
from spatial.v2.tests.test_context_budget import FixtureTokenizer


def config(tmp_path, *, engineering=True, count=2):
    registry = Path(__file__).parents[1] / "benchmark-cells.yaml"
    source = {
        "version": 1,
        "name": "fixture",
        "engineering_only": engineering,
        "seed": 912,
        "context_budget": {
            "tokenizer_name": "fixture",
            "tokenizer_revision": "fixed",
            "train_max_tokens": 100000,
            "eval_max_tokens": 200000,
            "max_new_tokens": 100000,
        },
        "registries": [str(registry)],
        "contamination_files": [],
        "attempts_per_base": 4,
        "train_fractions": [0.5, 1],
        "roles": {
            "train": [{"cell": "DIR-ARC-UNI", "count": count}],
            "dev": [{"cell": "DIR-ARC-UNI", "count": 1}],
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(source))
    return path


def test_release_pairs_views_preserves_nesting_and_records_admission(tmp_path):
    manifest = build_release(
        config(tmp_path), tmp_path / "release", tokenizer=FixtureTokenizer()
    )
    assert manifest["artifacts"]["train"]["bases"] == 2
    assert manifest["artifacts"]["train"]["rows"] == 6
    assert manifest["artifacts"]["train-0.5"]["bases"] == 1
    assert manifest["artifacts"]["dev"]["bases"] == 1
    import json

    full = [
        json.loads(line)
        for line in (tmp_path / "release/train.jsonl").read_text().splitlines()
    ]
    small = [
        json.loads(line)
        for line in (tmp_path / "release/train-0.5.jsonl").read_text().splitlines()
    ]
    assert {r["id"] for r in small} <= {r["id"] for r in full}
    assert all(r["metadata"]["context_group_accepted"] for r in full)
    assert manifest["status"] == "engineering-only"
    assert set(manifest["artifacts"]["train"]["target_tokens_by_package"]) == {
        "natural:checked-trace",
        "symbolic:checked-trace",
        "natural:answer-only",
    }


def test_infeasible_unique_quota_fails_without_partial_release(tmp_path):
    path = config(tmp_path, engineering=False, count=100)
    source = yaml.safe_load(path.read_text())
    source["roles"]["train"] = [
        {
            "stratum": "tiny",
            "count": 100,
            "policy": {
                "semantic_shape": "unique",
                "num_entities": 2,
                "num_premises": 1,
            },
        }
    ]
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(RuntimeError, match="quota failed"):
        build_release(path, tmp_path / "release", tokenizer=FixtureTokenizer())
    assert not (tmp_path / "release").exists()


def test_heldout_skeleton_cannot_enter_training(tmp_path):
    path = config(tmp_path)
    source = yaml.safe_load(path.read_text())
    source["roles"]["train"] = [
        {
            "stratum": "bad",
            "count": 2,
            "policy": {
                "query_kind": "direction",
                "semantic_shape": "unique",
                "composition": "mp-dje-trc",
                "num_entities": 6,
                "num_premises": 10,
            },
        }
    ]
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="held-out"):
        build_release(path, tmp_path / "release", tokenizer=FixtureTokenizer())


def test_training_depth_maximum_cannot_bypass_holdout_guard(tmp_path):
    path = config(tmp_path)
    source = yaml.safe_load(path.read_text())
    source["roles"]["train"] = [
        {
            "stratum": "bad-depth",
            "count": 2,
            "policy": {
                "semantic_shape": "unique",
                "num_entities": 10,
                "num_premises": 9,
                "min_axis_depth": 1,
                "max_axis_depth": 8,
            },
        }
    ]
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="depth"):
        build_release(path, tmp_path / "release", tokenizer=FixtureTokenizer())
