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


@pytest.mark.parametrize("role", ["train", "dev"])
def test_oracle_only_generation_is_forbidden_for_supervised_roles(tmp_path, role):
    path = config(tmp_path)
    source = yaml.safe_load(path.read_text())
    source["roles"][role] = [
        {
            "stratum": "bad-oracle-only",
            "count": 1,
            "policy": {
                "semantic_shape": "unique",
                "oracle_only": True,
            },
        }
    ]
    path.write_text(yaml.safe_dump(source))

    with pytest.raises(ValueError, match="oracle-only"):
        build_release(path, tmp_path / "release", tokenizer=FixtureTokenizer())


def test_depth_benchmark_can_persist_oracle_only_count_rows(tmp_path):
    import json

    path = config(tmp_path, engineering=False)
    source = yaml.safe_load(path.read_text())
    source["attempts_per_base"] = 20
    source["roles"]["bench-depth"] = [
        {
            "stratum": "count-depth-3",
            "count": 1,
            "policy": {
                "query_kind": "count",
                "semantic_shape": "unique",
                "num_entities": 9,
                "num_premises": 12,
                "omit_direct_query_relation": True,
                "min_membership_depth": 3,
                "max_membership_depth": 3,
                "membership_candidate_limit": 3,
                "measure_support": False,
                "oracle_only": True,
            },
        }
    ]
    path.write_text(yaml.safe_dump(source))

    manifest = build_release(
        path, tmp_path / "release", tokenizer=FixtureTokenizer()
    )
    row = json.loads((tmp_path / "release/bench-depth.jsonl").read_text())

    assert manifest["artifacts"]["bench-depth"]["rows"] == 1
    assert row["metadata"]["oracle_only"] is True
    assert row["metadata"]["process_evaluation"] == "not-applicable"
    assert row["metadata"]["supervision_arm"] == "answer-only"
    assert row["messages"][2]["content"].startswith("Answer: ")
    assert "Considering only [" in row["messages"][1]["content"]


def test_failed_release_can_resume_bound_completed_strata(tmp_path, monkeypatch):
    from spatial.v2.generation import SpatialGeneratorV2

    path = config(tmp_path)
    release = tmp_path / "release"
    original = SpatialGeneratorV2.generate
    calls = 0

    def interrupted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise RuntimeError("simulated interruption")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SpatialGeneratorV2, "generate", interrupted)
    with pytest.raises(RuntimeError, match="quota failed"):
        build_release(path, release, tokenizer=FixtureTokenizer())
    assert not release.exists()
    assert list((tmp_path / "release-work").glob("*.json"))
    monkeypatch.setattr(SpatialGeneratorV2, "generate", original)
    report = build_release(path, release, tokenizer=FixtureTokenizer())
    assert report["artifacts"]["train"]["bases"] == 2
