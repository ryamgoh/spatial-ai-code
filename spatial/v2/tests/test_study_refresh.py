import json

import pytest

from spatial.v2.study.build import build_release
from spatial.v2.study.refresh import refresh_count_prose
from spatial.v2.tests.test_context_budget import FixtureTokenizer
from spatial.v2.tests.test_study_release import config


def test_refresh_refuses_existing_or_engineering_release(tmp_path):
    release = tmp_path / "release"
    build_release(config(tmp_path), release, tokenizer=FixtureTokenizer())
    with pytest.raises(ValueError, match="unsealed"):
        refresh_count_prose(release, tmp_path / "new")
    with pytest.raises(FileExistsError):
        refresh_count_prose(release, release)


def test_count_refresh_preserves_gold_symbolic_and_nested_rows(tmp_path, monkeypatch):
    import yaml

    from spatial.v2.study import refresh
    from spatial.v2.tests.test_context_budget import FixtureTokenizer
    from spatial.v2.tests.test_study_release import config

    path = config(tmp_path)
    source = yaml.safe_load(path.read_text())
    source["roles"] = {
        "train": [
            {
                "stratum": "cnt",
                "count": 2,
                "policy": {
                    "query_kind": "count",
                    "semantic_shape": "unique",
                    "boolean_shape": "double-negation",
                    "num_entities": 6,
                    "num_premises": 7,
                },
            }
        ],
        "dev": [
            {
                "stratum": "cnt",
                "count": 1,
                "policy": {
                    "query_kind": "count",
                    "semantic_shape": "unique",
                    "boolean_shape": "double-negation",
                    "num_entities": 6,
                    "num_premises": 7,
                },
            }
        ],
    }
    path.write_text(yaml.safe_dump(source))
    release = tmp_path / "release"
    build_release(path, release, tokenizer=FixtureTokenizer())
    mpath = release / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["status"] = "candidate-awaiting-quality-review"
    mpath.write_text(json.dumps(manifest))
    monkeypatch.setattr(refresh, "load_tokenizer", lambda budget: FixtureTokenizer())
    receipt = refresh_count_prose(release, tmp_path / "new")
    before = [json.loads(x) for x in (release / "train.jsonl").read_text().splitlines()]
    after = [
        json.loads(x) for x in (tmp_path / "new/train.jsonl").read_text().splitlines()
    ]
    for left, right in zip(before, after, strict=True):
        assert left["id"] == right["id"]
        assert left["metadata"]["oracle_letters"] == right["metadata"]["oracle_letters"]
        if (
            right["metadata"]["supervision_arm"] == "checked-trace"
            and right["metadata"]["trace_format"] == "natural"
        ):
            completion = right["messages"][-1]["content"]
            assert completion.startswith("<think>\n")
            assert completion.count("Direction contract:") == 1
        if left["metadata"]["trace_format"] == "symbolic":
            assert left["messages"] == right["messages"]
    assert receipt["quality_receipts_reused"] is False

    full = {r["id"]: r for r in after}
    for line in (tmp_path / "new/train-0.5.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert r == full[r["id"]]
