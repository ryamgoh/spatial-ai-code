"""Saved admission recovery rejects altered or mismatched provenance."""

import json
from types import SimpleNamespace

import pytest

from spatial.v2 import entailment_assembly as assembly
from spatial.v2.composition_execution import fingerprint
from spatial.v2.composition_training import digest
from spatial.v2.entailment_assembly import check_origin, saved_candidates


def saved_run(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    checkpoints = source / "checkpoints"
    checkpoints.mkdir()
    policy = {"interface": "spatialentail-entailment-v1"}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(policy))
    identity = {
        "policy": policy,
        "config_sha256": digest(config),
        "tokenizer_sha256": {"tokenizer.json": "pinned"},
    }
    (source / "manifest.json").write_text(
        json.dumps({"status": "interrupted", "identity": identity})
    )
    record = {"base_id": "base1", "status": "accepted", "rows": []}
    (source / "attempts.jsonl.tmp").write_text(json.dumps(record) + "\n")
    payload = {
        "identity": fingerprint(identity),
        "index": 0,
        "record": record,
    }
    checkpoint = checkpoints / "000000.json"
    checkpoint.write_text(
        json.dumps({"payload": payload, "sha256": fingerprint(payload)})
    )
    return source, config, checkpoint


def test_saved_admissions_match_checkpoint_and_origin(tmp_path):
    source, config, _ = saved_run(tmp_path)
    check_origin(source, config, {"tokenizer.json": "pinned"})
    records, origin = saved_candidates(source)
    assert [record["base_id"] for record in records] == ["base1"]
    assert len(origin["checkpoint_hashes"]) == 1


def test_saved_admissions_reject_tampered_checkpoint(tmp_path):
    source, _, checkpoint = saved_run(tmp_path)
    envelope = json.loads(checkpoint.read_text())
    envelope["payload"]["record"]["status"] = "rejected"
    checkpoint.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="integrity failure"):
        saved_candidates(source)


def test_saved_admissions_reject_altered_journal(tmp_path):
    source, _, _ = saved_run(tmp_path)
    (source / "attempts.jsonl.tmp").write_text(
        json.dumps({"base_id": "base1", "status": "accepted", "rows": ["changed"]})
        + "\n"
    )
    with pytest.raises(ValueError, match="journal differs"):
        saved_candidates(source)


def test_saved_admissions_reject_missing_checkpoint(tmp_path):
    source, _, checkpoint = saved_run(tmp_path)
    checkpoint.unlink()
    with pytest.raises(ValueError, match="without a verified checkpoint"):
        saved_candidates(source)


def test_origin_rejects_different_configuration_or_tokenizer(tmp_path):
    source, config, _ = saved_run(tmp_path)
    with pytest.raises(ValueError, match="tokenizer differs"):
        check_origin(source, config, {"tokenizer.json": "other"})
    config.write_text("{}")
    with pytest.raises(ValueError, match="configuration differs"):
        check_origin(source, config, {"tokenizer.json": "pinned"})


def test_fixed_reserve_continuation_reuses_completed_slots_in_order(
    tmp_path, monkeypatch
):
    source, config, _ = saved_run(tmp_path)
    output = tmp_path / "revision"
    output.mkdir()
    exclusion = tmp_path / "exclusions.json"
    exclusion.write_text("{}")
    policy = {
        "development_exclusions": {"path": exclusion.name, "sha256": digest(exclusion)}
    }
    jobs = [
        (index, SimpleNamespace(story=SimpleNamespace(story_id=f"story{index}")), None)
        for index in range(3)
    ]
    records = [
        {"base_id": f"base{index}", "status": "accepted", "rows": []}
        for index in range(3)
    ]
    origin = {
        "completed_task_hashes": {
            "0": fingerprint({"index": 0}),
            "1": fingerprint({"index": 1}),
        },
        "consumed_frontier": 0,
    }
    loaded, submitted, saved = [], [], []

    class Checkpoints:
        def __init__(self, *args):
            pass

        def load(self, index):
            loaded.append(index)
            return records[index]

        def save(self, index, *args):
            saved.append(index)

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def imap(self, function, tasks):
            submitted.extend(task[0] for task in tasks)
            return iter((index, records[index], 0.1, 0) for index in submitted)

    monkeypatch.setattr(assembly, "schedule", lambda policy: jobs)
    monkeypatch.setattr(
        assembly,
        "preflight",
        lambda *args: {f"story{index}": f"key{index}" for index in range(3)},
    )
    monkeypatch.setattr(assembly, "work_items", lambda *args: jobs)
    monkeypatch.setattr(assembly, "work_identity", lambda task: {"index": task[0]})
    monkeypatch.setattr(assembly, "Checkpoints", Checkpoints)
    monkeypatch.setattr(
        assembly.multiprocessing, "get_context", lambda mode: SimpleNamespace(Pool=Pool)
    )
    monkeypatch.setattr(assembly, "balanced_metadata", lambda record: record)
    monkeypatch.setattr(assembly, "retention_metadata", lambda record, *args: record)
    monkeypatch.setattr(assembly, "supply_ready", lambda *args: True)
    monkeypatch.setattr(assembly, "select", lambda *args: {"base0", "base1", "base2"})
    args = SimpleNamespace(
        source_run=source, config=config, output=output, tokenizer=tmp_path, workers=2
    )
    originals, compact = [records[0]], [records[0]]
    selected = assembly.complete_reserve(args, policy, origin, originals, compact, {})
    assert selected == {"base0", "base1", "base2"}
    assert loaded == [1]
    assert submitted == saved == [2]
    assert [record["base_id"] for record in originals] == ["base0", "base1", "base2"]
    assert origin["reserve_completion"]["new_seeds"] is False


def test_raw_admission_gets_retention_metadata_without_evidence_changes():
    record = {
        "query_kind": "count",
        "source": "controlled",
        "status": "accepted",
        "semantics": {"entailed": [0]},
        "evidence": {"saved": "unchanged"},
    }
    entry = SimpleNamespace(split="train", depth=3)
    result = assembly.retention_metadata(record, entry, "canonical-story")
    assert result["stratum"] == "train/count/3"
    assert result["source_stratum"] == "train/count/controlled"
    assert result["story_key"] == "canonical-story"
    assert result["controlled_count_value"] == 0
    assert result["evidence"] == record["evidence"]
    assert "stratum" not in record


def test_semantic_admission_gets_semantic_stratum():
    result = assembly.retention_metadata(
        {"query_kind": "which", "source": "targeted", "status": "accepted"},
        SimpleNamespace(split="dev", depth=None),
        "story-key",
    )
    assert result["stratum"] == "dev/which/semantic"
    assert "controlled_count_value" not in result
