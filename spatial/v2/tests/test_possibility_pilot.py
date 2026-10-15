"""Structural contract for the ALL-POSSIBLE calibration pilot writer."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from spatial.v2.study import possibility_pilot as pilot


class _StubTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        prefix = "".join(f"<{m['role']}>{m['content']}\n" for m in messages[:-1])
        if add_generation_prompt:
            whole = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
            return whole + "<assistant>\n<think>\n"
        content = messages[-1]["content"]
        if content.startswith("<think>\n"):
            return prefix + "<assistant>\n" + content
        return prefix + "<assistant>\n<think>\n\n</think>\n" + content

    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text) // 4)))


def test_generation_preserves_natural_source_and_derives_evaluation_row(monkeypatch):
    checked_row = {
        "id": "long-checked-row",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<think>\n" + "evidence " * 100_000 + "\n</think>\nAnswer: A, B",
            },
        ],
        "metadata": {"trace_format": "natural"},
    }

    class _Sample:
        def as_sft_row(self, *, include_audit=False):
            if include_audit:
                checked_row["audit"] = {"answer_certificate": {"checked": True}}
            return checked_row

    class _Generator:
        def __init__(self, seed):
            pass

        def generate(self, policy, max_attempts):
            return _Sample()

    class _Cell:
        @staticmethod
        def generation_policy():
            return object()

    monkeypatch.setattr(pilot, "SpatialGeneratorV2", _Generator)

    rows = pilot._generate_cell_rows(_Cell(), samples_per_cell=1, seed=42)

    assert rows[0]["messages"][-1]["content"].startswith("<think>\n")
    assert "evidence " * 10 in rows[0]["messages"][-1]["content"]
    evaluation_row = pilot._evaluation_row(rows[0])
    assert evaluation_row["messages"][-1]["content"] == "Answer: A, B"
    assert evaluation_row["metadata"]["supervision_arm"] == "answer-only"


def test_worker_count_uses_slurm_allocation_and_caps_at_cell_count(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "256")
    assert pilot._resolve_worker_count(None, cell_count=34) == 34
    assert pilot._resolve_worker_count(8, cell_count=34) == 8


def test_worker_count_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    with pytest.raises(ValueError, match="SLURM_CPUS_PER_TASK"):
        pilot._resolve_worker_count(None, cell_count=34)
    with pytest.raises(ValueError, match="workers must be positive"):
        pilot._resolve_worker_count(0, cell_count=34)


def test_generation_cache_is_hash_checked_and_reusable(tmp_path):
    cache = tmp_path / "cache"
    contract = {
        "schema": "spatialentail-all-possible-generation-cache-v1",
        "registry_fingerprint": "registry",
        "samples_per_cell": 1,
        "seed": 42,
        "environment_sha256": {"generator": "hash"},
    }
    generated = {"CELL": [{"id": "row", "messages": [], "metadata": {}}]}

    pilot._write_generation_cache(cache, generated, contract)

    assert pilot._load_generation_cache(cache, contract, ["CELL"]) == generated
    changed = {**contract, "seed": 43}
    with pytest.raises(ValueError, match="cache contract changed"):
        pilot._load_generation_cache(cache, changed, ["CELL"])


def test_natural_length_statistics_report_quantiles_and_thresholds():
    stats = pilot._length_statistics(
        {
            "CELL": [
                {
                    "training_model_tokens": value,
                    "evaluation_prompt_model_tokens": value // 2,
                    "natural_target_model_tokens": value // 2,
                }
                for value in (10_000, 20_000, 30_000)
            ]
        }
    )

    training = stats["overall"]["training_model_tokens"]
    assert training["p50"] == 20_000
    assert training["p95"] == 30_000
    assert training["max"] == 30_000
    assert training["over_threshold"]["20000"] == 1


def test_pilot_bundle_is_set_valued_and_hash_consistent(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "_tokenizer", lambda: _StubTokenizer())
    repo_root = Path(__file__).resolve().parents[3]
    out = repo_root / "experiments/spatialentail-allpossible-pilot/_pytest"
    cache = pilot._generation_cache_path(out)
    if out.exists():
        pytest.skip("stale pilot output present")
    try:
        receipt = pilot.prepare_pilot(
            repo_root / "spatial/v2/all-possible-cells.yaml",
            out,
            samples_per_cell=1,
            seed=42,
            workers=2,
        )
        assert len(receipt["tasks"]) == 34
        assert [job["name"] for job in receipt["jobs"]] == [
            "untuned",
            "natural-seed-42",
        ]
        assert receipt["allow_multiple_answers"] is True
        assert receipt["choices"] == list("ABCDE")
        assert receipt["generation_workers"] == 2
        for name, digest in receipt["file_sha256"].items():
            assert (
                hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
            )
        config = (out / "configs/untuned.yaml").read_text()
        assert "allow_multiple_answers: true" in config
        assert "- E" in config
        for script in (out / "slurm").glob("*.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True)
        row = json.loads(
            (out / "data/DIR-MP-ALT-G2.jsonl").read_text().splitlines()[0]
        )
        assert row["metadata"]["answer_mode"] == "all-possible"
        assert row["metadata"]["supervision_arm"] == "answer-only"
        assert row["messages"][-1]["content"].startswith("Answer: ")
        assert len(row["metadata"]["oracle_letters"]) == 2
        source = json.loads(
            (out / "source-natural/DIR-MP-ALT-G2.jsonl")
            .read_text()
            .splitlines()[0]
        )
        assert source["messages"][-1]["content"].startswith("<think>\n")
        assert source["metadata"]["supervision_arm"] == "checked-trace"
        stats = json.loads((out / "target-length-stats.json").read_text())
        assert stats["overall"]["training_model_tokens"]["count"] == 34
    finally:
        if out.exists():
            import shutil

            shutil.rmtree(out)
        if cache.exists():
            import shutil

            shutil.rmtree(cache)


def test_hardening_bundle_preserves_comp_and_shift_sections(monkeypatch):
    monkeypatch.setattr(pilot, "_tokenizer", lambda: _StubTokenizer())
    repo_root = Path(__file__).resolve().parents[3]
    out = repo_root / "experiments/spatialentail-allpossible-pilot/_pytest-hardening"
    cache = pilot._generation_cache_path(out)
    if out.exists():
        pytest.skip("stale hardening pilot output present")
    try:
        receipt = pilot.prepare_pilot(
            repo_root / "spatial/v2/all-possible-hardening-cells.yaml",
            out,
            samples_per_cell=1,
            seed=43,
            max_new_tokens=20000,
            eval_max_tokens=32768,
            workers=14,
        )
        assert receipt["total_rows"] == 14
        assert receipt["generation_workers"] == 14
        assert {task["section"] for task in receipt["tasks"]} == {"comp", "shift"}
        row = json.loads(
            (out / "data/SEL-COMP-CSR-TRC-ALT-G2-D2.jsonl")
            .read_text()
            .splitlines()[0]
        )
        assert "Unstated relations remain unknown rather than false" in row[
            "messages"
        ][0]["content"]
        assert "do not imply equal unit distances" in row["messages"][0]["content"]
        assert row["metadata"]["supervision_arm"] == "answer-only"
        assert row["messages"][-1]["content"].startswith("Answer: ")
        source = json.loads(
            (out / "source-natural/SEL-COMP-CSR-TRC-ALT-G2-D2.jsonl")
            .read_text()
            .splitlines()[0]
        )
        assert source["messages"][-1]["content"].startswith("<think>\n")
        assert "answer_certificate" in source["audit"]
        stats = json.loads((out / "target-length-stats.json").read_text())
        assert stats["overall"]["training_model_tokens"]["count"] == 14
        assert receipt["generation_cache"] == str(cache.relative_to(repo_root))
    finally:
        if out.exists():
            import shutil

            shutil.rmtree(out)
        if cache.exists():
            import shutil

            shutil.rmtree(cache)
