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
        return prefix + "<assistant>\n" + messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text) // 4)))


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


def test_pilot_bundle_is_set_valued_and_hash_consistent(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "_tokenizer", lambda: _StubTokenizer())
    repo_root = Path(__file__).resolve().parents[3]
    out = repo_root / "experiments/spatialentail-allpossible-pilot/_pytest"
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
        assert len(row["metadata"]["oracle_letters"]) == 2
    finally:
        if out.exists():
            import shutil

            shutil.rmtree(out)
