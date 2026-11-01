"""Checkpoint discovery and evaluation guards without GPU execution."""

import json
import subprocess

import pytest

from spatial.v2 import entailment_evaluation as evaluation
from spatial.v2.entailment_thinking_training import MODEL


def fake_run(root):
    prepared = root / "thinking-test/prepared"
    for style in ("natural", "symbolic"):
        for epoch, step in ((1, 750), (2, 1500)):
            checkpoint = prepared / "models" / f"{style}-seed-42/checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": step,
                    }
                )
            )
            (checkpoint / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": MODEL,
                    }
                )
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"test")
    return prepared


def test_checkpoint_epochs_are_metadata_not_example_counts(tmp_path):
    prepared = fake_run(tmp_path)
    paths = evaluation.checkpoints(prepared, "natural")
    assert paths[1].name == "checkpoint-750"
    assert paths[2].name == "checkpoint-1500"
    assert evaluation.discover(prepared) == prepared.resolve()


def test_ambiguous_discovery_does_not_guess(tmp_path, monkeypatch):
    exp = tmp_path / "experiment"
    fake_run(exp / "runs")
    monkeypatch.setattr(evaluation, "EXPERIMENT", exp)
    assert evaluation.discover().name == "prepared"
    first = exp / "runs/thinking-test"
    import shutil

    shutil.copytree(first, exp / "runs/thinking-other")
    with pytest.raises(ValueError, match="one complete"):
        evaluation.discover()


def test_checkpoint_wrong_base_rejected(tmp_path):
    prepared = fake_run(tmp_path)
    path = prepared / "models/natural-seed-42/checkpoint-750/adapter_config.json"
    path.write_text('{"base_model_name_or_path":"wrong"}')
    with pytest.raises(ValueError, match="base mismatch"):
        evaluation.checkpoints(prepared, "natural")


def dev_result(epoch, correct):
    return {
        "epoch": epoch,
        "style": "natural",
        "split": "dev",
        "adapter": f"epoch{epoch}",
        "model": MODEL,
        "revision": evaluation.REVISION,
        "benchmark_sha256": "same",
        "limits": {"stage1": 4096},
        "scores": {"stage1": {"correct": correct, "total": 10}},
    }


def test_selection_accuracy_and_earliest_tie():
    assert evaluation.choose([dev_result(1, 8), dev_result(2, 7)])["epoch"] == 1
    assert evaluation.choose([dev_result(1, 7), dev_result(2, 8)])["epoch"] == 2
    assert evaluation.choose([dev_result(2, 8), dev_result(1, 8)])["epoch"] == 1


def test_selection_requires_identical_protocol():
    second = dev_result(2, 8)
    second["limits"] = {"stage1": 8192}
    with pytest.raises(ValueError, match="provenance"):
        evaluation.choose([dev_result(1, 7), second])


def test_stage2_replay_closes_only_unfinished_thinking():
    class Tokenizer:
        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 1, "</think>": 2}[token]

    assert evaluation.stage2_suffix(Tokenizer(), [3]).startswith("\n</think><|im_end|>")
    assert evaluation.stage2_suffix(Tokenizer(), [2, 3]).startswith("<|im_end|>")
    assert evaluation.stage2_suffix(Tokenizer(), [2, 1]).startswith(
        "\n<|im_start|>user"
    )


def test_external_current_protocol_loads():
    rows = evaluation.load_benchmark(
        evaluation.EXPERIMENT / "spatialmap-unique-v1", "external-test"
    )
    assert len(rows) == 1690


def test_old_scoped_internal_source_rejected():
    with pytest.raises(ValueError, match="all-scene"):
        evaluation.load_benchmark(
            evaluation.EXPERIMENT / "entailment-candidate-v5", "dev"
        )


def test_h200_scripts_have_resources_notifications_and_valid_syntax():
    slurm = evaluation.EXPERIMENT / "slurm"
    for name in ("eval-dev-h200.sh", "eval-selected-h200.sh", "eval-untuned-h200.sh"):
        script = slurm / name
        text = script.read_text()
        assert "--gres=gpu:h200-141:1" in text
        assert "--mail-user=e0958116@u.nus.edu" in text
        subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", "-n", str(slurm / "eval-common.bash")], check=True)


def test_fixed_internal_reconstruction_and_admission():
    import importlib
    import sys

    from transformers import AutoTokenizer

    sys.path.insert(0, str(evaluation.EXPERIMENT))
    materializer = importlib.import_module("prepare_evaluation")
    policy, jobs = materializer.fixed_jobs()
    assert len(jobs) == 4800
    materializer.POLICY = policy
    materializer.tokenizer_loader.TOKENIZER = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-4B",
        revision=materializer.tokenizer_loader.REVISION,
        local_files_only=True,
    )
    for kind in ("direction", "which", "count"):
        job = next(j for j in jobs if j[1] == kind and j[0].split == "dev")
        record = materializer.admit(job)
        assert record["split"] == "dev"
        assert all(
            r["prompt_version"] == "plain-unique-system-rules-v1"
            for r in record["rows"]
        )


def test_preparation_worker_count_validation():
    import argparse
    import importlib
    import sys

    sys.path.insert(0, str(evaluation.EXPERIMENT))
    materializer = importlib.import_module("prepare_evaluation")
    assert materializer.positive_workers("128") == 128
    for value in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError, match="at least 1"):
            materializer.positive_workers(value)


def test_preparation_workers_follow_slurm_allocation():
    script = evaluation.EXPERIMENT / "slurm/prepare-evaluation.sh"
    text = script.read_text()
    assert '--workers "${SLURM_CPUS_PER_TASK:?' in text
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert f"export {variable}=1" in text
    subprocess.run(["bash", "-n", str(script)], check=True)
