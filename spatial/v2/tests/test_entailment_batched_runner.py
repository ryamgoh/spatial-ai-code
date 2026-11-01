"""CPU-only checks of batched raw replay, progress, and safe restart."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from spatial.v2 import entailment_evaluation as evaluation


@pytest.fixture
def setup_runner(tmp_path, monkeypatch):
    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    for name in ("vllm", "vllm.lora", "vllm.lora.request", "vllm.sampling_params"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm"].SamplingParams = Params
    sys.modules["vllm.lora.request"].LoRARequest = lambda *a: a
    sys.modules["vllm.sampling_params"].StructuredOutputsParams = Params
    monkeypatch.setattr(evaluation, "version", lambda n: "test")
    monkeypatch.setattr(evaluation, "messages", lambda row, style: row)

    def score(rows, records):
        return {
            s: {"correct": len(records), "total": len(rows)}
            for s in ("stage1", "stage2")
        }

    monkeypatch.setattr(evaluation, "score", score)

    class Tokenizer:
        eos_token_id = 1

        def apply_chat_template(self, row, **kwargs):
            return row["id"]

        def encode(self, text, **kwargs):
            return [int(text)] if text.isdigit() else [9]

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 1, "</think>": 2}[token]

        def decode(self, ids, **kwargs):
            return "reasoning</think>\nAnswer: A<|im_end|>"

    class Engine:
        def __init__(self, fail_at=None):
            self.calls = []
            self.fail_at = fail_at

        def generate(self, prompts, params, **kwargs):
            self.calls.append((prompts, params, kwargs))
            if len(self.calls) == self.fail_at:
                raise RuntimeError("interrupted")
            outputs = []
            for p in prompts:
                ids = p["prompt_token_ids"]
                if params.max_tokens == 20:
                    tokens = [100 + ids[0], 2, 1]
                else:
                    assert ids == [ids[0], 100 + ids[0], 2, 1, 9]
                    tokens = [3]
                outputs.append(
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                token_ids=tokens, text="A", finish_reason="stop"
                            )
                        ]
                    )
                )
            return outputs

    (tmp_path / "references.jsonl").write_text("fixed benchmark")
    args = (
        Tokenizer(),
        [{"id": str(i)} for i in range(5)],
        tmp_path,
        tmp_path / "results.json",
        "natural",
        "dev",
        None,
        None,
        {"context": 100, "stage1": 20, "stage2": 4},
    )
    return Engine, args


def test_batches_replay_alignment_and_progress(setup_runner, capsys):
    Engine, args = setup_runner
    engine = Engine()
    result = evaluation.run(engine, *args, batch_size=2)
    assert [len(c[0]) for c in engine.calls] == [2, 2, 2, 2, 1, 1]
    assert all(c[2]["use_tqdm"] for c in engine.calls)
    assert [r["id"] for r in result["outputs"]] == ["0", "1", "2", "3", "4"]
    assert result["execution"]["batch_size"] == 2
    log = capsys.readouterr().out
    assert "Stage 1" in log and "Stage 2" in log and "5/5 completed" in log
    assert "generated tokens=" in log


def test_resume_skips_saved_examples_and_completed_run(setup_runner):
    Engine, args = setup_runner
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluation.run(Engine(fail_at=3), *args, batch_size=2)
    partial = args[3].with_suffix(".partial.jsonl")
    assert len(partial.read_text().splitlines()) == 2
    with partial.open("a") as stream:
        stream.write('{"id":')
    engine = Engine()
    result = evaluation.run(engine, *args, batch_size=2)
    assert len(result["outputs"]) == 5
    assert engine.calls[0][0][0]["prompt_token_ids"] == [2]
    unused = Engine()
    assert evaluation.run(unused, *args, batch_size=2) == result
    assert not unused.calls


def test_resume_rejects_changed_execution_or_benchmark(setup_runner):
    Engine, args = setup_runner
    with pytest.raises(RuntimeError):
        evaluation.run(Engine(fail_at=3), *args, batch_size=2)
    with pytest.raises(ValueError, match="mismatch"):
        evaluation.run(Engine(), *args, batch_size=4)
    (args[2] / "references.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        evaluation.run(Engine(), *args, batch_size=2)


def test_legacy_partial_is_not_silently_reused(setup_runner):
    Engine, args = setup_runner
    args[3].with_suffix(".partial.jsonl").write_text(json.dumps({"id": "0"}) + "\n")
    with pytest.raises(ValueError, match="Legacy partial"):
        evaluation.run(Engine(), *args, batch_size=2)


def test_single_and_batched_mock_outputs_match(setup_runner):
    Engine, args = setup_runner
    batched = evaluation.run(Engine(), *args, batch_size=2)
    serial_args = list(args)
    serial_args[3] = args[3].with_name("serial.json")
    serial = evaluation.run(Engine(), *serial_args, batch_size=1)
    assert batched["outputs"] == serial["outputs"]
    assert batched["scores"] == serial["scores"]


def test_stage2_overflow_is_not_truncated(setup_runner):
    Engine, args = setup_runner
    overflow_args = list(args)
    overflow_args[-1] = {"context": 21, "stage1": 20, "stage2": 20}
    engine = Engine()
    with pytest.raises(ValueError, match="Stage-2 overflow"):
        evaluation.run(engine, *overflow_args, batch_size=2)
    assert len(engine.calls) == 1


def test_concurrent_writer_is_rejected(setup_runner):
    import fcntl

    Engine, args = setup_runner
    with args[3].with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already running"):
            evaluation.run(Engine(), *args, batch_size=2)
