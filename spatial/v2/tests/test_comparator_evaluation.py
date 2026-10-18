import json

import yaml

from spatial.v2.evaluation_chat import CHAT_INPUT_PREFIX
from spatial.v2.study.comparator_evaluation import (
    GENERAL_INSTRUCTION_MODELS,
    prepare_general_instruction_evaluation,
)
from spatial.v2.study.randomized_v2 import _headline_row


def test_general_instruction_batch_has_the_six_agreed_conditions() -> None:
    assert [model.name for model in GENERAL_INSTRUCTION_MODELS] == [
        "qwen3-5-4b-thinking-off",
        "qwen3-5-9b-thinking-off",
        "gemma-4-e4b-it",
        "gemma-3-4b-it",
        "ministral-3-8b-instruct",
        "phi-4-mini-instruct",
    ]
    assert all(len(model.revision) == 40 for model in GENERAL_INSTRUCTION_MODELS)


def test_prepare_general_instruction_evaluation_uses_model_neutral_chat(tmp_path) -> None:
    row = _headline_row("direction", 1, "matched", 1, 0, 42)
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps(row) + "\n")
    output = tmp_path / "prepared-general-eval"

    receipt = prepare_general_instruction_evaluation(
        benchmark,
        output,
        repo_root=tmp_path,
    )

    assert receipt["total_bases"] == 1
    assert len(receipt["tasks"]) == 1
    assert len(receipt["jobs"]) == 6
    data = json.loads(next((output / "data").glob("*.jsonl")).read_text())
    assert data["model_input"].startswith(CHAT_INPUT_PREFIX)
    assert "<|im_start|>" not in data["model_input"]
    assert [message["role"] for message in data["messages"]] == ["system", "user"]

    qwen = yaml.safe_load(
        (output / "configs/qwen3-5-9b-thinking-off.yaml").read_text()
    )
    assert qwen["model"] == "vllm_staged_chat_pass"
    assert qwen["model_args"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert qwen["model_args"]["stages"] == 2
    assert qwen["model_args"]["allow_multiple_answers"] is True

    gemma3 = yaml.safe_load((output / "configs/gemma-3-4b-it.yaml").read_text())
    assert gemma3["model_args"]["system_message_policy"] == "prepend_user"
    smoke = yaml.safe_load(
        (output / "configs/smoke/gemma-3-4b-it.yaml").read_text()
    )
    assert smoke["limit"] == 1
    submit = (output / "submit-h200-chain.sh").read_text()
    assert submit.count("/slurm/") == 6
    assert 'dependency=(--dependency="afterok:$job_id")' in submit
    smoke_submit = (output / "submit-h200-smoke-chain.sh").read_text()
    assert smoke_submit.count("/slurm/smoke/") == 6
