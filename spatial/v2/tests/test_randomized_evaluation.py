import json

import yaml

from spatial.v2.study.randomized_evaluation import prepare_randomized_evaluation
from spatial.v2.study.randomized_v2 import _headline_row


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        prompt = " ".join(item["content"] for item in messages if item["role"] != "assistant") + " <assistant>"
        if add_generation_prompt:
            return prompt
        assistant = next(item["content"] for item in messages if item["role"] == "assistant")
        return prompt + assistant

    def encode(self, text, *, add_special_tokens):
        return text.split()


def test_randomized_evaluation_prepares_48_style_tasks_and_seven_jobs(tmp_path) -> None:
    row = _headline_row("direction", 1, "matched", 1, 0, 42)
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps(row) + "\n")

    receipt = prepare_randomized_evaluation(
        benchmark,
        tmp_path / "prepared-eval",
        repo_root=tmp_path,
        tokenizer=_Tokenizer(),
    )

    assert receipt["total_bases"] == 1
    assert len(receipt["tasks"]) == 1
    assert len(receipt["jobs"]) == 7
    config = yaml.safe_load(
        (tmp_path / "prepared-eval/configs/untuned.yaml").read_text()
    )
    assert config["model_args"]["allow_multiple_answers"] is True
    assert config["model_args"]["max_model_len"] == 8192
