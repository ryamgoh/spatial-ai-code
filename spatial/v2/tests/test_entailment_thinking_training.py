"""Regression checks for native thinking serialization and explicit loss masks."""

from copy import deepcopy
from pathlib import Path

import pytest

from spatial.v2.entailment_thinking_training import (
    MODEL,
    REVISION,
    audit_tokenized,
    matched_configs,
    prepare,
    thinking_messages,
    tokenize,
    training_config,
    validate_model,
)


@pytest.fixture
def model():
    return {
        "model_id": MODEL,
        "tokenizer_id": MODEL,
        "revision": REVISION,
        "tokenizer_revision": REVISION,
        "thinking_mode": "thinking-only",
        "chat_template_kwargs": {},
        "arms": ["natural", "symbolic"],
        "training_candidate": "unique-training-candidate-v3",
        "context_length": 8192,
        "stage1_max_new_tokens": 4096,
        "stage2_max_new_tokens": 16,
    }


@pytest.fixture
def source():
    return [
        {"role": "system", "content": "Explain and end with Answer: X."},
        {"role": "user", "content": "Gym is north of Bank. Which direction?\nA. North"},
        {"role": "assistant", "content": "P1 places Gym north of Bank.\n\nAnswer: A"},
    ]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )


def test_preserves_source_and_separates_answer(source):
    original = deepcopy(source)
    messages = thinking_messages(source, "A")
    assert source == original
    assert messages[:-1] == source[:-1]
    assert messages[-1] == {
        "role": "assistant",
        "reasoning_content": "P1 places Gym north of Bank.",
        "content": "Answer: A",
    }


@pytest.mark.parametrize(
    "target",
    [
        "Answer: A",
        "Proof.\n\nAnswer: B",
        "<think>Proof.</think>\n\nAnswer: A",
        "Proof.<|im_end|>\n\nAnswer: A",
        "Answer: A\nMore proof.\n\nAnswer: A",
        "Proof.\n\nAnswer: A\n",
    ],
)
def test_rejects_invalid_or_preformatted_targets(source, target):
    source[-1]["content"] = target
    with pytest.raises(ValueError):
        thinking_messages(source, "A")


@pytest.mark.parametrize("answer", list("ABCDEF"))
def test_real_native_template_and_labels(tokenizer, model, source, answer):
    source[-1]["content"] = f"P1 supports this answer.\n\nAnswer: {answer}"
    row, messages, measured = tokenize(tokenizer, source, answer, model)
    boundary = measured["prompt_tokens"]
    prefix = tokenizer.decode(row["input_ids"][:boundary])
    target = tokenizer.decode(row["input_ids"][boundary:])
    assert prefix.endswith("<|im_start|>assistant\n<think>\n")
    assert prefix.count("<think>") == 1
    assert "<think>" not in target
    assert (
        target == f"P1 supports this answer.\n</think>\n\nAnswer: {answer}<|im_end|>\n"
    )
    assert row["labels"][:boundary] == [-100] * boundary
    assert row["labels"][boundary:] == row["input_ids"][boundary:]
    assert messages[:-1] == source[:-1]


@pytest.mark.parametrize(
    "mutation", ["prompt_label", "target_mask", "target_label", "attention"]
)
def test_independent_mask_check_rejects_mutations(tokenizer, model, source, mutation):
    row, _, measured = tokenize(tokenizer, source, "A", model)
    boundary = measured["prompt_tokens"]
    prefix = tokenizer.decode(row["input_ids"][:boundary])
    suffix = tokenizer.decode(row["input_ids"][boundary:])
    if mutation == "prompt_label":
        row["labels"][0] = row["input_ids"][0]
    elif mutation == "target_mask":
        row["labels"][boundary + 1] = -100
    elif mutation == "target_label":
        row["labels"][boundary + 1] = 0
    else:
        row["attention_mask"][0] = 0
    with pytest.raises(ValueError):
        audit_tokenized(tokenizer, row, prefix, suffix)


@pytest.mark.parametrize(
    "key,value",
    [
        ("context_length", 10),
        ("stage1_max_new_tokens", 2),
        ("stage1_max_new_tokens", 8192),
    ],
)
def test_no_silent_truncation(tokenizer, model, source, key, value):
    model[key] = value
    with pytest.raises(ValueError):
        tokenize(tokenizer, source, "A", model)


def test_configs_matched_and_no_random_dev_split(model):
    configs = {
        s: training_config(model, s, "prepared") for s in ("natural", "symbolic")
    }
    matched_configs(configs)
    for config in configs.values():
        assert config["base_model"] == MODEL
        assert config["revision_of_model"] == REVISION
        assert config["tokenizer_config"] == "prepared/tokenizer"
        assert config["val_set_size"] == 0
        assert config["datasets"][0]["type"] is None
        assert config["num_epochs"] == 2
        assert not any("linear_attn" in s for s in config["lora_target_modules"])
        assert not config["sample_packing"]
        assert not config["logging_nan_inf_filter"]
        assert config["use_wandb"] is True
        assert config["wandb_project"] == "spatialentail-composition-v1-sft"
        assert config["wandb_mode"] == "online"
    assert configs["natural"]["wandb_run_id"] != configs["symbolic"]["wandb_run_id"]
    assert configs["natural"]["wandb_run_id"] != training_config(
        model, "natural", "another-run"
    )["wandb_run_id"]
    configs["symbolic"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="different settings"):
        matched_configs(configs)


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_id", "Qwen/Qwen3.5-4B"),
        ("revision", "main"),
        ("chat_template_kwargs", {"enable_thinking": False}),
        ("thinking_mode", "non-thinking"),
    ],
)
def test_rejects_wrong_model_policy(model, key, value):
    model[key] = value
    with pytest.raises(ValueError):
        validate_model(model)


def test_existing_output_is_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="overwrite"):
        prepare(Path("does-not-exist"), tmp_path)
