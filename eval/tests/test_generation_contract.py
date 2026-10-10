"""Real request preparation for both stages, without GPU libraries or AST mocks."""

from types import SimpleNamespace

import pytest

from eval.generation_contract import (
    finish_answer,
    prepare_requests,
    reasoning_without_answer,
)


def tokenizer():
    return SimpleNamespace(
        encode=lambda text, add_special_tokens: list(
            range(len(text) + int(add_special_tokens))
        )
    )


def test_each_request_keeps_its_budget_and_decoding_settings():
    prompts, params = prepare_requests(
        [
            (
                "abc",
                {
                    "max_gen_toks": 5,
                    "temperature": 0,
                    "repetition_penalty": 1,
                    "until": ["stop"],
                },
            ),
            ("defg", {"max_gen_toks": 9, "temperature": 0.2}),
        ],
        tokenizer(),
        context_limit=100,
        default_max_tokens=20,
        add_special_tokens=False,
    )
    assert prompts == [
        {"prompt_token_ids": [0, 1, 2]},
        {"prompt_token_ids": [0, 1, 2, 3]},
    ]
    assert [p["max_tokens"] for p in params] == [5, 9]
    assert params[0]["temperature"] == 0
    assert params[0]["repetition_penalty"] == 1
    assert params[0]["stop"] == ["stop"]
    assert params[1]["temperature"] == 0.2
    assert params[1]["repetition_penalty"] == 1.1


@pytest.mark.parametrize("budget", [0, -1, True, 3.5])
def test_invalid_generation_budget_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        prepare_requests(
            [("prompt", {"max_gen_toks": budget})],
            tokenizer(),
            context_limit=100,
            default_max_tokens=20,
            add_special_tokens=False,
        )


def test_stage_two_includes_reasoning_and_answer_reserve_in_context():
    prompt = "p" * 70
    prepare_requests(
        [(prompt, {})],
        tokenizer(),
        context_limit=100,
        default_max_tokens=20,
        add_special_tokens=False,
    )
    reasoning = "r" * 20
    with pytest.raises(ValueError, match="context"):
        prepare_requests(
            [(f"{prompt}\n{reasoning}\n\nAnswer: ", {"max_gen_toks": 16})],
            tokenizer(),
            context_limit=100,
            default_max_tokens=16,
            add_special_tokens=False,
        )


def test_both_stages_preserve_special_token_contract():
    for budget in (20, 16):
        prompts, _ = prepare_requests(
            [("prompt", {})],
            tokenizer(),
            context_limit=100,
            default_max_tokens=budget,
            add_special_tokens=False,
        )
        assert len(prompts[0]["prompt_token_ids"]) == len("prompt")


def test_second_stage_replaces_old_footer_and_canonicalizes_letters():
    first = "<think>proof</think>\nAnswer: B\nAnswer: A"
    reasoning = reasoning_without_answer(first)
    assert reasoning == "<think>proof</think>"
    final = finish_answer(reasoning, "C,A,C", ["A", "B", "C"])
    assert final == "<think>proof</think>\n\nAnswer: A, C"
    assert final.count("Answer:") == 1


def test_invalid_second_stage_answer_is_not_emitted():
    with pytest.raises(ValueError, match="invalid option"):
        finish_answer("proof", "A,Z", ["A", "B"])
