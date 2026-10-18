import json

import pytest

from spatial.v2.evaluation_chat import (
    CHAT_INPUT_PREFIX,
    encode_chat_input,
    render_chat_input,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"


def test_chat_payload_uses_native_template_and_thinking_switch() -> None:
    payload = encode_chat_input(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "problem"},
        ]
    )
    tokenizer = _Tokenizer()

    assert payload.startswith(CHAT_INPUT_PREFIX)
    assert render_chat_input(
        payload,
        tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    ) == "rendered"
    messages, kwargs = tokenizer.calls[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_chat_payload_can_prepend_system_for_templates_without_system_role() -> None:
    payload = CHAT_INPUT_PREFIX + json.dumps(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "problem"},
        ]
    )
    tokenizer = _Tokenizer()

    render_chat_input(payload, tokenizer, system_message_policy="prepend_user")

    messages, _ = tokenizer.calls[0]
    assert messages == [{"role": "user", "content": "rules\n\nproblem"}]


def test_chat_payload_rejects_assistant_content() -> None:
    with pytest.raises(ValueError, match="exactly one system and one user"):
        encode_chat_input(
            [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "problem"},
                {"role": "assistant", "content": "Answer: A"},
            ]
        )
