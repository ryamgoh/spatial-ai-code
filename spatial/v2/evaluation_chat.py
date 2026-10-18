"""Model-neutral chat payloads for cross-family SpatialEntail evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

CHAT_INPUT_PREFIX = "spatialentail-chat-v1:"


def encode_chat_input(messages: Sequence[Mapping[str, Any]]) -> str:
    """Serialize one system/user exchange without applying a model template."""
    normalized = [
        {
            "role": str(message.get("role") or ""),
            "content": str(message.get("content") or ""),
        }
        for message in messages
    ]
    if [message["role"] for message in normalized] != ["system", "user"]:
        raise ValueError(
            "comparator input requires exactly one system and one user message"
        )
    if any(not message["content"] for message in normalized):
        raise ValueError("comparator input messages must be nonempty")
    return CHAT_INPUT_PREFIX + json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":")
    )


def render_chat_input(
    payload: str,
    tokenizer: Any,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    system_message_policy: str = "native",
) -> str:
    """Render a serialized exchange using the evaluated model's own template."""
    if not payload.startswith(CHAT_INPUT_PREFIX):
        raise ValueError("comparator prompt is missing the chat payload prefix")
    try:
        messages = json.loads(payload[len(CHAT_INPUT_PREFIX) :])
    except json.JSONDecodeError as error:
        raise ValueError("comparator chat payload is invalid JSON") from error
    if not isinstance(messages, list):
        raise ValueError("comparator chat payload must contain a message list")
    encode_chat_input(messages)
    if system_message_policy == "prepend_user":
        messages = [
            {
                "role": "user",
                "content": (
                    f"{messages[0]['content']}\n\n{messages[1]['content']}"
                ),
            }
        ]
    elif system_message_policy != "native":
        raise ValueError(f"unknown system-message policy: {system_message_policy}")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **dict(chat_template_kwargs or {}),
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("model chat template returned an empty prompt")
    return rendered
