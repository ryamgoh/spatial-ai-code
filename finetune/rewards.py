"""GRPO reward functions for spatial-map multiple choice.

TRL/Axolotl call each function as:
    fn(prompts=..., completions=..., completion_ids=..., oracle_option=..., **kwargs)
and sum the returned per-sample scores (optionally weighted in the YAML).

`oracle_option` is an extra dataset column (e.g. "A" or "A,D") copied from the
synthetic generator gold. Completions are conversational
`[{"role": "assistant", "content": "..."}]` or plain strings.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

ANSWER_RE = re.compile(r"Answer:\s*([A-D](?:\s*,\s*[A-D])*)", re.IGNORECASE)

# SFT traces average ~708 completion tokens. Penalize above this budget.
LENGTH_SOFT_CAP = 1024


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return content if isinstance(content, str) else str(content)
    if isinstance(completion, list):
        parts: list[str] = []
        for message in completion:
            parts.append(_completion_text(message))
        return "\n".join(parts)
    return str(completion)


def _parse_answer_letters(text: str) -> list[str] | None:
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    letters = [tok.strip().upper() for tok in matches[-1].split(",")]
    letters = [tok for tok in letters if tok in {"A", "B", "C", "D"}]
    return letters or None


def _gold_letters(oracle_option: object) -> set[str]:
    raw = str(oracle_option or "").upper()
    return {tok for tok in re.split(r"[,;| ]+", raw) if tok in {"A", "B", "C", "D"}}


def outcome_reward(
    completions: Sequence[object],
    oracle_option: Sequence[object] | None = None,
    **kwargs,
) -> list[float]:
    """1.0 iff predicted letter set equals the oracle set, else 0.0."""
    golds = list(oracle_option) if oracle_option is not None else [""] * len(completions)
    rewards: list[float] = []
    for completion, gold in zip(completions, golds, strict=True):
        predicted = _parse_answer_letters(_completion_text(completion))
        gold_set = _gold_letters(gold)
        if not predicted or not gold_set:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if set(predicted) == gold_set else 0.0)
    return rewards


def format_reward(completions: Sequence[object], **kwargs) -> list[float]:
    """1.0 if the completion contains a parseable `Answer: A[,B...]` line."""
    rewards: list[float] = []
    for completion in completions:
        rewards.append(
            1.0 if _parse_answer_letters(_completion_text(completion)) else 0.0
        )
    return rewards


def length_penalty(
    completions: Sequence[object],
    completion_ids: Sequence[Sequence[int]] | None = None,
    **kwargs,
) -> list[float]:
    """0 until LENGTH_SOFT_CAP tokens, then linear to -1.0 at 2x the cap.

    Uses token ids when TRL provides them; falls back to whitespace tokens.
    """
    rewards: list[float] = []
    for i, completion in enumerate(completions):
        if completion_ids is not None and i < len(completion_ids) and completion_ids[i] is not None:
            n_tokens = len(completion_ids[i])
        else:
            n_tokens = len(_completion_text(completion).split())
        if n_tokens <= LENGTH_SOFT_CAP:
            rewards.append(0.0)
        else:
            over = (n_tokens - LENGTH_SOFT_CAP) / float(LENGTH_SOFT_CAP)
            rewards.append(-min(1.0, over))
    return rewards
