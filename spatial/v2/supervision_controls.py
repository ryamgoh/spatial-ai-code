"""Paired supervision controls derived from one checked training row."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from enum import Enum
from typing import Any

from spatial.v2.grading import MenuAnswer
from spatial.v2.solver import Direction, SpatialProblem
from spatial.v2.symbolic_trace_codec import score_symbolic_training_trace
from spatial.v2.trace import TraceFormat


class SupervisionArm(str, Enum):
    CHECKED_TRACE = "checked-trace"
    ANSWER_ONLY = "answer-only"
    CORRUPTED_TRACE = "corrupted-trace"


def _assistant_message(row: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        message
        for message in row.get("messages", ())
        if message.get("role") == "assistant"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("content"), str):
        raise ValueError("a supervision row needs exactly one assistant completion")
    return matches[0]


def _checked_completion(content: str) -> tuple[str, str]:
    prefix = "<think>\n"
    separator = "\n</think>\n"
    if not content.startswith(prefix) or content.count(separator) != 1:
        raise ValueError("control variants require one checked <think> completion")
    trace, answer = content[len(prefix) :].split(separator)
    if not trace or not answer.startswith("Answer:"):
        raise ValueError("checked completion has an invalid trace or answer")
    return trace, answer


def _annotate(row: dict[str, Any], arm: SupervisionArm) -> dict[str, Any]:
    content = str(_assistant_message(row)["content"])
    metadata = row.setdefault("metadata", {})
    metadata["supervision_arm"] = arm.value
    metadata["expected_process_valid"] = (
        None
        if arm is SupervisionArm.ANSWER_ONLY
        else arm is SupervisionArm.CHECKED_TRACE
    )
    metadata["process_status"] = (
        "absent"
        if arm is SupervisionArm.ANSWER_ONLY
        else "valid"
        if arm is SupervisionArm.CHECKED_TRACE
        else "invalid"
    )
    metadata["target_characters"] = len(content)
    metadata["target_whitespace_tokens"] = len(content.split())
    return row


def _evidence_mutations(value: Any, path: str = "reasoning"):
    """Change typed logical assertions, retaining every schema key and record."""
    if isinstance(value, dict):
        if value.get("kind") == "model":
            yield value, "expected", not value["expected"], f"{path}.expected"
        if value.get("kind") == "axis":
            replacement = ">" if value["relation"] != ">" else "<"
            yield value, "relation", replacement, f"{path}.relation"
        if value.get("kind") == "relation":
            replacement = next(
                [direction.name]
                for direction in Direction
                if [direction.name] != value["directions"]
            )
            yield value, "directions", replacement, f"{path}.directions"
        for key, child in value.items():
            yield from _evidence_mutations(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _evidence_mutations(child, f"{path}[{index}]")


def _corrupt_trace(
    trace: str,
    problem: SpatialProblem,
    expected: MenuAnswer,
) -> tuple[str, dict[str, Any]]:
    if not score_symbolic_training_trace(problem, trace, expected).fully_valid:
        raise ValueError("corrupted-trace requires a replay-checked source trace")
    reasoning, decision = trace.splitlines()
    payload = json.loads(reasoning)
    for node, key, replacement, path in _evidence_mutations(payload):
        original = node[key]
        node[key] = replacement
        corrupted = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            + decision
        )
        score = score_symbolic_training_trace(problem, corrupted, expected)
        node[key] = original
        if not score.reasoning_valid and score.decision_valid:
            return corrupted, {
                "kind": "logical-assertion",
                "path": path,
                "original": original,
                "replacement": replacement,
                "replay_error": score.error,
                "decision_preserved": True,
                "scope": "Symbolic",
            }
    raise ValueError("no replay-rejected semantic corruption found")


def apply_supervision_arm(
    row: Mapping[str, Any],
    arm: SupervisionArm | str,
    *,
    problem: SpatialProblem | None = None,
    expected: MenuAnswer | None = None,
) -> dict[str, Any]:
    """Derive one answer-matched control without mutating the checked row."""
    arm = SupervisionArm(arm)
    result = deepcopy(dict(row))
    message = _assistant_message(result)
    trace, answer = _checked_completion(str(message["content"]))
    if arm is SupervisionArm.CHECKED_TRACE:
        return _annotate(result, arm)
    if arm is SupervisionArm.ANSWER_ONLY:
        message["content"] = answer
    else:
        if row.get("metadata", {}).get("trace_format") != TraceFormat.SYMBOLIC.value:
            raise ValueError("corrupted-trace supports only Symbolic traces")
        if problem is None or expected is None:
            raise ValueError(
                "corrupted-trace requires problem and expected menu answer"
            )
        corrupted_trace, corruption = _corrupt_trace(trace, problem, expected)
        message["content"] = f"<think>\n{corrupted_trace}\n</think>\n{answer}"
        result.setdefault("metadata", {})["corruption"] = corruption
    result["id"] = f"{row.get('id', '')}-{arm.value}"
    return _annotate(result, arm)


def build_supervision_variants(
    row: Mapping[str, Any],
    arms: Iterable[SupervisionArm | str],
    *,
    problem: SpatialProblem | None = None,
    expected: MenuAnswer | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build a duplicate-free set of paired supervision controls."""
    normalized = tuple(SupervisionArm(arm) for arm in arms)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("supervision arms must be non-empty and unique")
    return tuple(
        apply_supervision_arm(row, arm, problem=problem, expected=expected)
        for arm in normalized
    )


def annotate_model_token_counts(
    rows: Iterable[Mapping[str, Any]],
    count_tokens: Callable[[str], int],
) -> tuple[dict[str, Any], ...]:
    """Attach exact target counts from the tokenizer selected for training."""
    annotated = []
    for row in rows:
        result = deepcopy(dict(row))
        content = str(_assistant_message(result)["content"])
        count = count_tokens(content)
        if type(count) is not int or count <= 0:
            raise ValueError("token counter must return a positive integer")
        result.setdefault("metadata", {})["target_model_tokens"] = count
        annotated.append(result)
    return tuple(annotated)
