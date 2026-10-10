"""Exact chat-template context admission for paired supervision groups."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContextBudget:
    tokenizer_name: str
    train_max_tokens: int
    eval_max_tokens: int
    max_new_tokens: int
    target_max_tokens: int | None = None
    tokenizer_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.tokenizer_name.strip():
            raise ValueError("tokenizer_name must not be empty")
        for name in (
            "train_max_tokens",
            "eval_max_tokens",
            "max_new_tokens",
            "target_max_tokens",
        ):
            value = getattr(self, name)
            if value is None and name == "target_max_tokens":
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_new_tokens >= self.eval_max_tokens:
            raise ValueError("max_new_tokens must leave room for the evaluation prompt")


@dataclass(frozen=True)
class GroupAdmission:
    rows: tuple[dict[str, Any], ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    row_rejections: dict[str, list[str]]


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _preserves_target(content: str, prompt: str, suffix: str) -> bool:
    # Qwen pre-fills the opening thinking tag and inserts one extra newline
    # between the closing tag and answer. Neither may reorder target content.
    if content.startswith("<think>\n"):
        if prompt.endswith("<think>\n"):
            suffix = "<think>\n" + suffix
        normalize = lambda text: re.sub(r"</think>\n+", "</think>\n", text)
        return normalize(suffix).startswith(normalize(content))
    if prompt.endswith("<think>\n"):
        suffix = suffix.lstrip("\n")
        if suffix.startswith("</think>\n"):
            suffix = suffix[len("</think>\n") :].lstrip("\n")
    return suffix.startswith(content)


def validate_context_admission(row: Mapping[str, Any]) -> None:
    """Check persisted admission integrity without reloading a tokenizer.

    Fingerprints detect stale or altered row data; they are not authentication.
    """
    metadata = row.get("metadata", {})
    if metadata.get("context_group_accepted") is not True:
        raise ValueError("row does not have successful context admission")
    if metadata.get("context_rejection_reasons") != []:
        raise ValueError("admitted row has missing or nonempty rejection reasons")
    messages = row.get("messages", ())
    if (
        not messages
        or messages[-1].get("role") != "assistant"
        or sum(message.get("role") == "assistant" for message in messages) != 1
    ):
        raise ValueError("admitted row requires one final assistant completion")
    prompt = metadata.get("evaluation_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("admitted row requires a rendered evaluation prompt")
    for field, value in (
        ("prompt_source_sha256", messages[:-1]),
        ("evaluation_prompt_sha256", prompt),
        ("context_messages_sha256", messages),
    ):
        if metadata.get(field) != _fingerprint(value):
            raise ValueError(f"context admission fingerprint mismatch: {field}")
    try:
        budget = ContextBudget(**metadata["context_budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("admitted row has invalid context budget") from exc
    if (
        metadata.get("context_tokenizer") != budget.tokenizer_name
        or metadata.get("context_tokenizer_revision") != budget.tokenizer_revision
    ):
        raise ValueError("admitted tokenizer does not match context budget")
    for field in (
        "training_model_tokens",
        "evaluation_prompt_model_tokens",
        "evaluation_reserved_model_tokens",
        "target_model_tokens",
        "generation_target_tokens",
    ):
        if type(metadata.get(field)) is not int or metadata[field] <= 0:
            raise ValueError(f"admitted row has invalid token count: {field}")
    if (
        metadata["evaluation_reserved_model_tokens"]
        != metadata["evaluation_prompt_model_tokens"] + budget.max_new_tokens
    ):
        raise ValueError("admitted evaluation reserve does not match budget")
    if (
        metadata["training_model_tokens"] > budget.train_max_tokens
        or metadata["evaluation_reserved_model_tokens"] > budget.eval_max_tokens
        or metadata["generation_target_tokens"]
        > min(budget.max_new_tokens, budget.target_max_tokens or budget.max_new_tokens)
    ):
        raise ValueError("admitted row exceeds its context budget")


def load_tokenizer(budget: ContextBudget) -> Any:
    """Load the configured tokenizer using the project's existing dependency."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        budget.tokenizer_name,
        revision=budget.tokenizer_revision,
    )


def admit_row_group(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    budget: ContextBudget,
) -> GroupAdmission:
    """Admit every format/control for one semantic instance, or reject all of them."""
    annotated = tuple(deepcopy(dict(row)) for row in rows)
    if not annotated:
        raise ValueError("context admission requires a nonempty semantic group")
    base_ids = {row.get("metadata", {}).get("base_id") for row in annotated}
    if len(base_ids) != 1 or None in base_ids:
        raise ValueError("context admission requires one shared base_id")
    row_ids = [str(row["id"]) for row in annotated]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("context admission requires unique row IDs")

    rejections = {}
    for row in annotated:
        messages = row["messages"]
        assistants = [message for message in messages if message["role"] == "assistant"]
        if len(assistants) != 1 or messages[-1]["role"] != "assistant":
            raise ValueError(
                "context admission requires one final assistant completion"
            )
        # Rendering first avoids BatchEncoding length being mistaken for token length.
        training_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        evaluation_prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        training_tokens, prompt_tokens, target_tokens = (
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in (training_text, evaluation_prompt, assistants[0]["content"])
        )
        if not training_text.startswith(evaluation_prompt):
            raise ValueError("training chat must extend the rendered evaluation prompt")
        completion_suffix = training_text[len(evaluation_prompt) :]
        if not _preserves_target(
            assistants[0]["content"], evaluation_prompt, completion_suffix
        ):
            raise ValueError(
                "chat template dropped or changed assistant target content"
            )
        generation_target_tokens = len(
            tokenizer.encode(
                completion_suffix,
                add_special_tokens=False,
            )
        )
        reasons = []
        if training_tokens > budget.train_max_tokens:
            reasons.append("train_context_exceeded")
        if prompt_tokens + budget.max_new_tokens > budget.eval_max_tokens:
            reasons.append("eval_context_exceeded")
        target_cap = min(
            budget.max_new_tokens, budget.target_max_tokens or budget.max_new_tokens
        )
        if generation_target_tokens > target_cap:
            reasons.append("target_cap_exceeded")
        row.setdefault("metadata", {}).update(
            evaluation_prompt=evaluation_prompt,
            training_model_tokens=training_tokens,
            evaluation_prompt_model_tokens=prompt_tokens,
            evaluation_reserved_model_tokens=prompt_tokens + budget.max_new_tokens,
            target_model_tokens=target_tokens,
            generation_target_tokens=generation_target_tokens,
            context_tokenizer=budget.tokenizer_name,
            context_tokenizer_revision=budget.tokenizer_revision,
            context_budget=asdict(budget),
            prompt_source_sha256=_fingerprint(messages[:-1]),
            evaluation_prompt_sha256=_fingerprint(evaluation_prompt),
            context_messages_sha256=_fingerprint(messages),
            context_rejection_reasons=reasons,
        )
        if reasons:
            rejections[str(row["id"])] = reasons
    accepted = not rejections
    for row in annotated:
        row["metadata"]["context_group_accepted"] = accepted
    return GroupAdmission(
        annotated,
        accepted,
        tuple(
            sorted({reason for reasons in rejections.values() for reason in reasons})
        ),
        rejections,
    )


def summarize_admissions(
    admissions: Iterable[GroupAdmission],
    budget: ContextBudget,
) -> dict[str, Any]:
    """Persist group counts and the exact rows/reasons responsible for rejection."""
    admissions = tuple(admissions)
    reasons = Counter(
        reason for admission in admissions for reason in admission.rejection_reasons
    )
    return {
        "config": asdict(budget),
        "accepted_groups": sum(admission.accepted for admission in admissions),
        "rejected_groups": sum(not admission.accepted for admission in admissions),
        "accepted_rows": sum(
            len(admission.rows) for admission in admissions if admission.accepted
        ),
        "rejected_rows": sum(
            len(admission.rows) for admission in admissions if not admission.accepted
        ),
        "rejection_reasons": dict(sorted(reasons.items())),
        "rejections": [
            {
                "base_id": admission.rows[0]["metadata"]["base_id"],
                "row_rejections": admission.row_rejections,
                "rejected_row_ids": [row["id"] for row in admission.rows],
            }
            for admission in admissions
            if not admission.accepted
        ],
    }


def rejection_output_path(output_file: str | Path) -> Path:
    path = Path(output_file)
    base = path.with_suffix("") if path.suffix else path
    return base.with_name(base.name + "_rejected.json")


def write_rejection_report(
    output_file: str | Path,
    admissions: Iterable[GroupAdmission],
    budget: ContextBudget,
    *,
    cell: str,
) -> Path:
    """Write admission diagnostics when generation cannot fill a requested cell."""
    admissions = tuple(admissions)
    report = summarize_admissions(admissions, budget)
    report.update(
        cell=cell,
        failed_rows=[
            {"id": row["id"], "metadata": row["metadata"]}
            for admission in admissions
            if not admission.accepted
            for row in admission.rows
        ],
    )
    path = rejection_output_path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
