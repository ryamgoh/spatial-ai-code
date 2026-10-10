"""Chat-template accounting and all-or-none paired context admission."""

from copy import deepcopy

import pytest

from spatial.v2.context_budget import (
    ContextBudget,
    admit_row_group,
    summarize_admissions,
)


class FixtureTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(
            f"<{message['role']}>{message['content']}|" for message in messages
        )
        text += "<assistant>" if add_generation_prompt else "<end>"
        # Current transformers can return this mapping, whose length is not tokens.
        return (
            {"input_ids": list(text), "attention_mask": [1] * len(text)}
            if tokenize
            else text
        )

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text)


def _row(row_id="natural", completion="proof"):
    return {
        "id": row_id,
        "metadata": {"base_id": "one-instance"},
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": completion},
        ],
    }


def test_template_boundaries_and_generation_reserve_are_counted_exactly():
    tokenizer = FixtureTokenizer()
    row = _row()
    original = deepcopy(row)
    admission = admit_row_group(
        [row], tokenizer, ContextBudget("fixture", 100, 100, 20, 11)
    )
    metadata = admission.rows[0]["metadata"]
    assert admission.accepted
    assert metadata["evaluation_prompt"] == "<system>rules|<user>question|<assistant>"
    assert metadata["evaluation_prompt_model_tokens"] == 40
    assert metadata["evaluation_reserved_model_tokens"] == 60
    assert metadata["training_model_tokens"] == 51
    assert metadata["target_model_tokens"] == 5
    assert row == original


@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        ({"train_max_tokens": 51}, "train_context_exceeded"),
        ({"eval_max_tokens": 57}, "eval_context_exceeded"),
        ({"target_max_tokens": 11}, "target_cap_exceeded"),
    ],
)
def test_one_over_budget_variant_rejects_whole_semantic_group(limits, reason):
    config = {
        "tokenizer_name": "fixture",
        "train_max_tokens": 200,
        "eval_max_tokens": 100,
        "max_new_tokens": 20,
    }
    config.update(limits)
    budget = ContextBudget(**config)
    rows = [_row("short", "proof"), _row("long", "proof extra")]
    if reason == "eval_context_exceeded":
        rows[0]["messages"][1]["content"] = "q"
    admission = admit_row_group(rows, FixtureTokenizer(), budget)
    assert not admission.accepted
    assert admission.rejection_reasons == (reason,)
    assert admission.row_rejections == {"long": [reason]}
    assert all(not row["metadata"]["context_group_accepted"] for row in admission.rows)
    summary = summarize_admissions([admission], budget)
    assert summary["rejected_groups"] == 1
    assert summary["rejected_rows"] == 2
    assert summary["rejection_reasons"] == {reason: 1}
    assert summary["rejections"][0]["rejected_row_ids"] == ["short", "long"]


def test_all_three_limits_accept_exact_boundary():
    result = admit_row_group(
        [_row()], FixtureTokenizer(), ContextBudget("fixture", 51, 60, 20, 11)
    )
    assert result.accepted


def test_admission_rejects_mixed_semantic_groups():
    other = _row("other")
    other["metadata"]["base_id"] = "another-instance"
    with pytest.raises(ValueError, match="shared base_id"):
        admit_row_group(
            [_row(), other], FixtureTokenizer(), ContextBudget("fixture", 100, 100, 20)
        )


def test_generation_target_must_fit_reserve_even_when_training_fits():
    admission = admit_row_group(
        [_row()], FixtureTokenizer(), ContextBudget("fixture", 100, 100, 10)
    )
    assert not admission.accepted
    assert admission.rejection_reasons == ("target_cap_exceeded",)
    assert admission.rows[0]["metadata"]["generation_target_tokens"] == 11


@pytest.mark.parametrize("mode", ["prefix", "target"])
def test_incompatible_or_lossy_chat_template_is_rejected(mode):
    class LossyTokenizer(FixtureTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            text = super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
            if not add_generation_prompt:
                return (
                    ("different prefix" + text)
                    if mode == "prefix"
                    else text.replace("proof", "")
                )
            return text

    with pytest.raises(ValueError, match="training chat|target content"):
        admit_row_group(
            [_row()], LossyTokenizer(), ContextBudget("fixture", 100, 100, 20)
        )


@pytest.mark.parametrize(
    "replacement",
    [
        "Answer: A\nproof",
        "<think>\nproof\n</think>\nproof",
        "<think>\nproof\nAnswer: A",
    ],
)
def test_template_cannot_reorder_drop_or_merge_target_parts(replacement):
    source = "<think>\nproof\n</think>\nAnswer: A"

    class ReorderingTokenizer(FixtureTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            text = super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
            return (
                text.replace(source, replacement) if not add_generation_prompt else text
            )

    with pytest.raises(ValueError, match="target content"):
        admit_row_group(
            [_row(completion=source)],
            ReorderingTokenizer(),
            ContextBudget("fixture", 200, 200, 100),
        )


@pytest.mark.parametrize(
    "field",
    [
        "evaluation_prompt",
        "source",
        "target",
        "context_group_accepted",
        "training_model_tokens",
        "context_tokenizer",
    ],
)
def test_admitted_row_integrity_detects_tampering(field):
    from spatial.v2.context_budget import validate_context_admission

    admission = admit_row_group(
        [_row()], FixtureTokenizer(), ContextBudget("fixture", 100, 100, 20)
    )
    row = admission.rows[0]
    validate_context_admission(row)
    if field == "source":
        row["messages"][0]["content"] = "changed system prompt"
    elif field == "target":
        row["messages"][-1]["content"] = "different target"
    else:
        row["metadata"][field] = (
            False
            if field == "context_group_accepted"
            else -1
            if field == "training_model_tokens"
            else "tampered"
        )
    with pytest.raises(ValueError):
        validate_context_admission(row)


def test_rejection_report_preserves_failed_group_token_counts(tmp_path):
    import json

    from spatial.v2.context_budget import rejection_output_path, write_rejection_report

    budget = ContextBudget("fixture", 100, 100, 10)
    admission = admit_row_group([_row()], FixtureTokenizer(), budget)
    output = tmp_path / "nested" / "pilot.jsonl"
    path = write_rejection_report(output, [admission], budget, cell="hard")
    assert path == rejection_output_path(output)
    report = json.loads(path.read_text())
    assert report["cell"] == "hard"
    assert report["rejected_groups"] == 1
    assert report["failed_rows"][0]["metadata"]["generation_target_tokens"] == 11
