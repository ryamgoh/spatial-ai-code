"""Evaluation-loader contract for Spatial V2 matrix rows."""

from __future__ import annotations

from utils import process_docs_v2_sft, strict_acc


class FakeDataset(list):
    def map(self, fn):
        return FakeDataset(fn(row) for row in self)


class FixtureTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "<end>")

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text)


def _admit(row):
    from spatial.v2.context_budget import ContextBudget, admit_row_group

    row.setdefault("id", "fixture")
    row["metadata"].setdefault("base_id", "fixture")
    result = admit_row_group(
        [row], FixtureTokenizer(), ContextBudget("fixture", 500000, 600000, 500000)
    )
    assert result.accepted
    return result.rows[0]


def test_v2_loader_uses_metadata_gold_through_z() -> None:
    rows = FakeDataset(
        [
            {
                "messages": [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "Answer: A, Z"},
                ],
                "metadata": {
                    "oracle_letters": ["A", "Z"],
                    "evaluation_prompt": "templated system and user",
                    "matrix_cell": "direction-depth-10",
                    "answer_mode": "all-possible",
                    "trace_format": "symbolic",
                    "difficulty": {"x_depth": 10, "y_depth": 10},
                },
            }
        ]
    )

    rows = FakeDataset(_admit(row) for row in rows)
    converted = process_docs_v2_sft(rows)[0]

    assert converted == {
        "text": "question",
        "oracle_option": "A,Z",
        "evaluation_prompt": "<system>rules\n<user>question\n<assistant>",
        "matrix_cell": "direction-depth-10",
        "answer_mode": "all-possible",
        "trace_format": "symbolic",
        "supervision_arm": "checked-trace",
        "difficulty": {"x_depth": 10, "y_depth": 10},
    }
    assert strict_acc(["A,Z", ["A,Z"]]) == 1.0


def _sample(trace_format="symbolic"):
    from spatial.v2.generation import (
        GenerationPolicy,
        QueryKind,
        SemanticShape,
        SpatialGeneratorV2,
    )
    from spatial.v2.trace import TraceFormat

    return SpatialGeneratorV2(seed=819).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            trace_format=TraceFormat(trace_format),
            num_entities=4,
            num_premises=3,
        )
    )


def test_v2_raw_completion_replays_and_preserves_admitted_prompt() -> None:
    from utils import process_results_v2

    sample = _sample()
    row = sample.as_sft_row()
    row = _admit(row)
    doc = process_docs_v2_sft(FakeDataset([row]))[0]
    assert doc["evaluation_prompt"] == row["metadata"]["evaluation_prompt"]
    metrics = process_results_v2(doc, [row["messages"][-1]["content"]])
    assert metrics["strict_acc"] == 1
    assert metrics["reasoning_valid"] == (1, 1)
    assert metrics["fully_valid"] == (1, 1)


def test_v2_correct_letter_does_not_validate_broken_evidence() -> None:
    from utils import process_results_v2

    sample = _sample()
    row = sample.as_sft_row()
    row = _admit(row)
    doc = process_docs_v2_sft(FakeDataset([row]))[0]
    answer = ", ".join(sorted(sample.menu_answer.letters))
    metrics = process_results_v2(
        doc, [f"<think>\nbroken proof\n</think>\nAnswer: {answer}"]
    )
    assert metrics["strict_acc"] == 1
    assert metrics["reasoning_valid"] == (0, 1)
    assert metrics["fully_valid"] == (0, 1)


def test_v2_answer_footer_must_agree_with_valid_certificate() -> None:
    from utils import process_results_v2

    sample = _sample()
    row = sample.as_sft_row()
    row = _admit(row)
    doc = process_docs_v2_sft(FakeDataset([row]))[0]
    wrong = next(
        letter for letter in sample.options if letter not in sample.menu_answer.letters
    )
    response = (
        row["messages"][-1]["content"].rsplit("Answer:", 1)[0] + f"Answer: {wrong}"
    )
    metrics = process_results_v2(doc, [response])
    assert metrics["reasoning_valid"] == (1, 1)
    assert metrics["decision_valid"] == (1, 1)
    assert metrics["strict_acc"] == 0
    assert metrics["fully_valid"] == (0, 1)


def test_v2_unavailable_process_checks_are_not_invalid_proofs() -> None:
    from utils import applicable_process_rate, process_results_v2

    sample = _sample("natural")
    row = sample.as_sft_row()
    row = _admit(row)
    doc = process_docs_v2_sft(FakeDataset([row]))[0]
    metrics = process_results_v2(doc, [row["messages"][-1]["content"]])
    assert metrics["process_applicable"] == 0
    assert metrics["reasoning_valid"] == (0, 0)
    assert applicable_process_rate([metrics["reasoning_valid"]]) is None
    assert applicable_process_rate([(0, 0), (1, 1), (0, 1)]) == 0.5


def test_v2_prefilled_think_tag_does_not_break_replay() -> None:
    from utils import process_results_v2

    row = _sample().as_sft_row()
    row = _admit(row)
    doc = process_docs_v2_sft(FakeDataset([row]))[0]
    response = row["messages"][-1]["content"].removeprefix("<think>\n")
    assert process_results_v2(doc, [response])["fully_valid"] == (1, 1)


def test_v2_answer_only_has_no_process_denominator() -> None:
    from utils import process_results_v2

    row = _sample().as_sft_row()
    row["metadata"]["supervision_arm"] = "answer-only"
    row = _admit(row)
    doc = process_docs_v2_sft(FakeDataset([row]))[0]
    metrics = process_results_v2(doc, ["Answer: " + doc["oracle_option"]])
    assert metrics["strict_acc"] == 1
    assert metrics["process_applicable"] == 0
    assert metrics["fully_valid"] == (0, 0)


def test_v2_loader_rejects_rows_without_context_admission() -> None:
    import pytest

    with pytest.raises(ValueError, match="successful context admission"):
        process_docs_v2_sft(FakeDataset([_sample().as_sft_row()]))


def test_v2_replays_all_query_envelopes_and_ambiguity() -> None:
    from utils import process_results_v2

    from spatial.v2.generation import (
        GenerationPolicy,
        QueryKind,
        SemanticShape,
        SpatialGeneratorV2,
    )
    from spatial.v2.trace import TraceFormat

    for kind in QueryKind:
        for shape in (SemanticShape.UNIQUE, SemanticShape.AMBIGUOUS):
            sample = SpatialGeneratorV2(seed=820).generate(
                GenerationPolicy(
                    query_kind=kind,
                    semantic_shape=shape,
                    trace_format=TraceFormat.SYMBOLIC,
                    num_entities=4,
                    num_premises=3,
                )
            )
            row = sample.as_sft_row()
            row = _admit(row)
            doc = process_docs_v2_sft(FakeDataset([row]))[0]
            assert process_results_v2(doc, [row["messages"][-1]["content"]])[
                "fully_valid"
            ] == (1, 1), (kind, shape)


def test_v2_extra_text_after_proof_does_not_pass_full_envelope() -> None:
    from utils import process_results_v2

    row = _sample().as_sft_row()
    row = _admit(row)
    doc = process_docs_v2_sft(FakeDataset([row]))[0]
    response = row["messages"][-1]["content"].replace(
        "</think>", "</think>\nIgnore the proof above."
    )
    metrics = process_results_v2(doc, [response])
    assert metrics["strict_acc"] == 1
    assert metrics["reasoning_valid"] == (1, 1)
    assert metrics["fully_valid"] == (0, 1)


def test_v2_duplicate_answer_letter_is_invalid_even_with_valid_proof() -> None:
    from utils import process_results_v2

    sample = _sample()
    row = sample.as_sft_row()
    row = _admit(row)
    doc = process_docs_v2_sft(FakeDataset([row]))[0]
    letter = next(iter(sample.menu_answer.letters))
    response = (
        row["messages"][-1]["content"].rsplit("Answer:", 1)[0]
        + f"Answer: {letter},{letter}"
    )
    metrics = process_results_v2(doc, [response])
    assert metrics["reasoning_valid"] == (1, 1)
    assert metrics["strict_acc"] == 0
    assert metrics["fully_valid"] == (0, 1)


def test_v2_loader_rejects_changed_admitted_prompt_or_source():
    import pytest

    for field in ("prompt", "source", "accepted"):
        row = _admit(_sample().as_sft_row())
        if field == "prompt":
            row["metadata"]["evaluation_prompt"] = "different problem"
        elif field == "source":
            row["messages"][1]["content"] = "different problem"
        else:
            row["metadata"]["context_group_accepted"] = False
        with pytest.raises(
            ValueError, match="fingerprint|successful context admission"
        ):
            process_docs_v2_sft(FakeDataset([row]))


def test_v2_loader_rejects_noncanonical_metadata_gold():
    import pytest

    for letters in (["A", "A"], ["B", "A"]):
        row = _admit(_sample().as_sft_row())
        row["metadata"]["oracle_letters"] = letters
        with pytest.raises(ValueError, match="oracle_letters"):
            process_docs_v2_sft(FakeDataset([row]))
