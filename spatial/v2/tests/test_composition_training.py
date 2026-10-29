"""Freeze, replay, independent ordering, and assistant-only mask safeguards."""

import json
from pathlib import Path

import pytest

from spatial.v2.composition_audit import aligned_pair
from spatial.v2.composition_contracts import parse_answer
from spatial.v2.composition_dataset import (
    candidate_stories,
    random_stories,
)
from spatial.v2.composition_evaluation import (
    adapter_hash,
    allowed_answers,
    replay_tokens,
    score,
    select_epoch,
)
from spatial.v2.composition_evidence import build_evidence, render_pair
from spatial.v2.composition_training import (
    MODEL,
    REVISION,
    audit_selection,
    completion_suffix,
    digest,
    prompt_text,
    seal,
    tokenize_example,
    training_config,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.tests.test_pivot_evidence import BY_ID


def test_random_candidate_budget_and_determinism():
    stories = candidate_stories()
    assert len(stories) == 200
    assert stories == candidate_stories()
    assert len({s.story_id for s in stories}) == 200
    assert sum(s.recipe == "random-world" for s in stories) == 100
    assert sum(s.recipe == "random-atoms" for s in stories) == 50
    assert random_stories(1, 2, 2) != random_stories(2, 2, 2)
    for s in stories:
        for p in s.problems():
            assert p.query.reference not in getattr(p.query, "candidates", ())
        assert all(len(a.allowed) == 1 for a in s.premise.operands)


def test_source_stratification_keeps_targeted_singletons_in_each_split():
    from spatial.v2.composition_pilot import assign_splits

    stories = candidate_stories()
    assignments = assign_splits(stories, 20261029, stratified=True)
    assert {assignments[s.story_id][1] for s in stories if s.recipe == "singleton"} == {
        "train",
        "dev",
        "test",
    }


def test_paired_alignment_rejects_missing_evidence():
    problem = BY_ID["D1"].problem()
    pair = render_pair(
        build_evidence(problem),
        ("West", "Northeast", "South", "Northwest"),
        AnswerMode.SINGLE,
        compact_unique_direction=True,
    )
    row = {"id": "test", "references": pair}
    aligned_pair(row)
    pair["natural"] = pair["natural"].replace("greater than", "less than", 1)
    with pytest.raises(ValueError, match="differ"):
        aligned_pair(row)


def test_compact_direction_removes_only_unneeded_exclusions():
    evidence = build_evidence(BY_ID["D1"].problem())
    menu = ("West", "Northeast", "South", "Northwest")
    original = render_pair(evidence, menu, AnswerMode.SINGLE)
    compact = render_pair(
        evidence, menu, AnswerMode.SINGLE, compact_unique_direction=True
    )
    for style in compact:
        assert len(compact[style]) < len(original[style])
        assert "contradiction" not in compact[style]
        assert parse_answer(compact[style], AnswerMode.SINGLE, stage=1) == "B"
    assert render_pair(evidence, menu, AnswerMode.ALL_POSSIBLE) == render_pair(
        evidence, menu, AnswerMode.ALL_POSSIBLE, compact_unique_direction=True
    )


@pytest.mark.parametrize(
    "contract,count", (("single", 6), ("visible-possible", 17), ("all-possible", 17))
)
def test_structured_decoder_supports_exact_contract_language(contract, count):
    choices = allowed_answers(contract)
    assert len(choices) == len(set(choices)) == count
    assert all(parse_answer(c, contract, stage=2) == c for c in choices)
    assert "AE" not in choices and "EF" not in choices


def test_stage2_replays_every_token_and_never_truncates():
    assert replay_tokens([1, 2], [3, 4, 5], [6]) == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="truncation"):
        replay_tokens([1] * 10, [2] * 20, [3], context=40)
    assert "<think>" in completion_suffix()
    assert completion_suffix(terminated=True).startswith("\n<|im_start|>user")
    assert completion_suffix().startswith("<|im_end|>\n")


def examples_and_outputs():
    rows = [
        {"id": mode, "contract": mode, "answer": "A"}
        for mode in ("single", "visible-possible", "all-possible")
    ]
    outputs = [
        {
            "id": r["id"],
            "stage1": {
                "text": "Reason.\nAnswer: A",
                "token_ids": [1, 2],
                "input_tokens": 10,
            },
            "stage2": {"text": "B", "token_ids": [3], "input_tokens": 20},
        }
        for r in rows
    ]
    return rows, outputs


def test_selection_uses_stage1_dev_only_with_earliest_tie():
    rows, outputs = examples_and_outputs()
    runs = [
        {
            "epoch": epoch,
            "split": "dev",
            "style": "natural",
            "manifest_sha256": "hash",
            "model": MODEL,
            "revision": REVISION,
            "adapter": f"checkpoint-{epoch}",
            "adapter_sha256": f"adapter-hash-{epoch}",
            "outputs": outputs,
        }
        for epoch in (2, 1)
    ]
    assert select_epoch(rows, runs, "natural", "hash")["epoch"] == 1
    result = score(rows, outputs)
    assert result["stage1"]["macro_accuracy"] == 1
    assert result["stage2"]["macro_accuracy"] == 0
    assert result["generated_tokens"] == {"stage1": 6, "stage2": 3, "total": 9}
    assert result["input_tokens"] == {"stage1": 30, "stage2": 60}
    assert result["regressions"] == 3
    runs[0]["split"] = "test"
    with pytest.raises(ValueError, match="provenance"):
        select_epoch(rows, runs, "natural", "hash")


def test_adapter_hash_detects_changed_weights(tmp_path):
    with pytest.raises(ValueError, match="weights"):
        adapter_hash(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}")
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"first")
    first = adapter_hash(tmp_path)
    weights.write_bytes(b"second")
    assert adapter_hash(tmp_path) != first


def test_scoring_reports_format_and_truncation_separately():
    rows, outputs = examples_and_outputs()
    outputs[0]["stage1"]["finish_reason"] = "length"
    outputs[0]["stage1"]["text"] = "unfinished"
    outputs[0]["stage2"]["text"] = "A"
    result = score(rows, outputs)
    assert result["stage1"]["format_valid"] == 2
    assert result["stage1"]["finish_reasons"]["length"] == 1
    assert result["transitions"]["length"]["rescues"] == 1
    assert result["transitions"]["completed"]["regressions"] == 2


@pytest.mark.parametrize("first", ("A", "B", None))
@pytest.mark.parametrize("second", ("A", "B", None))
@pytest.mark.parametrize("finish_reason", ("length", "stop", None))
def test_scoring_preserves_transitions_and_slices(first, second, finish_reason):
    rows, outputs = examples_and_outputs()
    for row, output in zip(rows, outputs, strict=True):
        row.update(query_kind="direction", semantic_case="unique", escape_reason=None)
        output["stage1"]["text"] = f"Answer: {first}" if first else "unfinished"
        output["stage2"]["text"] = second or "invalid"
        if finish_reason is not None:
            output["stage1"]["finish_reason"] = finish_reason
    result = score(rows, list(reversed(outputs)))
    for stage, answer in ((1, first), (2, second)):
        metrics = result[f"stage{stage}"]
        correct = 3 if answer == "A" else 0
        assert metrics["correct"] == correct
        assert metrics["format_valid"] == (3 if answer else 0)
        assert metrics["slices"]["query_kind"] == {
            "direction": {"correct": correct, "total": 3}
        }
        assert metrics["slices"]["escape_reason"] == {
            "None": {"correct": correct, "total": 3}
        }
        assert metrics["slices"]["recipe"] == {}
    rescues = 3 if first != "A" and second == "A" else 0
    regressions = 3 if first == "A" and second != "A" else 0
    assert result["rescues"] == rescues
    assert result["regressions"] == regressions
    status = "length" if finish_reason == "length" else "completed"
    assert result["transitions"][status] == {
        "total": 3,
        "answer_changes": 3 if first != second else 0,
        "rescues": rescues,
        "regressions": regressions,
    }
    other = "completed" if status == "length" else "length"
    assert result["transitions"][other] == dict.fromkeys(
        ("total", "answer_changes", "rescues", "regressions"), 0
    )


def test_missing_invalid_answers_and_incomplete_runs():
    rows, outputs = examples_and_outputs()
    outputs[0]["stage1"]["text"] = "No final answer"
    assert score(rows, outputs)["stage1"]["correct"] == 2
    with pytest.raises(ValueError, match="IDs"):
        score(rows, outputs[:-1])


def test_configs_match_except_style_paths():
    first = training_config("natural", "prepared")
    second = training_config("symbolic", "prepared")
    assert json.dumps(first).replace("natural", "STYLE") == json.dumps(second).replace(
        "symbolic", "STYLE"
    )
    assert first["datasets"][0]["type"] is None
    assert not first["load_best_model_at_end"]
    assert first["val_set_size"] == 0
    assert first["seed"] == first["data_seed"] == 42


def test_audit_requires_five_different_stories_not_five_questions():
    rows = [
        {
            "id": str(i),
            "story_id": "same",
            "base_id": "base",
            "semantic_case": "unique",
            "query_kind": "direction",
            "contract": "single",
            "recipe": "chain",
            "escape_reason": None,
        }
        for i in range(5)
    ]
    with pytest.raises(ValueError, match="five distinct"):
        audit_selection({"rows": rows, "bases": {"base": {}}})


def test_seal_rejects_missing_and_stale_pair_reviews(tmp_path, monkeypatch):
    import spatial.v2.composition_training as module

    monkeypatch.setattr(module, "verify_prepared", lambda output: {})
    selected = {"sample_ids": ["a"], "rows_sha256": {"a": "row-hash"}}
    (tmp_path / "audit-selection.json").write_text(json.dumps(selected))
    review = {
        "selection_sha256": digest(tmp_path / "audit-selection.json"),
        "reviewer": "test reviewer",
        "reviews": [],
        "unresolved_findings": [],
    }
    audit = tmp_path / "audit-review.json"
    audit.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="every selected"):
        seal(tmp_path, audit)
    review["reviews"] = [{"id": "a", "row_sha256": "wrong", "verdict": "pass"}]
    audit.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="stale"):
        seal(tmp_path, audit)


@pytest.mark.skipif(
    not Path.home()
    .joinpath(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots", REVISION)
    .exists(),
    reason="pinned local tokenizer unavailable",
)
def test_actual_pinned_template_mask_and_context():
    transformers = pytest.importorskip("transformers")
    from spatial.v2.composition_contracts import render_prompt

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots"
        / REVISION,
        local_files_only=True,
    )
    problem = BY_ID["D1"].problem()
    menu = ("West", "Northeast", "South", "Northwest")
    row = {
        "id": "d1",
        "split": "train",
        "contract": "single",
        "answer": "B",
        "prompt": render_prompt(problem, menu, AnswerMode.SINGLE),
        "references": render_pair(
            build_evidence(problem),
            menu,
            AnswerMode.SINGLE,
            compact_unique_direction=True,
        ),
    }
    for style in ("natural", "symbolic"):
        data, lengths = tokenize_example(tokenizer, row, style)
        start = lengths["prompt_tokens"]
        assert data["labels"][:start] == [-100] * start
        assert data["labels"][start:] == data["input_ids"][start:]
        assert lengths["stage2_reserved_tokens"] < 8192
        assert row["references"][style] not in prompt_text(tokenizer, row, style)
