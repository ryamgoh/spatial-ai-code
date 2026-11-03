"""Regression tests for final-v1 allocation, native exports and fail-closed runs."""

import ast
import os
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
import generation_policy as policy

from spatial.v2 import entailment_generation as shared

# Historical allocation baseline with the retired admission tokenizer removed.
POLICY_BASELINE_SHA = "8def543b64fcf3d634521ad81aa616b1f2729c821b4d09100a9ad155ea1e2f51"


@pytest.fixture(scope="module")
def api():
    return shared.GenerationWorkflow(policy.workflow_config())


@pytest.fixture(scope="module")
def tokens(api):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        policy.MODEL["model_id"],
        revision=policy.MODEL["revision"],
        local_files_only=True,
    )
    api.tokenizer = tokenizer
    return tokenizer


def test_single_pinned_tokenizer_snapshot_and_worker(tmp_path, monkeypatch):
    from unittest.mock import Mock

    download = Mock()
    tokenizer = object()
    load = Mock(return_value=tokenizer)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", load)
    monkeypatch.setattr(shared, "_WORKER", None)
    config = policy.workflow_config()
    workflow = shared.GenerationWorkflow(config)
    path = workflow.save_tokenizer(tmp_path)
    assert path == str(tmp_path / "tokenizers/active")
    download.assert_called_once()
    assert download.call_args.args == (policy.MODEL["model_id"],)
    assert download.call_args.kwargs["revision"] == policy.MODEL["revision"]
    assert download.call_args.kwargs["local_dir"] == Path(path)
    assert "historical_admission_tokenizer" not in config.policy
    workflow.initialize(path)
    shared.initialize_worker(config, path)
    assert workflow.tokenizer is tokenizer
    assert shared._WORKER.tokenizer is tokenizer
    assert load.call_count == 2
    load.assert_called_with(path, local_files_only=True)


def test_admission_uses_training_tokenizer(api, tokens, monkeypatch):
    build = shared.build_attempt
    seen = []

    def capture(story, problem, depth, tokenizer, **kwargs):
        seen.append(tokenizer)
        assert kwargs["measure_tokens"] == api.admission_tokens
        return build(story, problem, depth, tokenizer, **kwargs)

    monkeypatch.setattr(shared, "build_attempt", capture)
    slot = next(s for s in policy.schedule() if s.split == "train")
    result = api.generate_one(make_job(api, slot))
    assert result["status"] == "passed", result
    assert seen == [tokens]
    assert api.tokenizer is tokens


@pytest.mark.parametrize(
    "limits,reason",
    [
        ({"context_length": 1}, "Stage-2 context reserve exceeded"),
        ({"stage1_max_new_tokens": 1}, "supervised target exceeds Stage-1 cap"),
        ({"stage2_max_new_tokens": 8192}, "Stage-2 context reserve exceeded"),
    ],
)
def test_native_admission_rejects_invalid_budgets(
    api, tokens, limits, reason, monkeypatch, tmp_path
):
    import json
    from contextlib import nullcontext
    from types import SimpleNamespace

    config = replace(api.config, policy=deepcopy(api.config.policy))
    config.policy["model"].update(limits)
    workflow = shared.GenerationWorkflow(config)
    workflow.tokenizer = tokens
    slot = next(s for s in policy.schedule() if s.split == "train")
    job = make_job(api, slot)
    valid = api.generate_one(job)
    assert valid["status"] == "passed", valid
    row = valid["record"]["rows"][0]
    for style in ("natural", "symbolic"):
        measurement = workflow.admission_tokens(tokens, row, style)
        assert measurement["admitted"] is False
        assert reason in measurement["reason"]
    rejected = workflow.generate_one(job)
    assert rejected["status"] == "failed"
    assert "token-rejected" in rejected["reason"]
    assert reason in rejected["reason"]
    failures = rejected["token_rejections"]
    assert {t["style"] for t in failures} == {"natural", "symbolic"}
    assert all(t["id"] == row["id"] and reason in t["reason"] for t in failures)
    assert all(t["admitted"] is False for t in failures)
    # Exercise the real journal/receipt path with the already-computed rejection.
    monkeypatch.setattr(workflow, "source_inputs", lambda *_: [])
    monkeypatch.setattr(workflow, "save_tokenizer", lambda *_: "unused")
    monkeypatch.setattr(workflow, "initialize", lambda *_: None)
    monkeypatch.setattr(workflow, "preflight", lambda *_, **__: [job])
    pool = SimpleNamespace(map=lambda *_: iter([rejected]))
    monkeypatch.setattr(shared, "ProcessPoolExecutor", lambda **_: nullcontext(pool))
    run = tmp_path / "failed-run"
    with pytest.raises(ValueError, match="token-rejected"):
        workflow.generate(SimpleNamespace(run_dir=run, mode="full", workers=1))
    (entry,) = shared.read_jsonl(run / "audit/admissions.jsonl")
    assert entry["token_rejections"] == failures
    receipt = json.loads((run / "generation.json").read_text())
    assert receipt["status"] == "failed"
    assert reason in receipt["reason"]
    assert not (run / "manifest.json").exists()


def make_job(api, slot):
    _, problem, family, direction = api.construct(slot, 0)
    return {
        "slot": policy.asdict(slot),
        "alternative": 0,
        **shared.identities(problem),
        "anchor_key": family,
        "direction": direction.value,
    }


def test_depth_two_zero_count_preserves_admission_dependency(api, tokens):
    slot = next(
        s
        for s in policy.schedule()
        if (s.split, s.kind, s.level, s.case, s.index)
        == ("in-range-test", "count", 2, "A-D", 35)
    )
    assert slot.count_value == 0
    result = api.generate_one(make_job(api, slot))
    assert result["status"] == "passed", result
    assert result["record"]["measurement"]["depth"] == 2
    assert result["record"]["semantics"]["entailed"] == [0]


def test_full_schedule_exact_and_deterministic():
    slots = policy.schedule()
    assert slots == policy.schedule()
    assert len(slots) == 10800
    assert Counter(s.split for s in slots) == policy.SPLITS
    assert policy.ALTERNATIVES == 5
    assert (
        len({s.identity(a) for s in slots for a in range(policy.ALTERNATIVES)}) == 54000
    )
    for split, total in policy.SPLITS.items():
        for kind in policy.KINDS:
            rows = [s for s in slots if (s.split, s.kind) == (split, kind)]
            assert Counter(s.case for s in rows) == policy.case_counts(kind, total // 3)
            for level in policy.levels(split):
                cell = [s for s in rows if s.level == level]
                assert sum(s.case == "A-D" for s in cell) == len(cell) * 6 // 10
                assert sum(s.case == "F" for s in cell) == len(cell) // 10
                assert all(
                    sum(s.case == c for s in cell) > 0
                    for c in policy.case_counts(kind, total // 3)
                )
                if kind == "count":
                    for case in ("A-D", "F"):
                        counts = Counter(s.count_value for s in cell if s.case == case)
                        assert set(counts) == set(range(5))
                        assert max(counts.values()) - min(counts.values()) <= 1
                if kind == "which":
                    assert any(s.contingent for s in cell)


def test_schedule_rejects_missing_or_duplicated_slot():
    slots = policy.schedule()
    with pytest.raises(ValueError):
        policy.validate_schedule(slots[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        policy.validate_schedule([*slots, slots[0]])


def test_no_fake_semantic_depth():
    for s in policy.schedule():
        if s.case in ("ambiguous", "inconsistent") or s.contingent:
            assert s.requested_depth is None


def planned_logic_slots():
    """Exercise actual scheduled seeds in every split/level/semantic cell."""
    selected = {}
    for slot in policy.schedule():
        key = (
            slot.split,
            slot.kind,
            slot.level,
            slot.case,
            slot.cohort,
            slot.contingent,
            slot.count_value
            if slot.kind == "count" and slot.case in ("A-D", "F")
            else None,
        )
        selected.setdefault(key, slot)
    result = list(selected.values())
    directions = {(s.split, s.kind, s.seed(0) % 8) for s in result}
    for slot in policy.schedule():
        key = (slot.split, slot.kind, slot.seed(0) % 8)
        if key not in directions:
            result.append(slot)
            directions.add(key)
    return tuple(result)


def test_logic_matrix_covers_all_directions_per_split_and_query():
    assert {(s.split, s.kind, s.seed(0) % 8) for s in planned_logic_slots()} == {
        (split, kind, direction)
        for split in policy.SPLITS
        for kind in policy.KINDS
        for direction in range(8)
    }


@pytest.mark.parametrize(
    "slot", planned_logic_slots(), ids=lambda slot: slot.identity(0)
)
def test_every_planned_logic_cell_passes_real_admission(api, tokens, slot):
    result = api.generate_one(make_job(api, slot))
    assert result["status"] == "passed", result
    record = result["record"]
    assert len(record["rows"]) == 1
    row = record["rows"][0]
    problem, menu, contract = shared.parse_row(row)
    evidence = shared.load_evidence(record)
    semantics = evidence.semantics

    assert contract.value == "unique-entailed"
    assert len(menu) == len(set(menu)) == 4
    assert semantics.consistent == (slot.case != "inconsistent")
    assert row["construction_level"] == slot.level
    assert row["verified_depth"] == slot.requested_depth
    assert record["independent_audit"] == "passed"
    assert api.sample_size(slot, 0) == (
        len(problem.objects),
        len(problem.premise.operands),
    )
    assert record["requested_size"] == {
        "objects": len(problem.objects),
        "premises": len(problem.premise.operands),
    }
    assert record["semantic_complexity_status"] == (
        "verified-semantics-no-depth-claim"
        if slot.cohort == "semantic"
        else "verified-answer-support"
    )
    if slot.kind != "direction":
        assert set(problem.query.candidates) == (
            set(problem.objects) - {problem.query.reference}
        )
    if slot.kind == "count":
        assert set(menu) <= set(range(5))
        assert len(problem.query.candidates) > 4
        # Both previously investigated <=4 rules now reduce to a fixed choice.
        assert next(i for i, value in enumerate(menu) if value <= 4) == 0
    if slot.case in ("A-D", "F"):
        assert len(semantics.entailed) == 1
        value = next(iter(semantics.entailed))
        if slot.kind == "count":
            assert value == slot.count_value
        if slot.case == "F":
            assert value not in menu
            assert row["answer"] == "F"
        else:
            assert menu["ABCD".index(row["answer"])] == value
            assert row["answer"] == slot.answer_position
        assert row["selected_explanation_depth"] == slot.level
    else:
        assert row["answer"] == "E"
        if slot.case == "multiple":
            assert slot.kind == "which"
            assert len(semantics.entailed) == 2
            assert row["selected_explanation_depth"] == slot.level
        else:
            assert not semantics.entailed
            assert row["selected_explanation_depth"] is None
    if slot.contingent:
        assert row["sole_entailed_with_contingent"]
        assert row["verified_depth"] is None
    assert set(result["tokenized"]) == (
        {"natural", "symbolic"} if slot.split in ("train", "dev") else set()
    )
    assert all(t["stage2_reserved_tokens"] <= 8192 for t in record["tokens"])
    assert all(
        text.endswith(f"Answer: {row['answer']}") for text in row["references"].values()
    )


@pytest.mark.parametrize("contingent", (False, True))
def test_semantic_positive_cannot_claim_a_deeper_construction_level(
    api, tokens, monkeypatch, contingent
):
    slot = next(
        s
        for s in policy.schedule()
        if s.kind == "which"
        and s.case == "A-D"
        and s.cohort == "semantic"
        and s.level == 8
        and s.contingent == contingent
    )
    original = api.construct

    def shallow_construction(requested, alternative):
        return original(replace(requested, level=1), alternative)

    monkeypatch.setattr(api, "construct", shallow_construction)
    result = api.generate_one(make_job(api, slot))
    assert result["status"] == "failed"
    assert (
        "selected positive support differs from construction level" in result["reason"]
    )


def test_release_checks_do_not_require_semantic_difficulty_scaling(api):
    assert api.config.pending_release_checks == (
        "nuisance-shortcut-probes-and-paired-example-review",
        "actual-axolotl-loader-collator-and-gpu-validation",
        "final-v1-training-and-epoch-evaluation-integration",
    )


def test_smoke_is_not_a_full_release():
    chosen = policy.smoke_schedule(policy.schedule())
    assert 30 < len(chosen) < 10800
    assert {s.level for s in chosen} == {1, 8, 9, 10}
    assert {s.split for s in chosen} == set(policy.SPLITS)
    assert {s.case for s in chosen} == {
        "A-D",
        "F",
        "ambiguous",
        "multiple",
        "inconsistent",
    }


def test_inventory_canonical_exclusions(api):
    inventory = shared.Inventory()
    keys = {"query_key": "q", "scene_key": "scene"}
    inventory.add(keys, "dev", "family")
    assert inventory.conflict(keys, "train") == "duplicate-query"
    assert (
        inventory.conflict({"query_key": "q2", "scene_key": "scene"}, "train")
        == "cross-split-scene"
    )
    assert (
        inventory.conflict({"query_key": "q2", "scene_key": "s2"}, "train", "family")
        == "cross-split-family"
    )


@pytest.mark.parametrize(
    "left,right",
    [
        (
            "final-v1-train-direction-l01-A-D-0004-a0",
            "final-v1-in-range-test-direction-l01-F-0003-a0",
        ),
        (
            "final-v1-dev-direction-l01-A-D-0000-a0",
            "final-v1-in-range-test-which-l01-A-D-0013-a0",
        ),
    ],
)
def test_recorded_l1_collisions_have_distinct_contexts(api, left, right):
    slots = {s.identity(0): s for s in policy.schedule()}
    a = api.construct(slots[left], 0)[1]
    b = api.construct(slots[right], 0)[1]
    assert shared.identities(a)["scene_key"] != shared.identities(b)["scene_key"]


@pytest.fixture
def reversed_construction_pool(api, monkeypatch):
    """Complete queued work backwards, but expose failures only on result()."""
    from concurrent.futures import Future

    pools = []

    class Pending(Future):
        def result(self):
            for future, fn, args in reversed(self.pool.calls):
                if not future.done():
                    try:
                        future.set_result(fn(*args))
                    except ValueError as exc:
                        future.set_exception(exc)
                    self.pool.finished.append(args)
            self.pool.unconsumed -= 1
            return super().result()

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"
            self.calls = []
            self.finished = []
            self.unconsumed = self.maximum = 0
            self.closed = False
            pools.append(self)

        def submit(self, fn, *args):
            future = Pending()
            future.pool = self
            self.calls.append((future, fn, args))
            self.unconsumed += 1
            self.maximum = max(self.maximum, self.unconsumed)
            return future

        def shutdown(self, *, wait, cancel_futures):
            assert wait and cancel_futures
            self.closed = True
            for future, _, _ in self.calls:
                if not future.done():
                    future.cancel()

    monkeypatch.setattr(shared, "_WORKER", api)
    monkeypatch.setattr(shared, "ProcessPoolExecutor", Pool)
    return pools


@pytest.mark.parametrize("workers", (1, 3))
def test_five_attempts_then_fail_no_refill(
    api, monkeypatch, tmp_path, reversed_construction_pool, workers
):
    (tmp_path / "audit").mkdir()
    slot = policy.schedule()[0]
    calls = []
    monkeypatch.setattr(
        api, "construct", lambda s, a: (None, object(), "fam", shared.Direction.NORTH)
    )
    monkeypatch.setattr(
        shared, "identities", lambda p: {"query_key": "same", "scene_key": "same"}
    )
    inventory = shared.Inventory()
    inventory.add({"query_key": "same", "scene_key": "same"}, "external-test")
    original = inventory.conflict

    def conflict(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(inventory, "conflict", conflict)
    with pytest.raises(ValueError, match="no refill"):
        api.preflight([slot], inventory, tmp_path, workers=workers)
    assert len(calls) == 5
    assert len(shared.read_jsonl(tmp_path / "audit/constructions.jsonl")) == 5


@pytest.mark.parametrize("success_alternative", (3, 4))
@pytest.mark.parametrize("workers", (1, 3))
def test_later_retry_selects_only_a_nonconflicting_candidate(
    api, monkeypatch, tmp_path, success_alternative, reversed_construction_pool, workers
):
    (tmp_path / "audit").mkdir()
    slot = policy.schedule()[0]
    calls = []

    def construct(s, alternative):
        calls.append(alternative)
        return None, alternative, "family", shared.Direction.NORTH

    monkeypatch.setattr(api, "construct", construct)
    monkeypatch.setattr(
        shared,
        "identities",
        lambda a: {
            "query_key": "new" if a == success_alternative else "same",
            "scene_key": "new" if a == success_alternative else "same",
        },
    )
    inventory = shared.Inventory()
    inventory.add({"query_key": "same", "scene_key": "same"}, "external-test")
    jobs = api.preflight([slot], inventory, tmp_path, workers=workers)
    assert calls == list(range(success_alternative + 1))
    assert len(jobs) == 1
    assert jobs[0]["alternative"] == success_alternative
    assert jobs[0]["query_key"] == "new"
    assert len(inventory.query_splits) == 2


@pytest.mark.parametrize(
    "split", ("train", "dev", "in-range-test", "extrapolation-test")
)
@pytest.mark.parametrize("workers", (1, 3))
def test_duplicate_fallback_is_fifth_attempt_and_training_only(
    api, monkeypatch, tmp_path, split, reversed_construction_pool, workers
):
    (tmp_path / "audit").mkdir()
    slot = next(s for s in policy.schedule() if s.split == split)
    keys = {"query_key": "same", "scene_key": "scene"}
    inventory = shared.Inventory()
    inventory.add(keys, split, "family")
    calls = []

    def construct(s, alternative):
        calls.append(alternative)
        return None, object(), "family", shared.Direction.NORTH

    monkeypatch.setattr(api, "construct", construct)
    monkeypatch.setattr(shared, "identities", lambda p: keys)
    if split == "train":
        jobs = api.preflight([slot], inventory, tmp_path, workers=workers)
        assert jobs[0]["training_duplicate"] is True
        assert jobs[0]["alternative"] == 4
        assert len(inventory.query_splits) == 1
    else:
        with pytest.raises(ValueError, match="no refill"):
            api.preflight([slot], inventory, tmp_path, workers=workers)
    assert calls == list(range(5))


@pytest.mark.parametrize("failure_index", (None, 2))
def test_parallel_preflight_is_ordered_bounded_and_fail_closed(
    api, monkeypatch, tmp_path, reversed_construction_pool, failure_index
):
    slots = policy.schedule()[:7]

    def construct(slot, alternative):
        if slot.index == failure_index:
            raise ValueError("construction failed")
        return (
            None,
            (slot.index, alternative),
            f"family-{slot.index}",
            shared.Direction.NORTH,
        )

    def keys(pair):
        index, alternative = pair
        return {
            "query_key": "shared" if index < 2 and alternative == 0 else str(pair),
            "scene_key": f"scene-{pair}",
        }

    monkeypatch.setattr(api, "construct", construct)
    monkeypatch.setattr(shared, "identities", keys)
    outcomes = []
    for workers in (1, 3):
        run = tmp_path / str(workers)
        (run / "audit").mkdir(parents=True)
        inventory = shared.Inventory()
        if failure_index is None:
            jobs = api.preflight(slots, inventory, run, workers=workers)
            assert [j["alternative"] for j in jobs] == [0, 1, 0, 0, 0, 0, 0]
        else:
            with pytest.raises(ValueError, match="construction failed"):
                api.preflight(slots, inventory, run, workers=workers)
            jobs = None
            assert not (run / "audit/preflight.json").exists()
        outcomes.append(
            (jobs, (run / "audit/constructions.jsonl").read_bytes(), vars(inventory))
        )
    assert outcomes[0] == outcomes[1]
    (pool,) = reversed_construction_pool
    assert pool.maximum <= 3
    assert pool.closed
    assert pool.finished[0][0] == slots[2]
    assert [(s.index, a) for _, _, (s, a) in pool.calls if a] == [(1, 1)]


@pytest.mark.parametrize("workers", (2, 4))
def test_real_parallel_preflight_matches_serial(api, tmp_path, workers):
    selected = {}
    for slot in policy.schedule():
        if slot.split == "in-range-test" and slot.level == 1:
            selected.setdefault((slot.kind, slot.case), slot)
        if slot.split == "extrapolation-test" and slot.level == 10:
            selected.setdefault((slot.kind, "deep"), slot)
    slots = list(selected.values())
    blocked = api.construct_candidate(slots[0], 0)
    runs = []
    for count in (1, workers):
        run = tmp_path / str(count)
        (run / "audit").mkdir(parents=True)
        inventory = shared.Inventory()
        inventory.add(
            {k: blocked[k] for k in ("query_key", "scene_key")}, "external-test"
        )
        jobs = api.preflight(slots, inventory, run, workers=count)
        assert jobs[0]["alternative"] == 1
        runs.append(
            (
                jobs,
                (run / "audit/constructions.jsonl").read_bytes(),
                (run / "audit/preflight.json").read_bytes(),
                vars(inventory),
            )
        )
    assert runs[0] == runs[1]


def test_construction_worker_needs_no_tokenizer(api, monkeypatch):
    monkeypatch.setattr(shared, "_WORKER", None)
    monkeypatch.setattr(
        shared.GenerationWorkflow,
        "initialize",
        lambda *_: pytest.fail("preflight must not load a tokenizer"),
    )
    shared.initialize_worker(api.config)
    assert shared._WORKER.tokenizer is None
    slot = policy.schedule()[0]
    assert shared.construct_worker(slot, 0) == api.construct_candidate(slot, 0)


@pytest.mark.parametrize("overlap", ("query", "scene", "family"))
def test_training_duplicate_cannot_hide_heldout_overlap(overlap):
    inventory = shared.Inventory()
    inventory.add({"query_key": "train-q", "scene_key": "train-s"}, "train", "train-f")
    inventory.add({"query_key": "held-q", "scene_key": "held-s"}, "dev", "held-f")
    keys = {
        "query_key": "held-q" if overlap == "query" else "train-q",
        "scene_key": "held-s" if overlap == "scene" else "train-s",
    }
    family = "held-f" if overlap == "family" else "train-f"
    assert not inventory.training_duplicate_allowed(keys, "train", family)
    with pytest.raises(ValueError, match="inventory conflict"):
        inventory.add(keys, "train", family, allow_training_duplicate=True)
    assert inventory.query_splits["held-q"] == "dev"


def test_training_duplicate_flag_survives_admission_and_inventory_audit(api, tokens):
    first = next(
        s
        for s in policy.schedule()
        if s.split == "train" and s.kind == "direction" and s.case == "A-D"
    )
    original = api.generate_one(make_job(api, first))
    assert original["status"] == "passed", original
    second = next(
        s
        for s in policy.schedule()
        if s.split == first.split
        and s.kind == first.kind
        and s.level == first.level
        and s.case == first.case
        and s != first
        and api.sample_size(s, 4) == api.sample_size(first, 0)
    )
    job = make_job(api, second)
    _, problem, family, direction = api.construct(second, 4)
    job.update(
        alternative=4,
        training_duplicate=True,
        **shared.identities(problem),
        anchor_key=family,
        direction=direction.value,
    )
    generated = api.generate_one(job)
    assert generated["status"] == "passed", generated
    assert generated["record"]["training_duplicate"] is True
    # Audit must reject a claimed repeat that does not match an earlier query.
    with pytest.raises(ValueError, match="invalid training duplicate"):
        api.audit_inventory(
            [original["record"], generated["record"]], [first, second], "smoke"
        )
    duplicate = deepcopy(original["record"])
    duplicate.update(slot=policy.asdict(second), alternative=4, training_duplicate=True)
    duplicate["rows"][0]["base_id"] = f"{second.identity(4)}-{second.kind}"
    summary = api.audit_inventory(
        [original["record"], duplicate], [first, second], "smoke"
    )
    assert summary["training_duplicates_accepted"] == 1
    assert summary["canonical_query_frequencies"]["train"] == {
        original["record"]["query_key"]: 2
    }
    duplicate["alternative"] = 3
    duplicate["rows"][0]["base_id"] = f"{second.identity(3)}-{second.kind}"
    with pytest.raises(ValueError, match="invalid training duplicate"):
        api.audit_inventory([original["record"], duplicate], [first, second], "smoke")


@pytest.mark.parametrize(
    "kind,case,level,contingent",
    [
        ("direction", "A-D", 1, False),
        ("direction", "F", 8, False),
        ("direction", "ambiguous", 10, False),
        ("direction", "inconsistent", 9, False),
        ("which", "A-D", 8, True),
        ("which", "F", 10, False),
        ("which", "multiple", 9, False),
        ("which", "ambiguous", 1, False),
        ("which", "inconsistent", 10, False),
        ("count", "A-D", 10, False),
        ("count", "F", 8, False),
        ("count", "ambiguous", 9, False),
        ("count", "inconsistent", 1, False),
    ],
)
def test_generated_cases_audit_and_native_tokens(
    api, tokens, kind, case, level, contingent
):
    slot = next(
        s
        for s in policy.schedule()
        if (s.kind, s.case, s.level, s.contingent) == (kind, case, level, contingent)
        and (kind != "count" or s.count_value == 0)
    )
    result = api.generate_one(make_job(api, slot))
    assert result["status"] == "passed", result
    record = result["record"]
    assert len(record["rows"]) == 1
    row = record["rows"][0]
    assert row["split"] == slot.split
    assert row["answer"] == (
        slot.answer_position if case == "A-D" else "F" if case == "F" else "E"
    )
    assert row["verified_depth"] == slot.requested_depth
    assert record["independent_audit"] == "passed"
    assert all(t["stage2_reserved_tokens"] <= 8192 for t in record["tokens"])
    if slot.split in ("train", "dev"):
        for tok in result["tokenized"].values():
            assert len(tok["input_ids"]) == len(tok["labels"])
            assert -100 in tok["labels"]
            decoded = tokens.decode([v for v in tok["labels"] if v != -100])
            assert "</think>" in decoded and f"Answer: {row['answer']}" in decoded
    mutated = deepcopy(record)
    mutated["rows"][0]["answer"] = "A" if row["answer"] != "A" else "B"
    with pytest.raises(ValueError):
        shared.audit_record(mutated)


def test_external_preserves_source_options_and_source_provenance(api, tokens):
    sources = shared.verify_source_export(
        ROOT / "data/spatialeval_org.jsonl",
        ROOT
        / "experiments/spatialentail-randomized-v2/prepared-corr-eval/data/corr.jsonl",
        ROOT
        / "experiments/spatialentail-randomized-v2/prepared-corr-eval/correction-summary.json",
    )
    assert len(sources) == 1500
    for suffix in (".0", ".1", ".2"):
        source = next(s for s in sources if s["id"].endswith(suffix))
        original = deepcopy(source)
        item = api.external_one(source)
        row = item["row"]
        assert source == original
        assert row["base_id"] == source["id"]
        assert row["dataset"] == "SpatialMap-TQA-Corr"
        assert row["menu_origin"] == "source-A-D"
        parsed = shared.SpatialTextAdapter().parse(source["text"])
        visible = [row.get("display_names", {}).get(v, str(v)) for v in row["menu"]]
        assert visible == [parsed.options[c] for c in "ABCD"]
        shared.verify_view(row)


def test_train_and_dev_disk_exports_are_native_and_gold_free(api, tokens, tmp_path):
    from contextlib import ExitStack

    for split in ("train", "dev"):
        slot = next(
            s
            for s in policy.schedule()
            if s.split == split and s.kind == "direction" and s.case == "A-D"
        )
        result = api.generate_one(make_job(api, slot))
        assert result["status"] == "passed", result
        assert set(result["tokenized"]) == {"natural", "symbolic"}
        with ExitStack() as stack:
            handles = shared.open_split(stack, tmp_path, split)
            shared.write_synthetic(handles, result)
    assert api.audit_exports(tmp_path, {"train": 1, "dev": 1})["status"] == "passed"
    path = tmp_path / "data/dev/natural-inputs.jsonl"
    rows = shared.read_jsonl(path)
    rows[0]["gold"] = "A"
    with path.open("w") as stream:
        shared.append_json(stream, rows[0])
    with pytest.raises(ValueError, match="gold/audit"):
        api.audit_exports(tmp_path, {"train": 1, "dev": 1})


@pytest.mark.parametrize("corruption", ("subgroup", "identity", "gold"))
def test_combined_benchmark_exports_preserve_subgroups(
    api, tokens, tmp_path, corruption
):
    from contextlib import ExitStack

    subsets = {"in-range-test": 1, "extrapolation-test": 1}
    results = []
    for split in subsets:
        slot = next(s for s in api.config.smoke_slots if s.split == split)
        result = api.generate_one(make_job(api, slot))
        assert result["status"] == "passed", result
        results.append(result)
    with ExitStack() as stack:
        handles = shared.open_split(stack, tmp_path, "sft-bench")
        for result in results:
            shared.write_synthetic(handles, result)
    directory = tmp_path / "data/sft-bench"
    rows = shared.read_jsonl(directory / "references.jsonl")
    assert rows == [r["record"]["rows"][0] for r in results]
    assert Counter(r["split"] for r in rows) == subsets
    assert {p.name for p in (tmp_path / "data").iterdir()} == {"sft-bench"}
    assert not (directory / "natural-tokenized.jsonl").exists()
    audit = api.audit_exports(tmp_path, subsets)
    assert audit["split_counts"] == {"sft-bench": 2}
    assert audit["generation_split_counts"] == subsets
    path = directory / "references.jsonl"
    if corruption == "subgroup":
        rows[0]["split"] = "train"
        message = "subgroup counts"
    elif corruption == "identity":
        rows[0]["id"] = rows[1]["id"]
        message = "count/identity"
    else:
        path = directory / "natural-inputs.jsonl"
        rows = shared.read_jsonl(path)
        rows[0]["answer"] = "A"
        message = "gold/audit"
    with path.open("w") as stream:
        for row in rows:
            shared.append_json(stream, row)
    with pytest.raises(ValueError, match=message):
        api.audit_exports(tmp_path, subsets)


def test_benchmark_export_mapping_preserves_counts(api):
    mapping = api.config.policy["export_splits"]
    counts = Counter()
    for slot in policy.schedule():
        counts[mapping.get(slot.split, slot.split)] += 1
    assert counts == {"train": 6000, "dev": 1200, "sft-bench": 3600}


def test_unmapped_benchmark_exports_keep_historical_layout(api, tokens, tmp_path):
    from contextlib import ExitStack

    historical_policy = {
        k: v for k, v in api.config.policy.items() if k != "export_splits"
    }
    historical = shared.GenerationWorkflow(
        replace(api.config, policy=historical_policy)
    )
    counts = {"in-range-test": 1, "extrapolation-test": 1}
    for split in counts:
        slot = next(s for s in api.config.smoke_slots if s.split == split)
        result = api.generate_one(make_job(api, slot))
        assert result["status"] == "passed", result
        with ExitStack() as stack:
            shared.write_synthetic(shared.open_split(stack, tmp_path, split), result)
    assert historical.audit_exports(tmp_path, counts)["split_counts"] == counts


def minimal_manifest(api, tmp_path):
    for name in ("inputs", "data", "audit", "tokenizers"):
        (tmp_path / name).mkdir()
    (tmp_path / "data/payload.jsonl").write_text("{}\n")
    manifest = {
        "schema": "spatialentail-final-candidate-v1",
        "status": "candidate-audited",
        "training_ready": False,
        "pending_release_checks": list(api.config.pending_release_checks),
        "payload_sha256": shared.payload_hashes(tmp_path),
        "source_sha256": api.fingerprints(),
    }
    shared.atomic_json(tmp_path / "manifest.json", manifest)
    shared.atomic_json(
        tmp_path / "generation.json",
        {
            "status": manifest["status"],
            "manifest_sha256": shared.digest(tmp_path / "manifest.json"),
        },
    )
    return manifest


def test_candidate_verification_never_authorizes_training(api, tmp_path):
    minimal_manifest(api, tmp_path)
    assert not api.verify(tmp_path)["training_ready"]
    with pytest.raises(ValueError, match="not a training release"):
        api.verify(tmp_path, require_training_ready=True)
    (tmp_path / "data/payload.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="payload changed"):
        api.verify(tmp_path)


def test_failed_manifest_and_extra_payload_rejected(api, tmp_path):
    manifest = minimal_manifest(api, tmp_path)
    (tmp_path / "data/unexpected.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="inventory changed"):
        api.verify(tmp_path)
    manifest["status"] = "failed"
    shared.atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="did not finish"):
        api.verify(tmp_path)


@pytest.mark.parametrize("script", ["generate.sh", "train.sh", "evaluate.sh"])
def test_workers_email_on_start_end_failure_and_time_limit(script):
    worker = (EXP / "slurm" / script).read_text()
    assert "#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT" in worker.splitlines()
    assert "#SBATCH --mail-user=e0958116@u.nus.edu" in worker.splitlines()


@pytest.mark.parametrize(
    "script,partition,gpu,time",
    [
        ("evaluate.sh", "gpu", "h100-47", "03:00:00"),
        ("train.sh", "gpu-long", "h100-47", "24:00:00"),
    ],
)
def test_gpu_workers_match_cluster_partition_limits(script, partition, gpu, time):
    lines = (EXP / "slurm" / script).read_text().splitlines()
    assert f"#SBATCH --partition={partition}" in lines
    assert f"#SBATCH --gres=gpu:{gpu}:1" in lines
    assert f"#SBATCH --time={time}" in lines


def test_evaluation_preflight_uses_cpu_script_environment():
    launch = (EXP / "slurm/submit-evaluation.sh").read_text()
    assert '--script "$EXP/generate.py" verify-evaluation' in launch
    assert "--native-tls" in launch and "--no-project" in launch
    assert '--project "$ROOT"' not in launch
    worker = (EXP / "slurm/evaluate.sh").read_text()
    assert '--project "$ROOT" --frozen --no-sync' in worker


def test_cpu_evaluation_command_delegates_without_generating(tmp_path, monkeypatch):
    from spatial.v2 import entailment_final_evaluation as final

    calls = []
    monkeypatch.setattr(final, "verify_campaign", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(
        shared.GenerationWorkflow,
        "generate",
        lambda *a, **k: pytest.fail("verification must not generate data"),
    )
    shared.main(
        policy.workflow_config(),
        [
            "verify-evaluation",
            "--run",
            str(tmp_path),
            "--training-revision",
            "a" * 40,
        ],
    )
    assert calls == [((tmp_path.resolve(),), {"training_revision": "a" * 40})]


def test_launcher_is_cpu_generation_only():
    launch = (EXP / "slurm/submit-generation.sh").read_text()
    worker = (EXP / "slurm/generate.sh").read_text()
    assert launch.count("sbatch --parsable") == 1
    assert "--dependency" not in launch
    assert "--partition=normal" in worker
    assert "#SBATCH --cpus-per-task=128" in worker
    assert "#SBATCH --mem=128G" in worker
    assert "--gres" not in worker
    assert "uv sync" not in worker
    assert worker.count('--script "$FINAL_EXPERIMENT_DIR/generate.py"') == 2
    assert worker.count("--no-project") == 2
    assert "UV_PROJECT_ENVIRONMENT" not in worker
    assert "--with-requirements" not in worker
    for path in (EXP / "slurm").glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize(
    ("cpus", "override", "expected"),
    [("128", "", "128"), ("16", "", "16"), ("128", "8", "8"), ("128", "129", None)],
)
def test_worker_count_follows_allocation(cpus, override, expected):
    script = (EXP / "slurm/generate.sh").read_text()
    start = script.index("WORKERS=")
    end = script.index('echo "Generation only:')
    env = {
        **os.environ,
        "SLURM_CPUS_PER_TASK": cpus,
        "FINAL_GENERATION_WORKERS": override,
    }
    result = subprocess.run(
        ["bash", "-c", script[start:end] + '\nprintf "%s" "$WORKERS"'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if expected is None:
        assert result.returncode == 2
        assert "Workers must fit" in result.stderr
    else:
        assert result.returncode == 0
        assert result.stdout == expected


def test_generation_worker_uses_backward_compatible_tls_flag(tmp_path):
    bins = tmp_path / "bin"
    bins.mkdir()
    capture = tmp_path / "calls"
    for name, body in (
        ("srun", '#!/bin/bash\nexec "$@"\n'),
        (
            "uv",
            (
                "#!/bin/bash\n"
                'if [[ "$1" != "--native-tls" ]]; then\n'
                "  echo \"error: unexpected argument '$1' found\" >&2\n"
                "  exit 2\n"
                "fi\n"
                'printf "%s\\n" "$*" >> "$CAPTURE"\n'
            ),
        ),
    ):
        path = bins / name
        path.write_text(body)
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", str(EXP / "slurm/generate.sh")],
        env={
            **os.environ,
            "PATH": f"{bins}:{os.environ['PATH']}",
            "CAPTURE": str(capture),
            "FINAL_RUN_DIR": str(tmp_path / "artifacts"),
            "FINAL_EXPERIMENT_DIR": str(EXP),
            "FINAL_GENERATION_MODE": "full",
            "FINAL_GENERATION_WORKERS": "",
            "SLURM_JOB_ID": "123",
            "SLURM_TRES_PER_TASK": "cpu=128",
            "SLURM_SRUN_CPUS_PINNED": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = capture.read_text().splitlines()
    assert len(calls) == 2
    assert all(
        line.startswith("--native-tls run --python 3.12 --no-project") for line in calls
    )
    assert " build " in calls[0] and " verify " in calls[1]
    assert "--workers 128" in calls[0]


def test_submitter_records_one_job_and_paths(tmp_path):
    root = tmp_path / "repo with spaces"
    exp = root / "experiments/spatialentail-final-v1"
    scripts = exp / "slurm"
    scripts.mkdir(parents=True)
    for name in ("submit-generation.sh", "generate.sh"):
        (scripts / name).write_text((EXP / "slurm" / name).read_text())
    for name in (
        "data/spatialeval_org.jsonl",
        "experiments/spatialentail-randomized-v2/prepared-corr-eval/data/corr.jsonl",
        "experiments/spatialentail-randomized-v2/prepared-corr-eval/correction-summary.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    bins = tmp_path / "bin"
    bins.mkdir()
    capture = tmp_path / "calls"
    for name, body in (
        ("uv", "#!/bin/bash\nexit 0\n"),
        (
            "sbatch",
            '#!/bin/bash\nprintf "%s\\n" "$@" >> "$CAPTURE"\necho "12345;cluster"\n',
        ),
    ):
        path = bins / name
        path.write_text(body)
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bins}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
    }
    result = subprocess.run(
        ["bash", str(scripts / "submit-generation.sh"), "smoke"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    (run,) = (exp / "runs").iterdir()
    assert "generation_job=12345" in (run / "submission.txt").read_text()
    assert f"run_dir={run}/artifacts" in (run / "submission.txt").read_text()
    assert not (run / "artifacts").exists()
    assert str(root) in capture.read_text()
    assert "Generation job: 12345" in result.stdout


def test_verify_rejects_unbound_completion_receipt(api, tmp_path):
    minimal_manifest(api, tmp_path)
    shared.atomic_json(
        tmp_path / "generation.json", {"status": "failed", "manifest_sha256": "invalid"}
    )
    with pytest.raises(ValueError, match="receipt differs"):
        api.verify(tmp_path)


def test_shared_module_import_does_not_load_experiments():
    script = """
import sys
before = tuple(sys.path)
from spatial.v2 import entailment_generation, entailment_artifact_audit
assert tuple(sys.path) == before
assert 'generation_policy' not in sys.modules
assert 'audit_entailment_candidate' not in sys.modules
assert 'rerender_controlled_training' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_policy_schedule_fingerprint_is_unchanged():
    import hashlib
    import json

    # Menu policy and retry cap change, but every original slot/seed/quota stays frozen.
    manifest = policy.policy_manifest()
    assert manifest.pop("export_splits") == {
        "in-range-test": "sft-bench",
        "extrapolation-test": "sft-bench",
    }
    assert manifest.pop("admission_tokenizer_policy") == "pinned-training-tokenizer-v1"
    assert manifest.pop("menu_policy_version") == "synthetic-count-options-0-4-v1"
    assert manifest.pop("synthetic_count_menu_values") == list(range(5))
    assert (
        manifest.pop("scene_policy_version") == "explicit-size-checked-augmentation-v1"
    )
    assert manifest.pop("size_policy")["version"] == "linear-coupled-sizes-v1"
    assert manifest.pop("augmentation_candidate_limit") == 512
    assert manifest.pop("multiple_which_entailed_objects") == 2
    assert "varied_context" not in manifest
    assert (
        manifest.pop("training_duplicate_policy") == "last-attempt-within-train-only-v1"
    )
    assert manifest["construction_candidates_per_slot"] == 5
    manifest["construction_candidates_per_slot"] = 3
    encoded = json.dumps(manifest, sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == POLICY_BASELINE_SHA


def test_explicit_size_policy_reaches_every_declared_pair(api):
    observed = {}
    for slot in policy.schedule():
        observed.setdefault(slot.level, set()).add(api.sample_size(slot, 0))
    for level in range(1, 11):
        expected = {
            (n, n + extra)
            for n in range(2 * level + 4, 2 * level + 9)
            for extra in range(2, 7)
        }
        assert observed[level] == expected
        assert policy.SIZE_TABLE[str(level)] == {
            "objects": [2 * level + 4, 2 * level + 8],
            "premises": [2 * level + 6, 2 * level + 14],
        }


@pytest.mark.parametrize(
    "identifier,expected",
    [
        (
            "final-v1-extrapolation-test-which-l10-multiple-0000-a0",
            "38459902958452005953245546424105191186621656683718127745177535629981031273698",
        ),
        (
            "final-v1-extrapolation-test-count-l10-A-D-0004-a0",
            "61198149903076494997852316741356304880101993749407754686838206335199307858264",
        ),
        (
            "final-v1-in-range-test-which-l02-multiple-0000-a0",
            "57135732903260282151405340736792605891783273443561062487762196264301397136869",
        ),
        (
            "final-v1-in-range-test-count-l02-A-D-0004-a0",
            "85989471253802887786963008667235528917000673250537728468729645450719094964498",
        ),
        (
            "final-v1-in-range-test-which-l01-A-D-0000-a0",
            "164420418524556471117466636013384305485020696655951904774157574833079296155",
        ),
        (
            "final-v1-in-range-test-count-l01-ambiguous-0000-a0",
            "45801562641526918532257497820503407723629362262884065258065184915480463303207",
        ),
    ],
)
def test_integrated_constructor_matches_successful_trial(api, identifier, expected):
    slot = next(s for s in policy.schedule() if s.identity(0) == identifier)
    problem = api.construct(slot, 0)[1]
    assert shared.identities(problem)["query_key"] == expected


def test_explicit_construction_is_hash_seed_independent():
    code = """
import sys, json, hashlib
sys.path.insert(0, sys.argv[1])
import generation_policy as policy
from spatial.v2.entailment_generation import GenerationWorkflow
from spatial.v2.serialization import tagged_dataclass_to_dict
workflow = GenerationWorkflow(policy.workflow_config())
selected = {}
for slot in policy.schedule():
    if slot.case == "multiple":
        selected.setdefault((slot.split, slot.level), slot)
digest = hashlib.sha256()
for slot in selected.values():
    for alternative in (0, 1):
        story, problem, _, _ = workflow.construct(slot, alternative)
        digest.update(json.dumps(tagged_dataclass_to_dict(problem), sort_keys=True).encode())
print(digest.hexdigest())
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(EXP)],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("7", "42")
    ]
    assert len(outputs[0]) == 64
    assert outputs[0] == outputs[1]


def test_count_menu_sampling_law_has_no_category_or_position_cue():
    from itertools import permutations

    pool = policy.policy_manifest()["synthetic_count_menu_values"]
    # Enumerate the sampling law, with equally weighted sole values and letters.
    # For each ordered menu, A-D, E and F have identical menu probabilities.
    ad, f = Counter(), Counter()
    for gold in pool:
        others = [v for v in pool if v != gold]
        for distractors in permutations(others, 3):
            for position in range(4):
                menu = list(distractors)
                menu.insert(position, gold)
                ad[tuple(menu), "ABCD"[position]] += 1
        for menu in permutations(others):
            f[menu] += 1
    e = Counter(permutations(pool, 4))
    assert len(e) == 120
    for menu in e:
        assert f[menu] == e[menu] == 1
        assert [ad[menu, letter] for letter in "ABCD"] == [1] * 4
    # Actual finite-seed distributions still require diagnostics; this proves
    # that the menu sampling law itself no longer leaks the true value range.
    for split, size in policy.SPLITS.items():
        rows = [s for s in policy.schedule() if s.split == split and s.kind == "count"]
        assert len(rows) == size // 3
        # Every displayed option is <=4: the old rules become always E / always A.
        assert sum(s.case not in ("A-D", "F") for s in rows) == len(rows) * 30 // 100
        assert sum(s.answer_position == "A" for s in rows) == len(rows) * 15 // 100


def test_review_packet_covers_policy_and_actual_longest_rows():
    records = [
        {
            "rows": [{"id": s.identity(0)}],
            "split": s.split,
            "query_kind": s.kind,
            "construction_level": s.level,
            "case": s.case,
            "slot": policy.asdict(s),
            "tokens": [
                {"style": "natural", "training_tokens": i + 1},
                {"style": "symbolic", "training_tokens": 20000 - i},
            ],
        }
        for i, s in enumerate(policy.schedule())
    ]
    selected, receipt = shared.select_review_records(records)
    assert (selected, receipt) == shared.select_review_records(records[::-1])
    ids = {r["rows"][0]["id"] for r in selected}
    assert len(ids) == len(selected) == receipt["selected_count"]
    assert receipt["status"] == "selected-not-reviewed"
    assert {
        (r["split"], r["query_kind"], r["construction_level"]) for r in selected
    } == {(r["split"], r["query_kind"], r["construction_level"]) for r in records}
    cases = Counter((r["query_kind"], r["case"]) for r in selected)
    assert len(cases) == 13
    assert min(cases.values()) >= 10
    for split in policy.SPLITS:
        for level in policy.levels(split):
            cell = [
                r
                for r in selected
                if r["split"] == split and r["construction_level"] == level
            ]
            assert any(r["slot"]["contingent"] for r in cell)
            assert {
                r["slot"]["count_value"]
                for r in cell
                if r["query_kind"] == "count" and r["case"] in ("A-D", "F")
            } == set(range(5))
        members = [r for r in records if r["split"] == split]
        for index in (0, 1):
            longest = max(members, key=lambda r: r["tokens"][index]["training_tokens"])
            assert longest["rows"][0]["id"] in ids


def test_review_packet_handles_small_smoke_without_claiming_full_coverage():
    records = [
        {
            "rows": [{"id": "only"}],
            "split": "train",
            "query_kind": "direction",
            "construction_level": 1,
            "case": "A-D",
            "slot": {"contingent": False},
            "tokens": [{"style": "natural", "training_tokens": 5}],
        }
    ]
    selected, receipt = shared.select_review_records(records)
    assert selected == records
    assert receipt["available_case_counts"] == {"case/direction/A-D": 1}
    assert receipt["status"] == "selected-not-reviewed"


def test_workflows_do_not_share_experiment_configuration(api, tmp_path):
    exp = tmp_path / "experiments/other"
    exp.mkdir(parents=True)
    (exp / "generate.py").write_text("# separate experiment\n")
    coverage = exp / "coverage.json"
    coverage.write_text('{"policy": {"coverage_minima": []}}\n')
    config = replace(
        api.config,
        root=tmp_path,
        experiment=exp,
        coverage_policy=coverage,
        family_prefix="other",
        anchor_prefix="other-anchor",
    )
    other = shared.GenerationWorkflow(config)
    assert other.tokenizer is None
    other.tokenizer = "other-active"
    assert api.tokenizer != other.tokenizer
    hashes = other.fingerprints()
    assert set(hashes) == {
        "experiments/other/generate.py",
        "experiments/other/coverage.json",
    }
    assert api.config.experiment == EXP


def test_historical_audit_imports_use_shared_implementations():
    import importlib.util

    from spatial.v2 import entailment_artifact_audit

    for filename, names in (
        ("audit_entailment_candidate.py", ("audit_record", "replay_paths", "label")),
        ("audit_selected_explanation_depth.py", ("replay", "certificate_depth")),
    ):
        spec = importlib.util.spec_from_file_location(
            "compatibility_test",
            ROOT / "experiments/spatialentail-composition-v1" / filename,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in names:
            assert getattr(module, name) is getattr(entailment_artifact_audit, name)


def test_experiment_launcher_has_no_generation_implementation():
    tree = ast.parse((EXP / "generate.py").read_text())
    assert not any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body)
    assert len((EXP / "generate.py").read_text().splitlines()) < 40


def test_script_dependencies_preserve_isolated_generation_versions():
    import tomllib

    text = (EXP / "generate.py").read_text()
    metadata = text.split("# /// script\n", 1)[1].split("# ///", 1)[0]
    config = tomllib.loads(
        "\n".join(line.removeprefix("# ") for line in metadata.splitlines())
    )
    assert config["requires-python"] == ">=3.12,<3.13"
    assert "z3-solver==4.16.0.0" in config["dependencies"]
    assert len(config["dependencies"]) == 7
    assert not any(
        dep.startswith(("torch", "vllm", "axolotl")) for dep in config["dependencies"]
    )
