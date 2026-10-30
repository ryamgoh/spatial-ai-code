"""Existing development fixtures only: ordered workers, quotas and resume."""

import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from spatial.v2 import composition_execution as execution
from spatial.v2 import composition_production as production
from spatial.v2.composition_scene_sampler import sample_scene
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_targeted_sampler import sample_targeted
from spatial.v2.solver import SpatialSolverV2

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/spatialentail-composition-v1/production-config-v1.json"
TOKENIZER = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots"
    / production.REVISION
)
EMPTY = {"family_ids": [], "story_keys": []}


@pytest.fixture
def policy():
    result = json.loads(CONFIG.read_text())
    result["paired_omission_percent"] = 100
    return result


@pytest.fixture
def tokenizer():
    from transformers import AutoTokenizer

    if not TOKENIZER.is_dir():
        pytest.skip("pinned local tokenizer is required")
    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def entries():
    return (
        production.Entry(
            sample_scene(3, 2), "train", 2, ("direction", "which", "count")
        ),
        production.Entry(sample_scene(7, 2, matches=0), "train", 2, ("count",)),
        production.Entry(
            sample_targeted(3, "contingent"),
            "train",
            None,
            ("direction", "which", "count"),
        ),
    )


def normalized(value):
    return json.loads(json.dumps(value))


def test_full_admission_reuse_preserves_all_outputs(policy, tokenizer):
    entry = entries()[0]
    for problem in entry.story.problems():
        before = production.admit(
            entry, problem, policy, tokenizer, SpatialSolverV2(backend="z3")
        )
        after = production.admit(entry, problem, policy, tokenizer, QuestionSolver())
        assert after["status"] == "accepted", after.get("reason")
        assert after == before


@pytest.mark.parametrize("workers", (1, 2))
def test_process_workers_match_serial_and_resume_without_readmitting(
    tmp_path, policy, tokenizer, workers, monkeypatch
):
    fixtures = entries()
    expected = normalized(
        list(
            production.generate(
                fixtures, policy, EMPTY, tokenizer, SpatialSolverV2(backend="z3")
            )
        )
    )
    jobs = execution.work_items(fixtures, policy)
    store = execution.Checkpoints(tmp_path, "fixed-test-identity", jobs)
    keys = production.preflight(fixtures, EMPTY)
    with execution.admissions(
        jobs, policy, TOKENIZER, workers, store, lambda *a: None
    ) as lookup:
        actual = list(
            production.generate(
                fixtures, policy, EMPTY, None, None, admission=lookup, keys=keys
            )
        )
    assert normalized(actual) == expected
    assert len(store.existing()) == len(jobs)

    # A fully checkpointed replay must not need a worker task or tokenizer.
    with execution.admissions(
        jobs, policy, "/not/a/tokenizer", workers, store, lambda *a: None
    ) as lookup:
        replay = list(
            production.generate(
                fixtures, policy, EMPTY, None, None, admission=lookup, keys=keys
            )
        )
    assert normalized(replay) == expected


def test_quota_reduction_is_original_order_not_worker_finish_order(
    policy, tokenizer, monkeypatch
):
    fixtures = entries()
    fixtures = (fixtures[0], replace(fixtures[1], query_kinds=("direction", "count")))
    targets = production.quotas(policy)
    targets["train/direction/2"] = 1
    monkeypatch.setattr(production, "quotas", lambda _: targets)
    # Duplicated canonical input must be suppressed globally.
    fixtures = (
        *fixtures,
        replace(
            fixtures[0], story=replace(fixtures[0].story, story_id="duplicate-scene")
        ),
    )
    jobs = execution.work_items(fixtures, policy)
    raw = {}
    for _, entry, problem in reversed(jobs):
        raw[(entry.story.story_id, type(problem.query))] = production.admit(
            entry, problem, policy, tokenizer, QuestionSolver()
        )
    expected = list(
        production.generate(
            fixtures, policy, EMPTY, tokenizer, SpatialSolverV2(backend="z3")
        )
    )
    actual = list(
        production.generate(
            fixtures,
            policy,
            EMPTY,
            None,
            None,
            admission=lambda entry, problem: deepcopy(
                raw[(entry.story.story_id, type(problem.query))]
            ),
        )
    )
    assert actual == expected
    assert [r["status"] for r in actual][-3:] == ["duplicate"] * 3
    assert any(r["status"] == "quota-excess" for r in actual)


def test_checkpoints_reject_corruption_identity_changes_and_extra_work(
    tmp_path, policy
):
    jobs = execution.work_items(entries(), policy)
    store = execution.Checkpoints(tmp_path, "identity", jobs)
    store.save(0, {"status": "accepted"}, 1.0)
    assert store.load(0) == {"status": "accepted"}
    altered = execution.Checkpoints(tmp_path, "different", jobs)
    with pytest.raises(ValueError, match="integrity"):
        altered.load(0)
    value = json.loads(store.path(0).read_text())
    value["payload"]["record"]["status"] = "changed"
    store.path(0).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity"):
        store.existing()
    store.save(0, {"status": "accepted"}, 1.0)
    (tmp_path / "999999.json").write_text("{}")
    with pytest.raises(ValueError, match="outside"):
        store.existing()


def test_partial_temporary_checkpoint_is_not_considered_completed(tmp_path, policy):
    store = execution.Checkpoints(
        tmp_path, "identity", execution.work_items(entries(), policy)
    )
    (tmp_path / "000000.json.tmp").write_text("{")
    assert not store.existing()


def test_worker_limit_respects_slurm_allocation(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    with pytest.raises(ValueError, match="exceeds"):
        execution.execute(CONFIG, tmp_path / "out", TOKENIZER, 32)
    with pytest.raises(ValueError, match="1-128"):
        execution.execute(CONFIG, tmp_path / "out", TOKENIZER, 129)


def frozen_config(tmp_path, policy):
    policy["status"] = "frozen"
    policy["source_batches"] = [
        {
            "generator": "sampled-forest",
            "seed": 3,
            "count": 1,
            "split": "train",
            "query_kinds": ["direction"],
            "depth": 1,
        }
    ]
    policy["attempt_budget"] = 1
    policy["coverage_minima"] = [{"match": {"answer": "A"}, "minimum_bases": 1}]
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps(EMPTY))
    policy["development_exclusions"] = {
        "path": exclusions.name,
        "sha256": production.artifact_digest(exclusions),
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(policy))
    return config


def test_executor_completes_and_reports_shortfalls(tmp_path, policy, tokenizer):
    config = frozen_config(tmp_path, policy)
    output = tmp_path / "execution"
    summary = execution.execute(config, output, TOKENIZER, 1, progress_interval=0)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed-unsealed-candidate"
    assert manifest["acceptance"] == "shortfalls"
    assert production.has_shortfalls(summary)
    assert not manifest["training_authorized"]
    assert json.loads((output / "progress.json").read_text())["completed"] == 1
    assert len((output / "attempts.jsonl").read_text().splitlines()) == 1
    for name, expected in manifest["output_sha256"].items():
        assert production.artifact_digest(output / name) == expected
    with pytest.raises(ValueError, match="already complete"):
        execution.execute(config, output, TOKENIZER, 2, resume=True)
    policy["seed"] += 1
    config.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="mismatch"):
        execution.execute(config, output, TOKENIZER, 2, resume=True)


def test_resume_after_export_failure_uses_checkpoint(
    tmp_path, policy, tokenizer, monkeypatch
):
    config = frozen_config(tmp_path, policy)
    output = tmp_path / "execution"
    real = execution.write_exports
    monkeypatch.setattr(
        execution,
        "write_exports",
        lambda *a: (_ for _ in ()).throw(RuntimeError("export failed")),
    )
    with pytest.raises(RuntimeError, match="export failed"):
        execution.execute(config, output, TOKENIZER, 1)
    checkpoint = next((output / "checkpoints").glob("*.json"))
    before = checkpoint.read_bytes()
    monkeypatch.setattr(execution, "write_exports", real)
    execution.execute(config, output, TOKENIZER, 2, resume=True)
    assert checkpoint.read_bytes() == before
    assert (
        json.loads((output / "manifest.json").read_text())["status"]
        == "completed-unsealed-candidate"
    )


def fail_worker(task):
    raise RuntimeError("intentional worker failure")


def exit_worker(task):
    os._exit(3)


@pytest.mark.parametrize("worker", (fail_worker, exit_worker))
def test_worker_failure_aborts_instead_of_hanging(
    tmp_path, policy, tokenizer, monkeypatch, worker
):
    jobs = execution.work_items(entries()[:1], policy)
    store = execution.Checkpoints(tmp_path, "identity", jobs)
    monkeypatch.setattr(execution, "worker_admit", worker)
    with (
        pytest.raises(RuntimeError, match="worker"),
        execution.admissions(
            jobs, policy, TOKENIZER, 1, store, lambda *a: None
        ) as lookup,
    ):
        lookup(jobs[0][1], jobs[0][2])
    assert not store.existing()


def test_exclusive_coordinator_lock(tmp_path, monkeypatch):
    import fcntl

    output = tmp_path / "execution"
    with (tmp_path / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another coordinator"):
            execution.execute(CONFIG, output, TOKENIZER, 1)


def test_partial_resume_only_dispatches_missing_jobs(tmp_path, policy, tokenizer):
    jobs = execution.work_items(entries(), policy)
    store = execution.Checkpoints(tmp_path, "identity", jobs)
    index, entry, problem = jobs[0]
    expected = production.admit(entry, problem, policy, tokenizer, QuestionSolver())
    store.save(index, expected, 1.0)
    before = store.path(index).read_bytes()
    with execution.admissions(
        jobs, policy, TOKENIZER, 2, store, lambda *a: None
    ) as lookup:
        for _, entry, problem in jobs:
            lookup(entry, problem)
    assert len(store.existing()) == len(jobs)
    assert store.path(index).read_bytes() == before


def passing_gate_summary():
    return {
        "quota_shortfalls": {},
        "controlled_count_shortfalls": {},
        "source_shortfalls": {},
        "family_shortfalls": [],
        "family_minima_configured": True,
        "coverage_shortfalls": [],
        "direction_shortfalls": [],
        "unresolved_verification_failures": 0,
    }


@pytest.mark.parametrize("missing", list(passing_gate_summary()) + [None])
def test_early_stop_requires_every_audit_gate(monkeypatch, missing):
    summary = passing_gate_summary()
    if missing:
        summary[missing] = (
            False if missing == "family_minima_configured" else ["missing"]
        )
    calls = []

    def audit(records, policy):
        calls.append(len(records))
        return summary

    monkeypatch.setattr(execution, "summarize", audit)
    gate = execution.CompletionGate({"base_targets": {"train": 1, "dev": 1}})
    first = {"status": "accepted", "split": "train"}
    second = {"status": "accepted", "split": "dev"}
    assert not gate.satisfied(first, [first])
    assert not calls
    assert gate.satisfied(second, [first, second]) is (missing is None)
    assert calls == [2]
    assert not gate.satisfied({"status": "quota-excess"}, [first, second])
    assert calls == [2]


def test_total_size_without_correct_split_sizes_cannot_stop(monkeypatch):
    monkeypatch.setattr(execution, "summarize", lambda *a: pytest.fail("too early"))
    gate = execution.CompletionGate({"base_targets": {"train": 1, "dev": 1}})
    record = {"status": "accepted", "split": "train"}
    assert not gate.satisfied(record, [record])
    assert not gate.satisfied(record, [record, record])


def test_parallel_read_ahead_is_bounded_by_ordered_frontier(
    tmp_path, policy, monkeypatch
):
    jobs = execution.work_items(entries(), policy)
    assert len(jobs) > 4
    store = execution.Checkpoints(tmp_path, "identity", jobs)
    submitted = []

    class Pool:
        _pool = ()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def apply_async(self, function, args, callback, error_callback):
            index = args[0][0]
            submitted.append(index)
            result = (index, {"status": "accepted"}, 0.0, 0)
            if index == 0:
                self.first = (callback, result)
            else:
                callback(result)
            if index == 3:
                self.first[0](self.first[1])

        def terminate(self):
            pass

        def join(self):
            pass

    class Context:
        def Pool(self, *args, **kwargs):
            return Pool()

    monkeypatch.setattr(execution.multiprocessing, "get_context", lambda _: Context())
    activity = {}
    with execution.admissions(
        jobs, policy, TOKENIZER, 2, store, lambda *a: None, activity=activity
    ) as lookup:
        assert lookup(jobs[0][1], jobs[0][2])["status"] == "accepted"
    assert submitted == [0, 1, 2, 3]
    assert activity["consumed_admissions"] == 1
    assert activity["completed_admissions"] == 4


@pytest.mark.parametrize("workers", (1, 2))
def test_executor_stops_at_successful_prefix_and_resume_stops_there_too(
    tmp_path, policy, tokenizer, monkeypatch, workers
):
    fixtures = entries()
    config = frozen_config(tmp_path, policy)
    policy["attempt_budget"] = sum(len(e.query_kinds) for e in fixtures)
    policy["base_targets"] = {
        role: 2 if role == "train" else 0 for role in production.ROLES
    }
    config.write_text(json.dumps(policy))
    # A reduced execution fixture exercises orchestration; the gate tests above
    # independently require every real has_shortfalls field.
    monkeypatch.setattr(execution, "plan", lambda _: {"blockers": []})
    monkeypatch.setattr(execution, "schedule", lambda _: fixtures)
    monkeypatch.setattr(
        production, "quotas", lambda _: {"train/direction/2": 1, "train/which/2": 1}
    )
    monkeypatch.setattr(
        production,
        "source_quotas",
        lambda _: {
            "train/direction/controlled": 1,
            "train/which/controlled": 1,
        },
    )
    monkeypatch.setattr(production, "count_quotas", lambda _: {})
    monkeypatch.setattr(execution, "summarize", lambda *a: passing_gate_summary())
    output = tmp_path / "stopped"
    # Interrupt export after the stop decision, then verify resume never consumes
    # speculative checkpoints beyond the same successful ordered prefix.
    export = execution.write_exports
    monkeypatch.setattr(execution, "write_exports", fail_export)
    with pytest.raises(RuntimeError, match="export"):
        execution.execute(config, output, TOKENIZER, workers)
    monkeypatch.setattr(execution, "write_exports", export)
    summary = execution.execute(config, output, TOKENIZER, workers, resume=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stop_reason"] == "requirements-satisfied"
    assert manifest["ordered_attempts"] == 2
    assert manifest["unprocessed_attempts"] == policy["attempt_budget"] - 2
    assert manifest["submitted_this_run"] == 0
    assert manifest["completed_admissions"] <= 2 * workers + 1
    assert not production.has_shortfalls(summary)
    records = [
        json.loads(line)
        for line in (output / "attempts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert all(r["status"] == "accepted" for r in records)
    progress = json.loads((output / "progress.json").read_text())
    assert progress["completed"] == manifest["completed_admissions"]
    assert progress["ordered_attempts"] == 2
    assert progress["stop_reason"] == "requirements-satisfied"


def fail_export(*args):
    raise RuntimeError("export interrupted")
