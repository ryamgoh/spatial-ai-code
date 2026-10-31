"""Resumable CPU execution of a fixed composition plan. Never submit or train."""

import argparse
import json
import multiprocessing
import os
import resource
import signal
import sys
import tempfile
import time
from collections import Counter, deque
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from queue import Empty, Queue

from spatial.v2.composition_production import (
    KINDS,
    REVISION,
    admit,
    artifact_digest,
    generate,
    has_shortfalls,
    plan,
    preflight,
    query_profile,
    schedule,
    summarize,
    summary_record,
    tagged_dataclass_to_dict,
    write_exports,
)
from spatial.v2.composition_solver_reuse import QuestionSolver

VERSION = "composition-execution-v2"
_worker = None


def production_api(policy):
    """Select versioned semantics while sharing ordered execution and checkpoints."""
    if policy.get("schema") == "entailment-production-v1":
        from spatial.v2 import entailment_production

        return entailment_production
    from types import SimpleNamespace

    return SimpleNamespace(
        admit=admit,
        plan=plan,
        schedule=schedule,
        generate=generate,
        summarize=summarize,
        has_shortfalls=has_shortfalls,
        write_exports=write_exports,
    )


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return sha256(encoded(value).encode()).hexdigest()


def atomic_json(path, value):
    """A killed write cannot turn a partially written record into a checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(encoded(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def worker_init(policy, tokenizer_path):
    global _worker
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from transformers import AutoTokenizer

    _worker = (
        policy,
        AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True),
        QuestionSolver(timeout_ms=policy["solver_timeout_ms"]),
    )


def worker_admit(task):
    index, entry, problem = task
    policy, tokenizer, solver = _worker
    solver.clear()
    started = time.perf_counter()
    try:
        result = production_api(policy).admit(entry, problem, policy, tokenizer, solver)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (
            index,
            result,
            time.perf_counter() - started,
            rss if sys.platform == "darwin" else rss * 1024,
        )
    finally:
        solver.clear()


def work_items(entries, policy):
    """Apply exactly the legacy bounds/dedup order before dispatching admission."""
    seen, jobs = set(), []
    for entry in entries:
        sizes = {
            "objects": len(entry.story.objects),
            "premises": len(entry.story.premise.operands),
            "candidates": len(entry.story.candidates),
        }
        if any(n > policy["scene_limits"][key] for key, n in sizes.items()):
            continue
        for kind, problem in zip(KINDS, entry.story.problems()):
            if kind not in entry.query_kinds:
                continue
            key = query_profile(problem)["key"]
            if key in seen:
                continue
            seen.add(key)
            jobs.append((len(jobs), entry, problem))
    return jobs


def work_identity(task):
    index, entry, problem = task
    return {
        "index": index,
        "story": tagged_dataclass_to_dict(entry.story),
        "problem": tagged_dataclass_to_dict(problem),
        "split": entry.split,
        "depth": entry.depth,
    }


class Checkpoints:
    def __init__(self, directory, identity, jobs):
        self.directory = directory
        self.identity = identity
        self.jobs = {
            i: fingerprint(work_identity(task)) for task in jobs for i in [task[0]]
        }

    def path(self, index):
        return self.directory / f"{index:06d}.json"

    def save(self, index, record, seconds, peak_rss_bytes=0):
        payload = {
            "identity": self.identity,
            "task": self.jobs[index],
            "index": index,
            "seconds": seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "record": record,
        }
        atomic_json(
            self.path(index), {"sha256": fingerprint(payload), "payload": payload}
        )

    def load(self, index):
        envelope = json.loads(self.path(index).read_text())
        payload = envelope["payload"]
        if (
            fingerprint(payload) != envelope["sha256"]
            or payload["identity"] != self.identity
            or payload["task"] != self.jobs[index]
            or payload["index"] != index
        ):
            raise ValueError(f"checkpoint integrity mismatch: {index}")
        return payload["record"]

    def existing(self):
        found = set()
        for path in sorted(self.directory.glob("*.json")):
            try:
                index = int(path.stem)
            except ValueError as exc:
                raise ValueError(f"unexpected checkpoint: {path.name}") from exc
            if index not in self.jobs or path != self.path(index):
                raise ValueError(f"checkpoint is outside the fixed plan: {path.name}")
            self.load(index)
            found.add(index)
        return found


class CompletionGate:
    """Only run the full audit once exact retained split sizes make success possible."""

    def __init__(self, policy):
        self.policy = policy
        self.retained = Counter()

    def satisfied(self, record, records):
        if record["status"] != "accepted":
            return False
        self.retained[record["split"]] += 1
        if self.retained != Counter(self.policy["base_targets"]):
            return False
        api = production_api(self.policy)
        return not api.has_shortfalls(api.summarize(records, self.policy))


@contextmanager
def admissions(
    jobs, policy, tokenizer, workers, checkpoints, progress, *, activity=None
):
    """Bound submitted work; persist completion order but consume fixed order."""
    existing = checkpoints.existing()
    pending = deque(task for task in jobs if task[0] not in existing)
    activity = {} if activity is None else activity
    activity.update(
        reused_checkpoints=len(existing),
        submitted_this_run=0,
        completed_admissions=len(existing),
        consumed_admissions=0,
    )
    indices = {
        (entry.story.story_id, type(problem.query).__name__): i
        for i, entry, problem in jobs
    }
    completed = set(existing)

    def read(index):
        record = checkpoints.load(index)
        activity["consumed_admissions"] += 1
        return record

    if len(existing) == len(jobs):
        progress(len(existing), len(jobs), "checkpoint-replay", len(existing))
        yield lambda entry, problem: read(
            indices[(entry.story.story_id, type(problem.query).__name__)]
        )
        return
    results = Queue()
    active = 0
    frontier = 0
    context = multiprocessing.get_context("spawn")
    # Pool's context manager terminates workers on an exception or interruption.
    with context.Pool(
        workers, initializer=worker_init, initargs=(policy, str(tokenizer))
    ) as pool:
        # Pool otherwise silently replaces a killed worker and loses its result.
        original_processes = tuple(pool._pool)

        def fill():
            nonlocal active
            # Bound distance ahead of ordered consumption, not just active jobs.
            # A slow early question must not let the entire reserve run ahead.
            while (
                active < 2 * workers
                and pending
                and pending[0][0] < frontier + 2 * workers
            ):
                task = pending.popleft()
                pool.apply_async(
                    worker_admit,
                    (task,),
                    callback=lambda result: results.put((True, result)),
                    error_callback=lambda exc: results.put((False, exc)),
                )
                active += 1
                activity["submitted_this_run"] += 1

        progress(len(completed), len(jobs), "admission", len(existing))

        def lookup(entry, problem):
            nonlocal active, frontier
            index = indices[(entry.story.story_id, type(problem.query).__name__)]
            frontier = index
            if index not in completed:
                fill()
            while index not in completed:
                if not active:
                    raise RuntimeError("fixed admission work exhausted unexpectedly")
                try:
                    ok, result = results.get(timeout=10)
                except Empty:
                    if any(p.exitcode is not None for p in original_processes):
                        raise RuntimeError(
                            "worker exited unexpectedly; resume unchanged work"
                        )
                    progress(len(completed), len(jobs), "admission", len(existing))
                    continue
                active -= 1
                if not ok:
                    raise RuntimeError(
                        "worker failed; checkpoints preserved"
                    ) from result
                done, record, seconds, rss = result
                checkpoints.save(done, record, seconds, rss)
                completed.add(done)
                activity["completed_admissions"] = len(completed)
                fill()
                progress(len(completed), len(jobs), "admission", len(existing))
            return read(index)

        try:
            yield lookup
        finally:
            # Early stop and interruption both cancel queued/running reserve work.
            pool.terminate()
            pool.join()
            # Preserve results already delivered before cancellation, without
            # adding speculative records to the ordered retained dataset.
            while not results.empty():
                ok, result = results.get_nowait()
                if ok:
                    done, record, seconds, rss = result
                    checkpoints.save(done, record, seconds, rss)
                    completed.add(done)
            activity["completed_admissions"] = len(completed)


def input_identity(config_path, policy, exclusion_path, tokenizer):
    source = Path(__file__).parent
    # Paths are relative identifiers so the same pinned run can move to HPC.
    return {
        "schema": VERSION,
        "production_schema": policy.get("schema"),
        "semantic_interface": policy.get("interface"),
        "config_sha256": artifact_digest(config_path),
        "policy": policy,
        "exclusions_sha256": artifact_digest(exclusion_path),
        "sources_sha256": {
            p.name: artifact_digest(p) for p in sorted(source.glob("*.py"))
        },
        "tokenizer_sha256": {
            p.name: artifact_digest(p)
            for p in sorted(tokenizer.iterdir())
            if p.is_file()
        },
        "versions": {
            p: version(p)
            for p in (
                "z3-solver",
                "transformers",
                "tokenizers",
                "jinja2",
                "PyYAML",
                "numpy",
            )
            + (
                ("scipy",)
                if policy.get("retention") == "coverage-constrained-v1"
                else ()
            )
        },
        "python": list(sys.version_info[:3]),
    }


def _execute(config, output, tokenizer, workers, *, resume=False, progress_interval=10):
    if type(workers) is not int or not 1 <= workers <= 128:
        raise ValueError("workers must be an integer in 1-128")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated is not None and workers > int(allocated):
        raise ValueError("workers exceeds SLURM_CPUS_PER_TASK")
    if tokenizer.name != REVISION or not tokenizer.is_dir():
        raise ValueError("provide the pinned local tokenizer snapshot")
    if output.exists() != resume or not output.parent.is_dir():
        raise ValueError(
            "use a fresh output path, or --resume for an existing execution"
        )
    policy = json.loads(config.read_text())
    api = production_api(policy)
    blockers = api.plan(policy)["blockers"]
    if blockers:
        raise ValueError("; ".join(blockers))
    frozen_sources = policy.get("frozen_sources_sha256")
    if frozen_sources is not None and frozen_sources != {
        p.name: artifact_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }:
        raise ValueError(
            "frozen generation software changed; compile a new configuration"
        )
    audit_spec = policy.get("support_audit")
    if audit_spec is not None:
        if artifact_digest(config.parent / audit_spec["path"]) != audit_spec["sha256"]:
            raise ValueError("support-audit receipt changed")
        if audit_spec["tokenizer_sha256"] != {
            p.name: artifact_digest(p) for p in tokenizer.iterdir() if p.is_file()
        }:
            raise ValueError("runtime tokenizer differs from the support audit")
    exclusion_path = config.parent / policy["development_exclusions"]["path"]
    if artifact_digest(exclusion_path) != policy["development_exclusions"]["sha256"]:
        raise ValueError("development exclusion inventory hash mismatch")
    identity = input_identity(config, policy, exclusion_path, tokenizer)
    manifest_path = output / "manifest.json"
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity:
            raise ValueError(
                "resume input/version mismatch; use a new output directory"
            )
        if manifest.get("status") == "completed-unsealed-candidate":
            raise ValueError("execution already complete")
    else:
        output.mkdir()
        (output / "checkpoints").mkdir()
        manifest = {
            "identity": identity,
            "status": "preflight",
            "training_authorized": False,
            "manual_audit": "pending",
        }
    manifest["workers"] = workers
    atomic_json(manifest_path, manifest)
    started, last_report = time.monotonic(), 0.0
    statuses = Counter()

    def progress(done, total, stage, reused=0, force=False):
        nonlocal last_report
        now = time.monotonic()
        if not force and now - last_report < progress_interval:
            return
        last_report = now
        elapsed = now - started
        report = {
            "stage": stage,
            "completed": done,
            "total": total,
            "reused_checkpoints": reused,
            "elapsed_seconds": round(elapsed, 2),
            "new_admissions_per_second": round(
                max(0, done - reused) / max(elapsed, 0.001), 4
            ),
            "workers": workers,
            "ordered_statuses": dict(statuses),
            "ordered_attempts": sum(statuses.values()),
            "stop_reason": manifest.get("stop_reason"),
        }
        atomic_json(output / "progress.json", report)
        print(encoded(report), file=sys.stderr, flush=True)

    try:
        progress(0, policy["attempt_budget"], "reconstructing-plan", force=True)
        entries = api.schedule(policy)
        if sum(len(e.query_kinds) for e in entries) != policy["attempt_budget"]:
            raise ValueError("fixed attempt budget mismatch")
        progress(0, policy["attempt_budget"], "structural-preflight", force=True)
        exclusions = json.loads(exclusion_path.read_text())
        keys = preflight(entries, exclusions)
        progress(0, policy["attempt_budget"], "deduplicating-plan", force=True)
        jobs = work_items(entries, policy)
        checkpoints = Checkpoints(output / "checkpoints", fingerprint(identity), jobs)
        manifest.update(
            status="running-unsealed-candidate",
            scheduled_stories=len(entries),
            attempt_budget=policy["attempt_budget"],
            admission_jobs=len(jobs),
            stopping_rule="first-ordered-prefix-satisfying-all-requirements",
        )
        atomic_json(manifest_path, manifest)
        records = []
        gate = api.make_gate(policy) if hasattr(api, "make_gate") else None
        gate = gate if gate is not None else CompletionGate(policy)
        constrained = hasattr(gate, "selected_ids")
        stop_reason = "attempt-budget-exhausted"
        activity = {}
        # Rebuild the ordered journal from immutable raw checkpoints on resume.
        with (
            admissions(
                jobs,
                policy,
                tokenizer,
                workers,
                checkpoints,
                progress,
                activity=activity,
            ) as lookup,
            (output / "attempts.jsonl.tmp").open("w") as stream,
        ):
            generation_options = {"admission": lookup, "keys": keys}
            if constrained:
                generation_options["admission_ids"] = {
                    f"{entry.story.story_id}-{type(problem.query).__name__.removesuffix('Query').lower()}"
                    for _, entry, problem in jobs
                }
            for record in api.generate(
                entries,
                policy,
                exclusions,
                None,
                None,
                **generation_options,
            ):
                records.append(summary_record(record))
                statuses[record["status"]] += 1
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if gate.satisfied(record, records):
                    stop_reason = "requirements-satisfied"
                    break
        if constrained:
            selected_ids = (
                gate.selected_ids
                if gate.selected_ids is not None
                else gate.diagnostic_ids()
            )
            for record in records:
                if record["status"] == "accepted":
                    record["admission_status"] = "accepted"
                    if record["base_id"] not in selected_ids:
                        record["status"] = "selection-excess"
            statuses = Counter(record["status"] for record in records)
            journal = output / "attempts.jsonl.tmp"
            selected_journal = output / "attempts.selected.jsonl.tmp"
            with journal.open() as source, selected_journal.open("w") as stream:
                for line in source:
                    record = json.loads(line)
                    if record["status"] == "accepted":
                        record["admission_status"] = "accepted"
                        if record["base_id"] not in selected_ids:
                            record["status"] = "selection-excess"
                    stream.write(json.dumps(record) + "\n")
            os.replace(selected_journal, journal)
            atomic_json(
                output / "retention.json",
                {
                    "policy": policy["retention"],
                    "solver_calls": gate.solver_calls,
                    "selection": "coverage-passing"
                    if gate.selected_ids is not None
                    else "diagnostic-quota-only",
                    "selected_base_ids": sorted(selected_ids),
                },
            )
        if input_identity(config, policy, exclusion_path, tokenizer) != identity:
            raise ValueError("inputs changed during execution")
        os.replace(output / "attempts.jsonl.tmp", output / "attempts.jsonl")
        progress(
            activity["completed_admissions"],
            len(jobs),
            "summarizing",
            activity["reused_checkpoints"],
            force=True,
        )
        summary = api.summarize(records, policy)
        if stop_reason == "requirements-satisfied" and api.has_shortfalls(summary):
            raise ValueError("early-stop audit disagrees with final audit")
        stopping = {
            "stop_reason": stop_reason,
            "ordered_attempts": len(records),
            "unprocessed_attempts": policy["attempt_budget"] - len(records),
            **activity,
            "speculative_completed_admissions": (
                activity["completed_admissions"] - activity["consumed_admissions"]
            ),
        }
        summary["execution"] = stopping
        atomic_json(output / "summary.json", summary)
        # Interrupted exports are rebuilt from checkpoints, never appended.
        with tempfile.TemporaryDirectory(prefix=".exports-", dir=output) as staging:
            with (output / "attempts.jsonl").open() as stream:
                api.write_exports((json.loads(line) for line in stream), Path(staging))
            for path in Path(staging).iterdir():
                os.replace(path, output / path.name)
        manifest.update(
            status="completed-unsealed-candidate",
            acceptance="shortfalls"
            if api.has_shortfalls(summary)
            else "automated-checks-passed",
            statuses=dict(statuses),
            **stopping,
            output_sha256={
                p.name: artifact_digest(p)
                for p in output.iterdir()
                if p.is_file() and p.name not in ("manifest.json", "progress.json")
            },
        )
        atomic_json(manifest_path, manifest)
        progress(
            activity["completed_admissions"],
            len(jobs),
            "completed-unsealed",
            activity["reused_checkpoints"],
            force=True,
        )
        return summary
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_json(manifest_path, manifest)
        raise


def execute(config, output, tokenizer, workers, *, resume=False, progress_interval=10):
    """Prevent two coordinators from writing the same run, including on resume."""
    import fcntl

    if not output.parent.is_dir():
        raise ValueError("output parent must exist")
    lock_path = output.parent / f".{output.name}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another coordinator owns this output") from exc
        return _execute(
            config,
            output,
            tokenizer,
            workers,
            resume=resume,
            progress_interval=progress_interval,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    summary = execute(
        args.config, args.output, args.tokenizer, args.workers, resume=args.resume
    )
    if has_shortfalls(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
