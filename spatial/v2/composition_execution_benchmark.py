"""Compare baseline, reuse and CPU workers on fixed existing development scenes."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from spatial.v2 import composition_balance
from spatial.v2.composition_execution import (
    Checkpoints,
    admissions,
    atomic_json,
    fingerprint,
    input_identity,
)
from spatial.v2.composition_production import REVISION, Entry, admit
from spatial.v2.composition_readable import check_pair
from spatial.v2.composition_scene_sampler import sample_scene
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.solver import Direction, SpatialSolverV2


def development_jobs(repeats=1):
    """These scaffolds/depths were already exercised in development tests."""
    scenes = [
        sample_scene(3, 2, matches=2),
        sample_scene(
            7,
            8,
            direction=Direction.NORTHEAST,
            proof_pattern="axis-split",
            matches=0,
        ),
    ]
    jobs = []
    for _ in range(repeats):
        for story in scenes:
            entry = Entry(
                story,
                "train",
                2 if story == scenes[0] else 8,
                ("direction", "which", "count"),
            )
            for problem in story.problems():
                jobs.append((len(jobs), entry, problem))
    return jobs


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("provide a fresh report path under an existing parent")
    if args.tokenizer.name != REVISION or not args.tokenizer.is_dir():
        parser.error("pinned local tokenizer required")
    if args.repeats < 1 or any(not 1 <= w <= 128 for w in args.workers):
        parser.error("positive repeats and worker counts in 1-128 required")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and max(args.workers) > int(allocated):
        parser.error("benchmark workers exceeds allocated Slurm CPUs")
    policy = json.loads(args.config.read_text())
    policy["paired_omission_percent"] = 100
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    jobs = development_jobs(args.repeats)
    baseline_jobs = development_jobs()
    exclusions = (
        args.config.parent
        / json.loads(args.config.read_text())["development_exclusions"]["path"]
    )
    identity = input_identity(args.config, policy, exclusions, args.tokenizer)
    report = {
        "status": "development-performance-check",
        "identity": identity,
        "jobs": len(jobs),
        "unique_scenes": 2,
        "production_generation": False,
        "runs": [],
    }
    expected = {}
    # Reconstruct the pre-change outline check as well as its uncached solver.
    old_check = composition_balance.check_rendered_outline
    try:
        composition_balance.check_rendered_outline = (
            lambda evidence, menu, mode, outline, pair: check_pair(
                evidence, menu, mode, pair, solver=SpatialSolverV2(backend="z3")
            )
        )
        start = time.perf_counter()
        for index, entry, problem in baseline_jobs:
            result = admit(
                entry, problem, policy, tokenizer, SpatialSolverV2(backend="z3")
            )
            expected[index] = fingerprint(result)
        report["runs"].append(
            {
                "mode": "legacy-serial",
                "jobs": len(baseline_jobs),
                "seconds": time.perf_counter() - start,
            }
        )
    finally:
        composition_balance.check_rendered_outline = old_check
    start = time.perf_counter()
    for index, entry, problem in baseline_jobs:
        result = admit(entry, problem, policy, tokenizer, QuestionSolver())
        if fingerprint(result) != expected[index]:
            raise ValueError(f"reuse changed output for development job {index}")
    report["runs"].append(
        {
            "mode": "reuse-serial",
            "jobs": len(baseline_jobs),
            "seconds": time.perf_counter() - start,
        }
    )
    expected = {i: expected[i % len(baseline_jobs)] for i, _, _ in jobs}
    print(json.dumps(report["runs"]), flush=True)
    for workers in args.workers:
        with tempfile.TemporaryDirectory(prefix="composition-cpu-check-") as directory:
            store = Checkpoints(Path(directory), fingerprint(identity), jobs)
            start = time.perf_counter()
            # Repeated benchmark tasks intentionally share story IDs; use one
            # uniquely identified story per replay, preserving the original base
            # ID for deterministic menus by dispatching each repeat separately.
            if args.repeats != 1:
                # Workers are evaluated directly here to keep repeated identities
                # independent without changing any scene, menu seed or gold.
                from multiprocessing import get_context

                from spatial.v2.composition_execution import worker_admit, worker_init

                with get_context("spawn").Pool(
                    workers,
                    initializer=worker_init,
                    initargs=(policy, str(args.tokenizer)),
                ) as pool:
                    for index, result, seconds, rss in pool.imap_unordered(
                        worker_admit, jobs
                    ):
                        if fingerprint(result) != expected[index]:
                            raise ValueError(f"parallel output mismatch at {index}")
                        store.save(index, result, seconds, rss)
            else:
                with admissions(
                    jobs, policy, args.tokenizer, workers, store, lambda *a: None
                ) as lookup:
                    for index, entry, problem in jobs:
                        if fingerprint(lookup(entry, problem)) != expected[index]:
                            raise ValueError(f"parallel output mismatch at {index}")
            elapsed = time.perf_counter() - start
            payloads = [
                json.loads(p.read_text())["payload"]
                for p in Path(directory).glob("*.json")
            ]
            run = {
                "mode": "reuse-processes",
                "workers": workers,
                "seconds": elapsed,
                "questions_per_second": len(jobs) / elapsed,
                "max_worker_peak_rss_bytes": max(p["peak_rss_bytes"] for p in payloads),
                "exact_output_match": True,
            }
            report["runs"].append(run)
            print(json.dumps(run), flush=True)
    if input_identity(args.config, policy, exclusions, args.tokenizer) != identity:
        raise ValueError("benchmark inputs changed during measurement")
    report["exact_output_match"] = True
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
