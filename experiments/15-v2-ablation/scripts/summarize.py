#!/usr/bin/env python3
"""Summarize verified artifacts from a V2 ablation run."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(run_dir: Path) -> dict:
    jobs_path = run_dir / "jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else {"arms": {}}
    arms = {}
    for arm, job_ids in sorted(jobs.get("arms", {}).items()):
        model_dir = run_dir / "models" / arm
        result_dir = run_dir / "results" / arm
        train_complete = (
            (model_dir / "COMPLETED").exists() if "train" in job_ids else None
        )
        eval_complete = (result_dir / "COMPLETED").exists() and (
            result_dir / "results.json"
        ).exists()
        arms[arm] = {
            "jobs": job_ids,
            "train_complete": train_complete,
            "eval_complete": eval_complete,
            "results": str(result_dir / "results.json") if eval_complete else None,
        }
    summary = {"run_dir": str(run_dir), "arms": arms}
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = ["# V2 Ablation Run", ""]
    for arm, status in arms.items():
        lines.append(
            f"- {arm}: train={'not applicable' if status['train_complete'] is None else ('complete' if status['train_complete'] else 'missing')}, "
            f"eval={'complete' if status['eval_complete'] else 'missing'}"
        )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    summarize(Path(sys.argv[1]).resolve())
