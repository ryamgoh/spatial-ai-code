"""CPU-only regression tests for the live Slurm result collector."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

SCRIPT = Path(__file__).resolve().parents[1] / "slurm/collect-results.sh"
POLICY = "final-v1-evaluation-8192-fewshot-v1"


def collect(evaluation):
    return subprocess.run(
        ["bash", str(SCRIPT), str(evaluation)],
        env={**os.environ, "SLURM_JOB_ID": "123"},
        capture_output=True,
        text=True,
    )


def test_collect_completed_and_partial_results(tmp_path):
    campaign = tmp_path / "campaign"
    evaluation = campaign / "evaluation" / POLICY
    trained = evaluation / "natural-seed-42"
    trained.mkdir(parents=True)
    base = trained / "natural-sft-bench"
    raw = b'{"outputs": []}'
    base.with_suffix(".json").write_bytes(raw)
    summary = {
        "style": "natural",
        "training_seed": 42,
        "evaluation": {"policy": {"version": POLICY}, "shots": 0},
        "result_sha256": hashlib.sha256(raw).hexdigest(),
        "scores": {
            "stage1": {"correct": 1800, "total": 3600, "slices": {"case": {}}},
            "stage2": {"correct": 2700, "total": 3600},
        },
        "token_profile": {"stage1": {"mean": 1234.5, "length_finish_count": 2}},
    }
    base.with_suffix(".summary.json").write_text(json.dumps(summary))
    pending = evaluation / "untuned-0-shot"
    pending.mkdir()
    partial = pending / "natural-sft-bench.partial.jsonl"
    partial.write_bytes(b'{"id": "1"}\n{"id": "2"}\n{"id":')
    selection = campaign / "natural-seed-42" / "selection.json"
    selection.parent.mkdir()
    selection.write_text('{"epoch": 2}')
    submission = campaign / "eval-submission-test"
    submission.mkdir()
    (submission / "jobs.txt").write_text("campaign=/somewhere\nselected-natural-42=456\n")
    (submission / "selected-natural-42-456.out").write_bytes(b"x" * 40000)

    result = collect(evaluation)

    assert result.returncode == 0, result.stderr
    assert "Completed summaries: 1/20" in result.stdout
    assert "50.00% | 75.00%" in result.stdout
    destination = next((campaign / "result-reports").glob("results-*/"))
    report = json.loads((destination / "report.json").read_text())
    assert len(report["results"]) == 20
    assert report["results"][0]["summary"] == summary
    entry = next(r for r in report["results"] if r["status"] == "partial")
    assert entry["completed"] == 2
    assert partial.read_bytes().endswith(b'{"id":')
    assert report["training"]["natural-seed-42"]["selection.json"]["epoch"] == 2
    log = destination / "evidence/eval-submission-test/selected-natural-42-456.out"
    assert log.stat().st_size == 32768
    with tarfile.open(destination.with_suffix(".tar.gz")) as archive:
        assert any(name.endswith("/report.json") for name in archive.getnames())
        assert not any(name.endswith("/natural-sft-bench.json") for name in archive.getnames())


def test_collect_reports_invalid_summary_without_stopping(tmp_path):
    evaluation = tmp_path / "campaign/evaluation" / POLICY
    model = evaluation / "natural-seed-42"
    model.mkdir(parents=True)
    (model / "natural-sft-bench.summary.json").write_text("{broken")
    result = collect(evaluation)
    assert result.returncode == 0, result.stderr
    assert "Completed summaries: 0/20" in result.stdout
    destination = next((tmp_path / "campaign/result-reports").glob("results-*/"))
    report = json.loads((destination / "report.json").read_text())
    assert report["results"][0]["status"] == "error"
    assert "error" in report["results"][0]


def test_collect_rejects_missing_directory(tmp_path):
    result = collect(tmp_path / "missing")
    assert result.returncode != 0
    assert "does not exist" in result.stderr
