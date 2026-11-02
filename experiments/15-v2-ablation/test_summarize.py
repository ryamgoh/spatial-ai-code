"""Historical result summarization remains usable after orchestration retirement."""

import importlib.util
import json
from pathlib import Path


def test_summary_reports_partial_arm_completion(tmp_path):
    path = Path(__file__).parent / "scripts/summarize.py"
    spec = importlib.util.spec_from_file_location("v2_ablation_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "jobs.json").write_text(
        json.dumps({
            "arms": {
                "natural": {"train": "1", "eval": "2"},
                "answer-only": {"train": "3", "eval": "4"},
            }
        })
    )
    model = tmp_path / "models/natural"
    result = tmp_path / "results/natural"
    model.mkdir(parents=True)
    result.mkdir(parents=True)
    (model / "COMPLETED").touch()
    (result / "COMPLETED").touch()
    (result / "results.json").write_text("{}")
    summary = module.summarize(tmp_path)
    assert summary["arms"]["natural"]["eval_complete"] is True
    assert summary["arms"]["answer-only"]["train_complete"] is False
    assert (tmp_path / "SUMMARY.md").exists()
