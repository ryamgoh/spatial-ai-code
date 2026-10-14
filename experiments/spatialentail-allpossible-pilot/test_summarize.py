import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).with_name("scripts") / "summarize.py"
SPEC = importlib.util.spec_from_file_location("spatialentail_allpossible_summary", MODULE)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


def _metrics(strict, loose, stage1, format_valid):
    return {
        "strict_acc,none": strict,
        "loose_acc,none": loose,
        "stage1_acc,none": stage1,
        "stage1_format_valid,none": format_valid,
    }


def test_summary_reports_set_errors_and_cell_factors(tmp_path: Path) -> None:
    tasks = [
        {
            "task": "direction-g2",
            "role": "bench-all-possible",
            "component": "direction",
            "cell": "DIR-MP-ALT-G2",
            "query_kind": "direction",
            "count": 25,
        },
        {
            "task": "count-depth",
            "role": "bench-all-possible",
            "component": "count",
            "cell": "CNT-TRC-ALT-D5",
            "query_kind": "count",
            "count": 25,
        },
    ]
    (tmp_path / "jobs.json").write_text(
        json.dumps(
            {
                "schema": "spatialentail-all-possible-pilot-v1",
                "status": "engineering-only",
                "total_rows": 50,
                "samples_per_cell": 25,
                "tasks": tasks,
                "jobs": [{"name": "model"}, {"name": "missing"}],
            }
        )
    )
    result_dir = tmp_path / "results/model"
    result_dir.mkdir(parents=True)
    (result_dir / "COMPLETED").touch()
    (result_dir / "results.json").write_text(
        json.dumps(
            {
                "results": {
                    "direction-g2": _metrics(0.8, 0.9, 0.6, 0.5),
                    "count-depth": _metrics(0.4, 0.7, 0.2, 0.3),
                }
            }
        )
    )

    summary = SUMMARY.summarize(tmp_path)
    model = summary["models"]["model"]

    assert model["exact_set_accuracy"] == pytest.approx(0.6)
    assert model["gold_covered_accuracy"] == pytest.approx(0.8)
    assert model["over_inclusive_rate"] == pytest.approx(0.2)
    assert model["missing_gold_rate"] == pytest.approx(0.2)
    assert model["stage1_exact_set_accuracy"] == pytest.approx(0.4)
    assert model["stage1_format_validity"] == pytest.approx(0.4)
    assert model["query_accuracy"] == {
        "count": {"count": 25, "exact_set_accuracy": 0.4},
        "direction": {"count": 25, "exact_set_accuracy": 0.8},
    }
    assert model["gold_size_accuracy"] == {
        "2": {"count": 25, "exact_set_accuracy": 0.8}
    }
    assert model["depth_accuracy"] == {
        "5": {"count": 25, "exact_set_accuracy": 0.4}
    }
    assert summary["models"]["missing"] == {"complete": False}
    rendered = (tmp_path / "SUMMARY.md").read_text()
    assert "Engineering-only zero-shot transfer result" in rendered
    assert "| model | 0.6000 | 0.4000 | 0.4000 | 0.8000 | 0.2000 | 0.2000 |" in rendered
    assert "| `CNT-TRC-ALT-D5` | count | 25 | 0.4000 |" in rendered
