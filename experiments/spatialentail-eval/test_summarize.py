import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).with_name("scripts") / "summarize.py"
SPEC = importlib.util.spec_from_file_location("spatialentail_eval_summary", MODULE)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


def test_summary_reports_sections_cells_and_depth_curves(tmp_path: Path) -> None:
    tasks = [
        {
            "task": "core",
            "role": "bench-core",
            "component": "core",
            "cell": "DIR-ARC-UNI",
            "query_kind": "direction",
            "count": 100,
        },
        {
            "task": "matched",
            "role": "bench-composition",
            "component": "composition",
            "cell": "DIR-MP-TRC-UNI",
            "query_kind": "direction",
            "count": 100,
        },
        {
            "task": "shift-composition",
            "role": "bench-shift",
            "component": "heldout-composition",
            "cell": "DIR-MP-TRC-MP-UNI",
            "query_kind": "direction",
            "count": 50,
        },
        {
            "task": "depth",
            "role": "bench-shift",
            "component": "depth",
            "cell": "direction-depth-3",
            "query_kind": "direction",
            "count": 50,
        },
    ]
    (tmp_path / "jobs.json").write_text(
        json.dumps(
            {
                "total_bases": 300,
                "tasks": tasks,
                "jobs": [{"name": "model"}, {"name": "missing"}],
            }
        )
    )
    result = tmp_path / "results/model"
    result.mkdir(parents=True)
    (result / "COMPLETED").touch()
    (result / "results.json").write_text(
        json.dumps(
            {
                "results": {
                    "core": {
                        "strict_acc,none": 0.8,
                        "loose_acc,none": 0.9,
                        "stage1_acc,none": 0.7,
                        "stage1_format_valid,none": 0.6,
                    },
                    "matched": {
                        "strict_acc,none": 0.7,
                        "loose_acc,none": 0.8,
                        "stage1_acc,none": 0.6,
                        "stage1_format_valid,none": 0.5,
                    },
                    "shift-composition": {
                        "strict_acc,none": 0.4,
                        "loose_acc,none": 0.5,
                        "stage1_acc,none": 0.3,
                        "stage1_format_valid,none": 0.2,
                    },
                    "depth": {
                        "strict_acc,none": 0.6,
                        "loose_acc,none": 0.7,
                        "stage1_acc,none": 0.5,
                        "stage1_format_valid,none": 0.4,
                    },
                }
            }
        )
    )

    summary = SUMMARY.summarize(tmp_path)

    assert summary["models"]["model"]["overall_micro_accuracy"] == pytest.approx(
        2 / 3
    )
    assert summary["models"]["model"][
        "stage1_overall_micro_accuracy"
    ] == pytest.approx(17 / 30)
    assert summary["models"]["model"]["stage1_format_validity"] == pytest.approx(
        14 / 30
    )
    assert summary["models"]["model"]["group_accuracy"] == {
        "bench-core": 0.8,
        "bench-composition": 0.7,
        "bench-shift": 0.5,
    }
    assert summary["models"]["model"]["group_query_accuracy"] == {
        "bench-core": {"direction": 0.8},
        "bench-composition": {"direction": 0.7},
        "bench-shift": {"direction": 0.5},
    }
    assert summary["models"]["model"]["depth_accuracy"] == {"direction": {"3": 0.6}}
    assert summary["models"]["missing"] == {"complete": False}
    rendered = (tmp_path / "SUMMARY.md").read_text()
    assert "| Model | Overall | CORE | COMP | SHIFT |" in rendered
    assert "SHIFT composition" not in rendered
    assert "| model | 0.5667 | 0.4667 | 0.6667 |" in rendered
    assert "## CORE breakdown (100 examples)" in rendered
    assert "## COMP breakdown (100 examples)" in rendered
    assert "## SHIFT breakdown (100 examples)" in rendered
    assert "| model | 0.8000 | 0.8000 | — | — |" in rendered
    assert "| `DIR-ARC-UNI` | core | 100 | 0.8000 | — |" in rendered
    assert "| `JCR` | Joint Count Reasoning obligation |" in rendered
