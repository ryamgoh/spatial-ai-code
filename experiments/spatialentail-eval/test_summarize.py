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


def test_summary_reports_module_macros_and_depth_curves(tmp_path: Path) -> None:
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
                    "core": {"strict_acc,none": 0.8, "loose_acc,none": 0.9},
                    "matched": {"strict_acc,none": 0.7, "loose_acc,none": 0.8},
                    "shift-composition": {
                        "strict_acc,none": 0.4,
                        "loose_acc,none": 0.5,
                    },
                    "depth": {"strict_acc,none": 0.6, "loose_acc,none": 0.7},
                }
            }
        )
    )

    summary = SUMMARY.summarize(tmp_path)

    assert summary["models"]["model"]["overall_micro_accuracy"] == pytest.approx(
        2 / 3
    )
    assert summary["models"]["model"]["group_accuracy"] == {
        "bench-core": 0.8,
        "bench-composition": 0.7,
        "bench-shift": 0.5,
    }
    assert summary["models"]["model"]["component_accuracy"] == {
        "bench-core/core": 0.8,
        "bench-composition/composition": 0.7,
        "bench-shift/depth": 0.6,
        "bench-shift/heldout-composition": 0.4,
    }
    assert summary["models"]["model"]["depth_accuracy"] == {"direction": {"3": 0.6}}
    assert summary["models"]["missing"] == {"complete": False}
    assert (tmp_path / "SUMMARY.md").is_file()
