from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).with_name("analyze_v13_error_traces.py")
SPEC = importlib.util.spec_from_file_location("trace_audit", MODULE)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _sample(prompt: str, prediction: str, *, depth: int = 4) -> dict:
    return {
        "doc": {
            "text": prompt,
            "oracle_option": "A",
            "semantic_subtype": "dir-1",
            "generation_cell": "depth-d4-cardinal-none",
            "difficulty": {
                "question_type": 0,
                "question_family": "direction",
                "semantic_subtype": "dir-1",
                "world_consistency": "consistent",
                "relation_mix": "cardinal-only",
                "x_depth": depth,
                "y_depth": depth,
                "num_relations": 2,
            },
        },
        "filtered_resps": [prediction],
        "resps": [[
            "### Step 1\n**X-Extraction**: Bank < Library\n**Y-Extraction**: none (cardinal X-only relation)\n"
            "### Step 2\n**X-Extraction**: Museum < Park\n**Y-Extraction**: none (wrong)\n"
            f"Answer: {prediction}"
        ]],
    }


def test_cardinal_axis_audit_detects_cross_axis_and_wrong_orientation() -> None:
    prompt = (
        "The Library is to the East of the Bank. "
        "The Park is to the West of the Museum.\n\nQuestion: x"
    )
    response = (
        "### Step 1\n**X-Extraction**: Bank < Library\n**Y-Extraction**: Library < Bank\n"
        "### Step 2\n**X-Extraction**: Museum < Park\n**Y-Extraction**: none\n"
    )

    counts = AUDIT.cardinal_axis_errors(prompt, response)

    assert counts["cardinal_cross_axis_update"] == 1
    assert counts["incorrect_active_axis_update"] == 1


def test_build_audit_finds_both_regression_cohorts(tmp_path) -> None:
    prompt = (
        "The Library is to the East of the Bank. "
        "The Park is to the West of the Museum.\n\nQuestion: x"
    )
    predictions = {
        "baseline-stage1": "A",
        "v12-4b-1.5k-stage1": "A",
        "v12-4b-6k-stage1": "B",
    }
    for tag, prediction in predictions.items():
        run = tmp_path / tag
        run.mkdir()
        (run / f"responses_{AUDIT.SUMMARY.TASK}.jsonl").write_text(
            json.dumps(_sample(prompt, prediction)) + "\n"
        )

    report, examples = AUDIT.build_audit(tmp_path, limit=2)

    assert "`1.5k-correct_6k-wrong` | 1" in report
    assert "`base-correct_1.5k-wrong` | 0" in report
    assert len(examples) == 1
    assert examples[0]["cohort"] == "1.5k-correct_6k-wrong"
