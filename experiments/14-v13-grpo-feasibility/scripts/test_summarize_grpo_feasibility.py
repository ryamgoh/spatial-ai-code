from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).with_name("summarize.py")
SPEC = importlib.util.spec_from_file_location("summarize_v14_grpo", MODULE)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def sample(text: str, answer: str) -> dict:
    return {
        "doc": {
            "text": text,
            "oracle_option": "A",
            "generation_cell": "break-rl-eval-d14-query-branch-cardinal-x14-y14-query-branch",
        },
        "filtered_resps": [answer],
    }


def write(root: Path, tag: str, rows: list[dict]) -> None:
    target = root / tag
    target.mkdir(parents=True, exist_ok=True)
    (target / f"responses_{REPORT.TASKS['holdout']}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_summary_reports_calibration_and_paired_holdout(tmp_path: Path) -> None:
    write(tmp_path, "sft-holdout-stage1", [sample("fixed", "B")])
    write(tmp_path, "grpo-holdout-stage1", [sample("fixed", "A")])
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {"pass_rate": 0.5, "mixed_groups": 20, "num_prompts": 48, "go": True}
        )
    )

    result = REPORT.summarize(tmp_path, calibration)

    assert "Pass rate: 50.0%" in result
    assert "fixed by GRPO" in result
    assert "GRPO improves the frozen hard holdout" in result
