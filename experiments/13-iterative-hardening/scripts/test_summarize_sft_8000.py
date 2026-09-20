from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).with_name("summarize_sft_8000.py")
SPEC = importlib.util.spec_from_file_location("summarize_v13_sft_8000", MODULE)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def sample(text: str, generation_cell: str, answer: str = "A") -> dict:
    return {
        "doc": {
            "text": text,
            "oracle_option": "A",
            "generation_cell": generation_cell,
            "semantic_subtype": "dir-1",
            "difficulty": {
                "semantic_subtype": "dir-1",
                "world_consistency": "consistent",
                "question_family": "direction",
            },
        },
        "filtered_resps": [answer],
        "resps": [f"Answer: {answer}"],
    }


def write(root: Path, tag: str, task: str, rows: list[dict]) -> None:
    target = root / tag
    target.mkdir()
    (target / f"responses_{task}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_8000_report_requires_breakpoint_gain_and_v13_retention(tmp_path: Path) -> None:
    v13_before = [sample("v13-fixed", "semantic-cardinal-dir-1", "B")]
    v13_after = [sample("v13-fixed", "semantic-cardinal-dir-1", "A")]
    break_before = [
        sample(
            "break-fixed", "break-long-query-branch-cardinal-x10-y10-query-branch", "B"
        )
    ]
    break_after = [
        sample(
            "break-fixed", "break-long-query-branch-cardinal-x10-y10-query-branch", "A"
        )
    ]
    write(tmp_path, "sft-6000-v13-stage1", REPORT.SUMMARY.TASK, v13_before)
    write(tmp_path, "sft-8000-v13-stage1", REPORT.SUMMARY.TASK, v13_after)
    write(tmp_path, "breakpoint-sft-6k-stage1", REPORT.BREAK.TASK, break_before)
    write(tmp_path, "breakpoint-sft-8k-stage1", REPORT.BREAK.TASK, break_after)

    result, examples = REPORT.summarize(tmp_path)

    assert "Original V13 retention" in result
    assert "V13.1 breakpoint generalization" in result
    assert "held-out depth/length 10 >= 70%" in result
    assert {row["suite"] for row in examples} == {"original-v13", "v13.1-breakpoint"}
