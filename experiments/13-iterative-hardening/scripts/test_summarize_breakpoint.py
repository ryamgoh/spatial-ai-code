from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).with_name("summarize_breakpoint.py")
SPEC = importlib.util.spec_from_file_location("summarize_v13_breakpoint", MODULE)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def sample(text: str, generation_cell: str, answer: str = "A") -> dict:
    return {
        "doc": {
            "text": text,
            "oracle_option": "A",
            "generation_cell": generation_cell,
            "difficulty": {},
        },
        "filtered_resps": [answer],
    }


def write(root: Path, tag: str, rows: list[dict]) -> None:
    target = root / tag
    target.mkdir()
    (target / f"responses_{REPORT.TASK}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_breakpoint_report_separates_scale_and_challenge(tmp_path: Path) -> None:
    rows = [
        sample("clean", "break-long-clean-cardinal-x6-y6-none"),
        sample("branch", "break-long-query-branch-mixed-x8-y8-query-branch"),
        sample("closed", "break-long-loop-which-mixed-y-disconnected-l10-closed"),
        sample("open", "break-long-loop-which-mixed-y-disconnected-l10-open-control"),
    ]
    for tag in REPORT.DEFAULT_TAGS:
        write(tmp_path, tag, rows)

    result = REPORT.summarize(tmp_path, REPORT.DEFAULT_TAGS)

    assert "V13.1 structural breakpoint" in result
    assert "long-query-branch" in result
    assert "matched open chain" in result
    assert "Structural scale" in result
    assert "50–80%" in result
