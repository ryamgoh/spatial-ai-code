from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).with_name("summarize_sft_6000.py")
SPEC = importlib.util.spec_from_file_location("sft_6000_summary", MODULE)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def sample(text: str, answer: str, family: str, subtype: str, cell: str) -> dict:
    return {
        "doc": {
            "text": text,
            "oracle_option": "A",
            "semantic_subtype": subtype,
            "generation_cell": cell,
            "difficulty": {
                "question_family": family,
                "semantic_subtype": subtype,
                "world_consistency": "consistent",
            },
        },
        "filtered_resps": [answer],
        "resps": [f"Answer: {answer}"],
    }


def write(root: Path, tag: str, rows: list[dict]) -> None:
    target = root / tag
    target.mkdir()
    (target / f"responses_{REPORT.SUMMARY.TASK}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_6000_report_emits_paired_fixes_and_regressions(tmp_path) -> None:
    before = [
        sample("fixed", "B", "direction", "dir-2", "semantic-dir2"),
        sample("regressed", "A", "which", "which-2", "semantic-which"),
        sample("stable", "A", "count", "count-1", "semantic-count"),
    ]
    after = [
        sample("fixed", "A", "direction", "dir-2", "semantic-dir2"),
        sample("regressed", "B", "which", "which-2", "semantic-which"),
        sample("stable", "A", "count", "count-1", "semantic-count"),
    ]
    write(tmp_path, "sft-1500-v13-stage1", before)
    write(tmp_path, "sft-6000-v13-stage1", after)

    result, examples = REPORT.summarize(tmp_path)

    assert "Paired stage-1 changes" in result
    assert "fixed by 6K" in result
    assert "regressed at 6K" in result
    assert {row["transition"] for row in examples} == {"fixed-by-6k", "regressed-at-6k"}
    assert "overall does not fall more than 2 pp versus 1.5K" in result
