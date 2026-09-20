from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).with_name("summarize_sft_1500.py")
SPEC = importlib.util.spec_from_file_location("sft_1500_summary", MODULE)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def sample(text: str, family: str, subtype: str, cell: str) -> dict:
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
        "filtered_resps": ["A"],
    }


def write(root: Path, tag: str, task: str, rows: list[dict]) -> None:
    target = root / tag
    target.mkdir()
    (target / f"responses_{task}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_1500_report_emits_scaling_gates(tmp_path) -> None:
    rows = [
        sample("d2", "direction", "dir-2", "semantic-dir2"),
        sample("which", "which", "which-2", "semantic-which"),
        sample("count", "count", "count-1", "semantic-count"),
        sample("depth", "direction", "dir-1", "depth-d5-cardinal-none"),
    ]
    for tag in ("baseline-stage1", "probe-v2-v13-stage1", "sft-1500-v13-stage1", "baseline", "probe-v2-v13-stage2", "sft-1500-v13-stage2"):
        write(tmp_path, tag, REPORT.SUMMARY.TASK, rows)
    v12 = [sample(f"v12-{index}", "direction", "dir-1", "v12") for index in range(2000)]
    write(tmp_path, "sft-1500-v12-stage1", REPORT.HELPERS.V1.V12_TASK, v12)

    result = REPORT.summarize(tmp_path)

    assert "1.5K scaling gates" in result
    assert "closed-loop and open-chain conditions each >= 60%" in result
    assert "Train 6K only if all six gates pass" in result
