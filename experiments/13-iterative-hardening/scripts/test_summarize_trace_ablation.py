from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).with_name("summarize_trace_ablation.py")
SPEC = importlib.util.spec_from_file_location("trace_ablation_summary", MODULE)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def sample(text: str, cell: str, answer: str = "A") -> dict:
    return {
        "doc": {
            "text": text, "oracle_option": "A",
            "generation_cell": cell, "semantic_subtype": "dir-1",
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


def test_trace_ablation_report_is_paired_and_scale_aware(tmp_path: Path) -> None:
    full = [sample("fixed", "break-long-query-branch-cardinal-x10-y10-query-branch", "B")]
    delta = [sample("fixed", "break-long-query-branch-cardinal-x10-y10-query-branch", "A")]
    write(tmp_path, "sft-8000-v13-stage1", REPORT.SUMMARY.TASK, full)
    write(tmp_path, "trace-delta-v13-stage1", REPORT.SUMMARY.TASK, delta)
    write(tmp_path, "breakpoint-sft-8k-stage1", REPORT.BREAK.TASK, full)
    write(tmp_path, "trace-delta-breakpoint-stage1", REPORT.BREAK.TASK, delta)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"delta_to_full_char_ratio": 0.5}))

    result, examples = REPORT.summarize(tmp_path, manifest)

    assert "matched trace-format ablation" in result
    assert "Delta-state assistant characters are 50.0%" in result
    assert "long-query-branch / 10" in result
    assert "fixed by delta" in result
    assert len(examples) == 2
