from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).with_name("summarize_probe_v2.py")
SPEC = importlib.util.spec_from_file_location("probe_v2_summary", MODULE)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def sample(text: str, family: str, subtype: str, cell: str, predicted: str = "A") -> dict:
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
        "filtered_resps": [predicted],
    }


def write(root: Path, tag: str, task: str, rows: list[dict]) -> None:
    target = root / tag
    target.mkdir()
    (target / f"responses_{task}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_probe_v2_report_emits_v13_only_gates(tmp_path) -> None:
    rows = [
        sample("d2", "direction", "dir-2", "semantic-dir2"),
        sample("which", "which", "which-2", "semantic-which"),
        sample("count", "count", "count-1", "semantic-count"),
        sample("depth", "direction", "dir-1", "depth-d5-cardinal-none"),
    ]
    for tag in ("baseline-stage1", "probe-v13-stage1", "probe-v2-v13-stage1", "baseline", "probe-v13-stage2", "probe-v2-v13-stage2"):
        write(tmp_path, tag, REPORT.SUMMARY.TASK, rows)
    v12_rows = [sample(f"v12-{i}", "direction", "dir-1", "v12") for i in range(2000)]
    write(tmp_path, "probe-v12-stage1", REPORT.V1.V12_TASK, v12_rows)
    write(tmp_path, "probe-v2-v12-stage1", REPORT.V1.V12_TASK, v12_rows)

    result = REPORT.summarize(tmp_path)

    assert "V12 compatibility is reported but is not a Probe v2 gate" in result
    assert "closed and open each >= 50%" in result
    assert "**GO** only if all six V13 gates pass" in result
