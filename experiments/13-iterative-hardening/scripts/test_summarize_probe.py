from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).with_name("summarize_probe.py")
SPEC = importlib.util.spec_from_file_location("probe_summary", MODULE)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def sample(text: str, family: str, gold: str = "A", predicted: str = "A") -> dict:
    qtype = {"direction": 0, "which": 1, "count": 2}[family]
    return {
        "doc": {
            "text": text,
            "oracle_option": gold,
            "generation_cell": "depth-d5-cardinal-none" if family == "direction" else f"semantic-{family}",
            "difficulty": {
                "question_type": qtype,
                "question_family": family,
                "world_consistency": "consistent",
            },
        },
        "filtered_resps": [predicted],
    }


def write_task(root: Path, tag: str, task: str, rows: list[dict]) -> None:
    target = root / tag
    target.mkdir()
    (target / f"responses_{task}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_probe_summary_emits_gate_verdicts(tmp_path) -> None:
    rows = [sample("d", "direction"), sample("w", "which"), sample("c", "count")]
    for tag in ("baseline-stage1", "probe-v13-stage1", "baseline", "probe-v13-stage2"):
        write_task(tmp_path, tag, PROBE.SUMMARY.TASK, rows)
    write_task(
        tmp_path,
        "probe-v12-stage1",
        PROBE.V12_TASK,
        [sample(f"v12-{index}", "direction") for index in range(2000)],
    )

    report = PROBE.summarize(tmp_path)

    assert "V12 retention" in report
    assert "depth-5 drop <= 5 pp" in report
    assert "dir-2 improves" in report
    assert "**GO** only if all five gates pass" in report
