from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).with_name("summarize_v12_bridge.py")
SPEC = importlib.util.spec_from_file_location("v12_bridge", MODULE)
assert SPEC and SPEC.loader
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def test_bridge_summary_scores_identical_prompts(tmp_path) -> None:
    prompt = "Consider a map. Question: In which direction is A relative to B?"
    test = tmp_path / "test.jsonl"
    test.write_text(json.dumps({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "### Final Deduction\n- A. Northeast — yes. In.\nAnswer: A"},
        ]
    }) + "\n")
    run = tmp_path / "tag"
    run.mkdir()
    (run / f"responses_{BRIDGE.TASK}.jsonl").write_text(json.dumps({
        "doc": {"text": prompt, "oracle_option": "A"},
        "filtered_resps": ["A"],
    }) + "\n")

    report = BRIDGE.summarize(tmp_path, ("tag",), test)

    assert "100.0% (1/1)" in report
    assert "`dir-1`" in report
