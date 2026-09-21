from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_SPEC = importlib.util.spec_from_file_location(
    "checkpoint_config", HERE / "make_checkpoint_eval_config.py"
)
assert CONFIG_SPEC and CONFIG_SPEC.loader
CONFIG = importlib.util.module_from_spec(CONFIG_SPEC)
CONFIG_SPEC.loader.exec_module(CONFIG)

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "checkpoint_audit", HERE / "summarize_checkpoint_audit.py"
)
assert AUDIT_SPEC and AUDIT_SPEC.loader
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def sample(text: str, answer: str = "A") -> dict:
    return {
        "doc": {
            "text": text,
            "oracle_option": "A",
            "generation_cell": "break-long-query-branch-cardinal-x10-y10-query-branch",
            "difficulty": {},
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


def test_checkpoint_config_replaces_exactly_one_lora_path(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    output = tmp_path / "output.yaml"
    source.write_text("model_args:\n  lora_path: old/path\n  max_lora_rank: 64\n")

    CONFIG.render(source, output, "new/checkpoint-1000")

    assert "lora_path: new/checkpoint-1000" in output.read_text()
    assert "old/path" not in output.read_text()


def test_checkpoint_audit_reports_paired_final_changes(tmp_path: Path) -> None:
    best = [sample("fixed", "B")]
    final = [sample("fixed", "A")]
    for tag, rows in (
        ("sft-8000-v13-stage1", best),
        ("sft-8000-final-v13-stage1", final),
    ):
        write(tmp_path, tag, AUDIT.SUMMARY.TASK, rows)
    for tag, rows in (
        ("breakpoint-sft-8k-stage1", best),
        ("breakpoint-sft-8k-final-stage1", final),
    ):
        write(tmp_path, tag, AUDIT.BREAK.TASK, rows)

    report, examples = AUDIT.summarize(tmp_path)

    assert "checkpoint-selection audit" in report
    assert "fixed by final" in report
    assert "+1" in report
    assert len(examples) == 2
