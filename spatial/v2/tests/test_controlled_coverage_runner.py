"""The coverage runner records failures and preserves existing receipts."""

import importlib.util
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
SPEC = importlib.util.spec_from_file_location(
    "controlled_coverage_runner", DIRECTORY / "run_controlled_coverage.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("suffix", (".json", ".manifest.json", ".md", ".examples.md"))
def test_existing_output_blocks_rerun(tmp_path, monkeypatch, suffix):
    path = tmp_path / "coverage.json"
    saved = path.with_suffix(suffix)
    saved.write_text("unchanged")
    monkeypatch.setattr(
        sys, "argv", ["runner", "--report", str(path), "--tokenizer", "/tmp/missing"]
    )
    with pytest.raises(SystemExit) as exc:
        MODULE.main()
    assert exc.value.code == 2
    assert saved.read_text() == "unchanged"
    assert len(list(tmp_path.iterdir())) == 1


def test_failure_is_not_retried(monkeypatch):
    calls = []

    def fail(*_args, **_kwargs):
        calls.append(1)
        return {"stage": "depth", "status": "unsupported", "reason": "test failure"}

    monkeypatch.setattr(MODULE, "build_attempt", fail)
    result = MODULE.run_slot(MODULE.slots()[0])
    assert calls == [1]
    assert result["status"] == "failed"
    assert "test failure" in result["reason"]
    assert "independent_audit" not in result
