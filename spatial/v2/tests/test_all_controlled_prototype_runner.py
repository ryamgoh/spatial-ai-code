"""Fixed-slot runner guards do not permit a silent retry or replacement."""

import importlib.util
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
)
SPEC = importlib.util.spec_from_file_location(
    "all_controlled_runner", DIRECTORY / "run_all_controlled_prototype.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("suffix", (".json", ".manifest.json", ".md"))
def test_existing_receipt_prevents_rerun(tmp_path, monkeypatch, suffix):
    report = tmp_path / "report.json"
    existing = report.with_suffix(suffix)
    existing.write_text("preserve this receipt")
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--report", str(report), "--tokenizer", f"/tmp/{MODULE.REVISION}"],
    )
    with pytest.raises(SystemExit) as exc:
        MODULE.main()
    assert exc.value.code == 2
    assert existing.read_text() == "preserve this receipt"
    assert len(list(tmp_path.iterdir())) == 1


def test_failed_admission_is_reported_once_without_refill(monkeypatch):
    calls = []

    def reject(*_args, **kwargs):
        calls.append(kwargs)
        return {"stage": "depth", "status": "unsupported", "reason": "test rejection"}

    monkeypatch.setattr(MODULE, "build_attempt", reject)
    result = MODULE.run_slot(MODULE.slots()[0])
    assert len(calls) == 1
    assert result["status"] == "failed"
    assert "test rejection" in result["reason"]
    assert "independent_audit" not in result
