"""Reference-only export cannot silently change protected dataset content."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
SPEC = importlib.util.spec_from_file_location(
    "spatial_prose_rerender", EXP / "rerender_controlled_training.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(EXP))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def test_only_reference_version_text_and_tokens_are_mutable():
    original = {
        "problem": {"objects": ["Cafe", "Library"]},
        "evidence": {"positive": ["proof"]},
        "rows": [
            {
                "id": "fixed-id",
                "answer": "A",
                "depth": 2,
                "menu": ["North"],
                "prompt": "fixed prompt",
                "references": {"natural": "old"},
                "reference_version": "old",
            }
        ],
        "tokens": [{"target_tokens": 100}],
    }
    updated = deepcopy(original)
    updated["rows"][0]["references"] = {"natural": "new"}
    updated["rows"][0]["reference_version"] = "new"
    updated["tokens"] = [{"target_tokens": 50}]
    assert MODULE.preserved_projection(updated) == MODULE.preserved_projection(original)
    for field, value in (
        ("answer", "E"),
        ("depth", 3),
        ("prompt", "new"),
        ("menu", ["West"]),
        ("id", "new"),
    ):
        changed = deepcopy(updated)
        changed["rows"][0][field] = value
        assert MODULE.preserved_projection(changed) != MODULE.preserved_projection(
            original
        )
    changed = deepcopy(updated)
    changed["evidence"]["positive"] = []
    assert MODULE.preserved_projection(changed) != MODULE.preserved_projection(original)
    assert original["rows"][0]["references"] == {"natural": "old"}


@pytest.mark.parametrize("suffix", (".json", ".manifest.json", ".md", ".examples.md"))
def test_existing_output_prevents_rerender(tmp_path, monkeypatch, suffix):
    receipt = tmp_path / "receipt.json"
    saved = receipt.with_suffix(suffix)
    saved.write_text("preserve")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--source",
            "/tmp/missing",
            "--receipt",
            str(receipt),
            "--output",
            str(tmp_path / "candidate"),
            "--tokenizer",
            "/tmp/missing",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        MODULE.main()
    assert exc.value.code == 2
    assert saved.read_text() == "preserve"
    assert len(list(tmp_path.iterdir())) == 1
