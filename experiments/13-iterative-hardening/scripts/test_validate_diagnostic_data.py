from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("validate_diagnostic_data.py")
SPEC = importlib.util.spec_from_file_location("v13_validate", MODULE_PATH)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_validator_rejects_old_cycle_subtype(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(VALIDATOR, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(VALIDATOR, "EXPECTED_CELLS", 1)
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "generation_cell": "cycle-which-2",
                "difficulty_schema_version": 4,
                "generator_version": VALIDATOR.EXPECTED_GENERATOR,
                "semantic_subtype": "which-cycle",
                "difficulty": {"semantic_subtype": "which-cycle"},
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="invalid semantic subtype"):
        VALIDATOR.validate(path)
