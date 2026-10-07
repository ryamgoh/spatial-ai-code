"""Tests for the reproducible SpatialMap-TQA audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from audit_spatialeval_v2 import audit_dataset


def _prompt(statements: list[str], target: str, reference: str) -> str:
    return (
        "Consider a map with multiple objects:\n"
        + " ".join(statements)
        + "\n\nPlease answer the following multiple-choice question based on the "
        + f"provided information. In which direction is {target} relative to "
        + f"{reference}? Available options:\nA. Northeast\nB. Northwest\n"
        + "C. Southwest\nD. Southeast."
    )


def _rows() -> list[dict]:
    return [
        {
            "id": "exact",
            "text": _prompt(["A is to the Northeast of B."], "A", "B"),
            "oracle_answer": "Northeast",
            "oracle_option": "A",
            "oracle_full_answer": "A. Northeast",
        },
        {
            "id": "ambiguous",
            "text": _prompt(
                [
                    "A is to the Northeast of B.",
                    "B is to the Northwest of C.",
                ],
                "A",
                "C",
            ),
            "oracle_answer": "Northeast",
            "oracle_option": "A",
            "oracle_full_answer": "A. Northeast",
        },
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_audit_dataset_preserves_source_and_builds_corr(tmp_path: Path) -> None:
    source = tmp_path / "original.jsonl"
    rows = _rows()
    _write_jsonl(source, rows)
    original_bytes = source.read_bytes()

    artifacts = audit_dataset(source, tmp_path / "audit", backend="reference")

    assert source.read_bytes() == original_bytes
    audits = _read_jsonl(artifacts.audit)
    corrected = _read_jsonl(artifacts.corrected)
    summary = json.loads(artifacts.summary.read_text())

    assert [row["status"] for row in audits] == [
        "exact-match",
        "underdetermined-oracle-possible",
    ]
    assert audits[1]["possible_answers"] == ["Northeast", "Northwest"]
    assert set(audits[1]["witnesses"]) == {"Northeast", "Northwest"}

    assert corrected[0]["oracle_option"] == "A"
    assert corrected[0]["oracle_answer"] == "Northeast"
    assert corrected[1]["oracle_option"] == "E"
    assert corrected[1]["oracle_answer"] == "Cannot be determined"
    assert corrected[1]["oracle_full_answer"] == "E. Cannot be determined"
    assert all("E. Cannot be determined" in row["text"] for row in corrected)
    assert all("audit" not in row for row in corrected)

    assert summary["input"]["rows"] == 2
    assert summary["counts"]["status"] == {
        "exact-match": 1,
        "underdetermined-oracle-possible": 1,
    }
    assert summary["counts"]["query_kind"] == {"direction": 2}
    assert summary["outputs"]["audit"]["rows"] == 2
    assert summary["outputs"]["corrected"]["rows"] == 2


def test_audit_dataset_is_deterministic_and_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "original.jsonl"
    _write_jsonl(source, _rows())
    first = audit_dataset(source, tmp_path / "first", backend="reference")
    second = audit_dataset(source, tmp_path / "second", backend="reference")

    assert first.audit.read_bytes() == second.audit.read_bytes()
    assert first.corrected.read_bytes() == second.corrected.read_bytes()

    with pytest.raises(FileExistsError, match="already exist"):
        audit_dataset(source, tmp_path / "first", backend="reference")

    replaced = audit_dataset(
        source,
        tmp_path / "first",
        backend="reference",
        replace=True,
    )
    assert replaced.audit.read_bytes() == second.audit.read_bytes()
    assert replaced.corrected.read_bytes() == second.corrected.read_bytes()
