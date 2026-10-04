"""Evaluation-loader contract for Spatial V2 matrix rows."""

from __future__ import annotations

from utils import process_docs_v2_sft, strict_acc


class FakeDataset(list):
    def map(self, fn):
        return FakeDataset(fn(row) for row in self)


def test_v2_loader_uses_metadata_gold_through_z() -> None:
    rows = FakeDataset(
        [
            {
                "messages": [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "Answer: A, Z"},
                ],
                "metadata": {
                    "oracle_letters": ["A", "Z"],
                    "matrix_cell": "direction-depth-10",
                    "answer_mode": "all-possible",
                    "trace_format": "symbolic",
                    "state_mode": "delta",
                    "difficulty": {"x_depth": 10, "y_depth": 10},
                },
            }
        ]
    )

    converted = process_docs_v2_sft(rows)[0]

    assert converted == {
        "text": "question",
        "oracle_option": "A,Z",
        "matrix_cell": "direction-depth-10",
        "answer_mode": "all-possible",
        "trace_format": "symbolic",
        "state_mode": "delta",
        "difficulty": {"x_depth": 10, "y_depth": 10},
    }
    assert strict_acc(["A,Z", ["A,Z"]]) == 1.0
