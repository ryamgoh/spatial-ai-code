"""Evaluation-loader contract for synthetic v13 chat rows."""

from __future__ import annotations

from utils import process_docs_v13_sft


class FakeDataset(list):
    def map(self, fn):
        return FakeDataset(fn(row) for row in self)


def test_v13_loader_preserves_solver_measured_difficulty() -> None:
    rows = FakeDataset(
        [
            {
                "messages": [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "<think>x</think>\nAnswer: B, D"},
                ],
                "oracle_option": "B,D",
                "difficulty": {
                    "relation_mix": "mixed",
                    "x_depth": 2,
                    "y_depth": 3,
                    "axes_independent": True,
                },
                "difficulty_schema_version": 1,
                "generator_version": "v13.0-cardinal-foundation",
                "generation_cell": "mixed-dir-1-independent",
            }
        ]
    )

    converted = process_docs_v13_sft(rows)[0]

    assert converted == {
        "text": "question",
        "oracle_option": "B,D",
        "difficulty": {
            "relation_mix": "mixed",
            "x_depth": 2,
            "y_depth": 3,
            "axes_independent": True,
        },
        "difficulty_schema_version": 1,
        "generator_version": "v13.0-cardinal-foundation",
        "generation_cell": "mixed-dir-1-independent",
    }
