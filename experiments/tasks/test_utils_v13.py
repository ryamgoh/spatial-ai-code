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
                "semantic_subtype": "dir-1",
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
        "semantic_subtype": "dir-1",
        "difficulty": {
            "relation_mix": "mixed",
            "x_depth": 2,
            "y_depth": 3,
            "axes_independent": True,
            "semantic_subtype": "dir-1",
        },
        "difficulty_schema_version": 1,
        "generator_version": "v13.0-cardinal-foundation",
        "generation_cell": "mixed-dir-1-independent",
    }


def test_v13_loader_normalizes_legacy_cycle_metadata() -> None:
    rows = FakeDataset(
        [
            {
                "messages": [{"role": "user", "content": "question"}],
                "oracle_option": "E",
                "base_semantic_subtype": "which-2",
                "difficulty": {
                    "question_type": 1,
                    "semantic_subtype": "which-cycle",
                    "world_consistency": "inconsistent",
                },
                "difficulty_schema_version": 3,
            }
        ]
    )

    converted = process_docs_v13_sft(rows)[0]

    assert converted["semantic_subtype"] == "which-2"
    assert converted["difficulty"]["semantic_subtype"] == "which-2"
    assert "base_semantic_subtype" not in converted
