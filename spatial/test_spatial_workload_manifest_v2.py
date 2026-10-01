"""Public contracts for V2 workload manifests and leakage validation."""

from __future__ import annotations

import json

import pytest
from generate_all_v2 import WorkloadSpec, generate_workload
from spatial_explanation_renderers_v2 import StateMode, TraceFormat
from spatial_generation_v2 import QueryKind, SemanticShape
from spatial_grading_v2 import AnswerMode
from spatial_workload_manifest_v2 import build_workload_manifest


def _row(row_id: str, base_id: str, prompt: str, trace_format: str) -> dict:
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Answer: A"},
        ],
        "metadata": {
            "base_id": base_id,
            "query_kind": "direction",
            "query_direction": None,
            "answer_mode": "single",
            "semantic_shape": "unique",
            "menu_coverage": "full",
            "trace_format": trace_format,
            "state_mode": "delta",
            "menu_status": "exact",
            "possible_values": ["North"],
            "round_trip_verified": True,
            "attempt": 0,
            "rejection_counts": {},
            "difficulty": {
                "x_depth": 2,
                "y_depth": 3,
                "possibility_count": 1,
                "num_distractor_premises": 2,
            },
        },
    }


def test_manifest_reports_distributions_and_paired_variants() -> None:
    train = [
        _row("base-1-natural", "base-1", "prompt one", "natural"),
        _row("base-1-symbolic", "base-1", "prompt one", "symbolic"),
    ]
    test = [
        _row("base-2-natural", "base-2", "prompt two", "natural"),
        _row("base-2-symbolic", "base-2", "prompt two", "symbolic"),
    ]

    manifest = build_workload_manifest(
        train,
        test,
        expected_trace_variants=(
            (TraceFormat.NATURAL, StateMode.DELTA),
            (TraceFormat.SYMBOLIC, StateMode.DELTA),
        ),
    )

    assert manifest["total_rows"] == 4
    assert manifest["base_problems"] == 2
    assert manifest["splits"] == {"train": 2, "test": 2}
    assert manifest["trace_distributions"]["trace_format"] == {
        "natural": 2,
        "symbolic": 2,
    }
    assert manifest["base_distributions"]["query_kind"] == {"direction": 2}
    assert manifest["difficulty"]["x_depth"] == {"2": 2}
    assert manifest["generation_efficiency"] == {
        "acceptance_rate": 1.0,
        "accepted_base_problems": 2,
        "max_rejections_before_accept": 0,
        "rejected_candidates": 0,
        "rejection_reasons": {},
        "total_candidate_attempts": 2,
    }
    assert manifest["validation"]["status"] == "passed"


def test_manifest_rejects_cross_split_base_or_prompt_leakage() -> None:
    train = [_row("train", "base-1", "same prompt", "natural")]
    test = [_row("test", "base-1", "same prompt", "symbolic")]

    with pytest.raises(ValueError, match="cross-split base IDs"):
        build_workload_manifest(train, test)


def test_manifest_reports_rejection_efficiency_per_base_problem() -> None:
    row = _row("base-1", "base-1", "prompt", "natural")
    row["metadata"]["attempt"] = 2
    row["metadata"]["rejection_counts"] = {
        "semantic shape did not match": 1,
        "difficulty controls did not match": 1,
    }

    manifest = build_workload_manifest([row], [])

    assert manifest["generation_efficiency"] == {
        "acceptance_rate": 1 / 3,
        "accepted_base_problems": 1,
        "max_rejections_before_accept": 2,
        "rejected_candidates": 2,
        "rejection_reasons": {
            "difficulty controls did not match": 1,
            "semantic shape did not match": 1,
        },
        "total_candidate_attempts": 3,
    }


def test_generator_writes_a_valid_manifest_next_to_splits(tmp_path) -> None:
    train_path, test_path = generate_workload(
        tmp_path / "manifested.jsonl",
        WorkloadSpec(
            samples_per_cell=2,
            query_kinds=(QueryKind.DIRECTION,),
            answer_modes=(AnswerMode.SINGLE,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            trace_formats=(TraceFormat.NATURAL, TraceFormat.SYMBOLIC),
            state_modes=(StateMode.DELTA,),
            test_split=0.5,
            seed=1737,
        ),
    )

    manifest_path = tmp_path / "manifested_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert train_path.exists() and test_path.exists()
    assert manifest["total_rows"] == 4
    assert manifest["base_problems"] == 2
    assert manifest["generation"]["seed"] == 1737
    assert manifest["validation"]["status"] == "passed"


def test_workload_forwards_proof_controls_into_manifest(tmp_path) -> None:
    generate_workload(
        tmp_path / "controlled.jsonl",
        WorkloadSpec(
            samples_per_cell=1,
            query_kinds=(QueryKind.DIRECTION,),
            answer_modes=(AnswerMode.SINGLE,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            trace_formats=(TraceFormat.SYMBOLIC,),
            state_modes=(StateMode.DELTA,),
            num_entities=7,
            num_premises=8,
            omit_direct_query_relation=True,
            min_axis_depth=2,
            max_axis_depth=4,
            test_split=0,
            seed=1721,
        ),
    )

    manifest = json.loads((tmp_path / "controlled_manifest.json").read_text())
    assert manifest["difficulty"]["direct_query_relation"] == {"False": 1}
    assert manifest["difficulty"]["x_depth"] in ({"2": 1}, {"3": 1}, {"4": 1})
    assert manifest["difficulty"]["y_depth"] in ({"2": 1}, {"3": 1}, {"4": 1})
