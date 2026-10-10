"""Public contracts for V2 workload manifests and leakage validation."""

from __future__ import annotations

import json

import pytest

from spatial.v2.generate_all import WorkloadSpec, generate_workload
from spatial.v2.generation import BooleanShape, QueryKind, SemanticShape
from spatial.v2.grading import AnswerMode
from spatial.v2.supervision_controls import SupervisionArm
from spatial.v2.trace import TraceFormat
from spatial.v2.workload_manifest import (
    SplitStrategy,
    build_workload_manifest,
    write_workload,
)


def _row(
    row_id: str,
    base_id: str,
    prompt: str,
    trace_format: str,
    answer_mode: str = "single",
    structure_signature: str = "default-structure",
) -> dict:
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Answer: A"},
        ],
        "metadata": {
            "base_id": base_id,
            "generation_provenance": "premise-first",
            "structure_signature": structure_signature,
            "query_kind": "direction",
            "query_direction": None,
            "answer_mode": answer_mode,
            "semantic_shape": "unique",
            "menu_coverage": "full",
            "trace_format": trace_format,
            "menu_status": "exact",
            "possible_values": ["North"],
            "oracle_letters": ["A"],
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
            TraceFormat.NATURAL,
            TraceFormat.SYMBOLIC,
        ),
    )

    assert manifest["total_rows"] == 4
    assert manifest["base_problems"] == 2
    assert manifest["splits"] == {"train": 2, "dev": 0, "test": 2}
    assert manifest["trace_distributions"]["trace_format"] == {
        "natural": 2,
        "symbolic": 2,
    }
    assert manifest["base_distributions"]["query_kind"] == {"direction": 2}
    assert manifest["base_distributions"]["generation_provenance"] == {
        "premise-first": 2
    }
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


def test_structural_split_keeps_each_signature_on_one_side(tmp_path) -> None:
    groups = [
        [_row("a-1", "a-1", "prompt a1", "natural", structure_signature="a")],
        [_row("a-2", "a-2", "prompt a2", "natural", structure_signature="a")],
        [_row("b-1", "b-1", "prompt b1", "natural", structure_signature="b")],
        [_row("b-2", "b-2", "prompt b2", "natural", structure_signature="b")],
    ]

    train_path, _, test_path, manifest_path = write_workload(
        tmp_path / "structural.jsonl",
        groups,
        test_split=0.5,
        seed=9,
        split_strategy=SplitStrategy.STRUCTURAL,
        expected_trace_variants=(TraceFormat.NATURAL,),
        manifest_metadata={},
    )

    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    test = [json.loads(line) for line in test_path.read_text().splitlines()]
    train_signatures = {row["metadata"]["structure_signature"] for row in train}
    test_signatures = {row["metadata"]["structure_signature"] for row in test}
    manifest = json.loads(manifest_path.read_text())

    assert train_signatures.isdisjoint(test_signatures)
    assert len(train) == len(test) == 2
    assert manifest["validation"]["split_strategy"] == "structural"
    assert manifest["validation"]["structural_overlap"] == []


def test_structural_manifest_rejects_signature_leakage() -> None:
    train = [_row("train", "train", "train prompt", "natural", structure_signature="x")]
    test = [_row("test", "test", "test prompt", "natural", structure_signature="x")]

    with pytest.raises(ValueError, match="cross-split structure signatures"):
        build_workload_manifest(
            train,
            test,
            split_strategy=SplitStrategy.STRUCTURAL,
        )


def test_manifest_rejects_incomplete_answer_trace_product() -> None:
    rows = [_row("single-natural", "base-1", "prompt", "natural")]

    with pytest.raises(ValueError, match="incomplete answer/trace variants"):
        build_workload_manifest(
            rows,
            [],
            expected_trace_variants=(TraceFormat.NATURAL,),
            expected_variants_by_base={
                "base-1": {
                    (AnswerMode.SINGLE, "full", TraceFormat.NATURAL, "checked-trace"),
                    (
                        AnswerMode.ALL_POSSIBLE,
                        "full",
                        TraceFormat.NATURAL,
                        "checked-trace",
                    ),
                }
            },
        )


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


def test_manifest_reports_exact_model_token_matching_multipliers() -> None:
    checked = _row("checked", "checked", "checked prompt", "natural")
    answer_only = _row("answer", "answer", "answer prompt", "natural")
    checked["metadata"].update(
        supervision_arm="checked-trace",
        target_characters=100,
        target_whitespace_tokens=20,
        target_model_tokens=80,
    )
    answer_only["metadata"].update(
        supervision_arm="answer-only",
        target_characters=20,
        target_whitespace_tokens=4,
        target_model_tokens=16,
    )

    manifest = build_workload_manifest([checked, answer_only], [])

    assert (
        manifest["supervision_volume"]["checked-trace"]["model_token_multiplier_to_max"]
        == 1
    )
    assert (
        manifest["supervision_volume"]["answer-only"]["model_token_multiplier_to_max"]
        == 5
    )


def test_generator_writes_a_valid_manifest_next_to_splits(tmp_path) -> None:
    train_path, _, test_path = generate_workload(
        tmp_path / "manifested.jsonl",
        WorkloadSpec(
            samples_per_cell=2,
            query_kinds=(QueryKind.DIRECTION,),
            answer_modes=(AnswerMode.SINGLE,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            trace_formats=(TraceFormat.NATURAL, TraceFormat.SYMBOLIC),
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


def test_generated_structural_workload_holds_out_proof_shapes(tmp_path) -> None:
    train_path, _, test_path = generate_workload(
        tmp_path / "structural-generated.jsonl",
        WorkloadSpec(
            samples_per_cell=1,
            query_kinds=(QueryKind.DIRECTION,),
            answer_modes=(AnswerMode.SINGLE,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            trace_formats=(TraceFormat.SYMBOLIC,),
            boolean_shapes=(BooleanShape.MODUS_PONENS, BooleanShape.CASE_SPLIT),
            num_entities=4,
            num_premises=3,
            test_split=0.5,
            split_strategy=SplitStrategy.STRUCTURAL,
            seed=1741,
        ),
    )

    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    test = [json.loads(line) for line in test_path.read_text().splitlines()]
    train_signatures = {row["metadata"]["structure_signature"] for row in train}
    test_signatures = {row["metadata"]["structure_signature"] for row in test}
    manifest = json.loads((tmp_path / "structural-generated_manifest.json").read_text())

    assert train and test
    assert train_signatures.isdisjoint(test_signatures)
    assert manifest["validation"]["split_strategy"] == "structural"


def test_workload_emits_paired_control_arms_and_volume_report(tmp_path) -> None:
    train_path, _, _ = generate_workload(
        tmp_path / "controls.jsonl",
        WorkloadSpec(
            samples_per_cell=1,
            query_kinds=(QueryKind.DIRECTION,),
            answer_modes=(AnswerMode.SINGLE,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            trace_formats=(TraceFormat.NATURAL, TraceFormat.SYMBOLIC),
            supervision_arms=(
                SupervisionArm.CHECKED_TRACE,
                SupervisionArm.ANSWER_ONLY,
                SupervisionArm.CORRUPTED_TRACE,
            ),
            boolean_shapes=(BooleanShape.MODUS_PONENS,),
            num_entities=4,
            num_premises=3,
            test_split=0,
            seed=1742,
        ),
    )

    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    manifest = json.loads((tmp_path / "controls_manifest.json").read_text())
    arms = [row["metadata"]["supervision_arm"] for row in rows]

    assert arms.count("checked-trace") == 2
    assert arms.count("corrupted-trace") == 1
    assert arms.count("answer-only") == 1
    assert manifest["supervision_volume"]["answer-only"]["rows"] == 1
    assert (
        manifest["supervision_volume"]["answer-only"]["target_characters"]
        < manifest["supervision_volume"]["checked-trace"]["target_characters"]
    )
    assert (
        manifest["supervision_volume"]["answer-only"]["character_multiplier_to_max"] > 1
    )


def test_development_split_is_checked_for_leakage() -> None:
    train = [_row("train", "same-base", "train prompt", "natural")]
    dev = [_row("dev", "same-base", "dev prompt", "natural")]
    with pytest.raises(ValueError, match="cross-split base IDs"):
        build_workload_manifest(train, [], dev_rows=dev)

    dev = [_row("dev", "dev", "train prompt", "natural")]
    with pytest.raises(ValueError, match="cross-split prompts"):
        build_workload_manifest(train, [], dev_rows=dev)


def test_three_way_structural_split_keeps_clusters_together(tmp_path) -> None:
    groups = [
        [
            _row(
                f"{s}-{i}",
                f"{s}-{i}",
                f"prompt {s} {i}",
                "natural",
                structure_signature=s,
            )
        ]
        for s in ("a", "b", "c", "d", "e")
        for i in range(2)
    ]
    paths = write_workload(
        tmp_path / "three-way.jsonl",
        groups,
        test_split=0.2,
        dev_split=0.2,
        seed=7,
        split_strategy="structural",
        expected_trace_variants=("natural",),
        manifest_metadata={},
    )
    signature_sets = [
        {
            json.loads(line)["metadata"]["structure_signature"]
            for line in path.read_text().splitlines()
        }
        for path in paths[:3]
    ]
    assert all(signature_sets)
    assert all(
        not a & b for i, a in enumerate(signature_sets) for b in signature_sets[i + 1 :]
    )
    assert json.loads(paths[3].read_text())["base_splits"] == {
        "train": 6,
        "dev": 2,
        "test": 2,
    }


def test_requested_nonempty_split_never_silently_rounds_to_zero(tmp_path) -> None:
    groups = [[_row("a", "a", "a", "natural")]]
    with pytest.raises(ValueError, match="not enough"):
        write_workload(
            tmp_path / "tiny.jsonl",
            groups,
            test_split=0.2,
            dev_split=0.2,
            seed=1,
            expected_trace_variants=("natural",),
            manifest_metadata={},
        )
    assert not list(tmp_path.iterdir())


def test_predeclared_holdout_cells_route_only_to_their_split(tmp_path) -> None:
    groups = [
        [_row(s, s, s, "natural", structure_signature=s)]
        for s in ("atomic", "implication", "case-split")
    ]
    for group, cell in zip(
        groups, ("atomic", "implication", "case-split"), strict=True
    ):
        group[0]["metadata"]["matrix_cell"] = cell
    paths = write_workload(
        tmp_path / "holdout.jsonl",
        groups,
        test_split=0.2,
        dev_split=0.2,
        seed=1,
        split_strategy="holdout",
        holdout_cells={"dev": ["implication"], "test": ["case-split"]},
        expected_trace_variants=("natural",),
        manifest_metadata={},
    )
    assert [
        json.loads(p.read_text())["metadata"]["matrix_cell"] for p in paths[:3]
    ] == ["atomic", "implication", "case-split"]


def test_predeclared_holdout_rejects_unseen_or_overlapping_cells(tmp_path) -> None:
    groups = [
        [_row(s, s, s, "natural", structure_signature=s)] for s in ("a", "b", "c")
    ]
    for group in groups:
        group[0]["metadata"]["matrix_cell"] = group[0]["id"]
    for holdouts in (
        {"dev": ["b"], "test": ["missing"]},
        {"dev": ["b"], "test": ["b"]},
    ):
        with pytest.raises(ValueError, match="holdout"):
            write_workload(
                tmp_path / "bad.jsonl",
                groups,
                test_split=0.2,
                dev_split=0.2,
                seed=1,
                split_strategy="holdout",
                holdout_cells=holdouts,
                expected_trace_variants=("natural",),
                manifest_metadata={},
            )
    assert not list(tmp_path.iterdir())


def test_manifest_counts_answer_positions_once_per_menu() -> None:
    rows = [_row(f"b-{fmt}", "b", "prompt", fmt) for fmt in ("natural", "symbolic")]
    for row in rows:
        row["metadata"].update(oracle_letters=["C"], matrix_cell="ambiguous")
    manifest = build_workload_manifest(rows, [])
    assert manifest["answer_positions_by_cell"] == {"ambiguous": {"C": 1}}


def test_manifest_rejects_missing_control_variant() -> None:
    row = _row("b-natural", "b", "prompt", "natural")
    row["metadata"]["supervision_arm"] = "checked-trace"
    with pytest.raises(ValueError, match="incomplete answer/trace variants"):
        build_workload_manifest(
            [row],
            [],
            expected_variants_by_base={
                "b": {
                    ("single", "full", "natural", "checked-trace"),
                    ("single", "full", "natural", "answer-only"),
                }
            },
        )


def test_manifest_metadata_cannot_overwrite_validation(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot overwrite"):
        write_workload(
            tmp_path / "forged.jsonl",
            [[_row("b", "b", "prompt", "natural")]],
            test_split=0,
            seed=1,
            expected_trace_variants=("natural",),
            manifest_metadata={"validation": {"status": "unchecked"}},
        )
    assert not list(tmp_path.iterdir())


def test_duplicate_variant_cannot_inflate_supervision_volume() -> None:
    rows = [
        _row(row_id, "b", "same prompt", "natural") for row_id in ("first", "second")
    ]
    with pytest.raises(ValueError, match="duplicate answer/trace/control variant"):
        build_workload_manifest(rows, [])


@pytest.mark.parametrize(
    "field,value", [("structure_signature", "changed"), ("possible_values", ["South"])]
)
def test_paired_variants_must_share_semantic_identity(field, value) -> None:
    rows = [_row(f"b-{fmt}", "b", "prompt", fmt) for fmt in ("natural", "symbolic")]
    rows[1]["metadata"][field] = value
    with pytest.raises(ValueError, match="paired variants disagree"):
        build_workload_manifest(rows, [])


def test_paired_variants_must_share_menu_gold() -> None:
    rows = [_row(f"b-{fmt}", "b", "prompt", fmt) for fmt in ("natural", "symbolic")]
    rows[0]["metadata"]["oracle_letters"] = ["A"]
    rows[1]["metadata"]["oracle_letters"] = ["B"]
    with pytest.raises(
        ValueError, match="paired variants disagree on prompt or answer"
    ):
        build_workload_manifest(rows, [])


@pytest.mark.parametrize(
    "field,value",
    [
        ("matrix_cell", "changed"),
        ("query_direction", "South"),
        ("target_direction", "West"),
        ("difficulty", {"x_depth": 999}),
        ("num_entities", 99),
        ("num_premises", 99),
        ("structure_profile", {"schema": "changed"}),
        ("seed", 99),
        ("sample_index", 99),
        ("attempt", 99),
        ("rejection_counts", {"changed": 99}),
    ],
)
def test_paired_reporting_metadata_cannot_depend_on_row_order(field, value):
    rows = [_row(f"b-{fmt}", "b", "prompt", fmt) for fmt in ("natural", "symbolic")]
    rows[1]["metadata"][field] = value
    with pytest.raises(ValueError, match="paired variants disagree"):
        build_workload_manifest(rows, [])


@pytest.mark.parametrize("letters", [None, [], ["A", "A"], ["B", "A"], ["AB"], ["a"]])
def test_manifest_rejects_missing_or_malformed_gold(letters):
    row = _row("b", "b", "prompt", "natural")
    row["metadata"]["oracle_letters"] = letters
    with pytest.raises(ValueError, match="invalid oracle_letters"):
        build_workload_manifest([row], [])


def test_paired_admitted_prompts_must_match():
    rows = [_row(f"b-{fmt}", "b", "prompt", fmt) for fmt in ("natural", "symbolic")]
    rows[0]["metadata"]["evaluation_prompt"] = "templated prompt"
    rows[1]["metadata"]["evaluation_prompt"] = "different templated prompt"
    with pytest.raises(
        ValueError, match="paired variants disagree on prompt or answer"
    ):
        build_workload_manifest(rows, [])


def test_holdout_fraction_flags_must_match_declared_cells(tmp_path):
    groups = [
        [_row(s, s, s, "natural", structure_signature=s)] for s in ("a", "b", "c")
    ]
    for group in groups:
        group[0]["metadata"]["matrix_cell"] = group[0]["id"]
    with pytest.raises(ValueError, match="holdout cells and requested"):
        write_workload(
            tmp_path / "bad.jsonl",
            groups,
            dev_split=0,
            test_split=0.2,
            seed=1,
            split_strategy="holdout",
            holdout_cells={"dev": ["b"], "test": ["c"]},
            expected_trace_variants=("natural",),
            manifest_metadata={},
        )


def test_manifest_reports_exact_variants_and_split_token_volume():
    train = [_row(f"b-{fmt}", "b", "prompt", fmt) for fmt in ("natural", "symbolic")]
    for row, tokens in zip(train, (10, 30), strict=True):
        row["metadata"].update(
            supervision_arm="checked-trace",
            target_characters=tokens,
            target_whitespace_tokens=1,
            target_model_tokens=tokens,
        )
    test = [_row("test", "test", "test prompt", "natural")]
    test[0]["metadata"].update(
        supervision_arm="checked-trace",
        target_characters=5,
        target_whitespace_tokens=1,
        target_model_tokens=5,
    )
    manifest = build_workload_manifest(train, test)
    volumes = manifest["split_supervision_volume_by_variant"]
    assert (
        volumes["train"]["single/full/natural/checked-trace"]["target_model_tokens"]
        == 10
    )
    assert (
        volumes["train"]["single/full/natural/checked-trace"][
            "model_token_multiplier_to_max"
        ]
        == 3
    )
    assert (
        volumes["train"]["single/full/symbolic/checked-trace"]["target_model_tokens"]
        == 30
    )
    assert (
        volumes["test"]["single/full/natural/checked-trace"]["target_model_tokens"] == 5
    )
    assert volumes["dev"] == {}
    assert (
        manifest["supervision_volume_by_variant"]["single/full/natural/checked-trace"][
            "target_model_tokens"
        ]
        == 15
    )


def test_manifest_distinguishes_answer_sets_from_marginal_positions():
    row = _row("b", "b", "prompt", "natural", answer_mode="all-possible")
    row["metadata"]["oracle_letters"] = ["A", "C"]
    manifest = build_workload_manifest([row], [])
    assert manifest["answer_positions_by_cell"] == {"all": {"A,C": 1}}
    assert manifest["marginal_answer_positions_by_cell"] == {"all": {"A": 1, "C": 1}}
