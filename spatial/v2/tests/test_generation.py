"""Contracts for the solver-validated V2 SpatialMap generator."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from spatial.v2.answer_certificates import (
    DirectionAnswerSetCertificate,
    DirectionEntailmentCertificate,
)
from spatial.v2.count_certificates import CountAnswerSetCertificate
from spatial.v2.generate_all import WorkloadSpec, app, generate_workload
from spatial.v2.generation import (
    BooleanShape,
    GenerationPolicy,
    GenerationProvenance,
    MenuCoverage,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
)
from spatial.v2.grading import AnswerMode, encode_menu_answer, resolve_answer
from spatial.v2.proofs import ProofRule
from spatial.v2.solver import (
    CountQuery,
    Direction,
    DirectionQuery,
    SpatialSolverV2,
    WhichQuery,
)
from spatial.v2.supervision_controls import (
    SupervisionArm,
    annotate_model_token_counts,
    build_supervision_variants,
)
from spatial.v2.symbolic_trace_codec import (
    parse_symbolic_training_trace,
    score_symbolic_training_trace,
)
from spatial.v2.text import SpatialTextAdapter
from spatial.v2.trace import TraceFormat
from spatial.v2.which_certificates import WhichAnswerSetCertificate


@pytest.mark.parametrize(
    ("query_kind", "query_type", "certificate_type", "domain_marker"),
    [
        (
            QueryKind.DIRECTION,
            DirectionQuery,
            DirectionAnswerSetCertificate,
            '"schema":"spatial-direction-trace-v2"',
        ),
        (
            QueryKind.WHICH,
            WhichQuery,
            WhichAnswerSetCertificate,
            '"schema":"spatial-which-trace-v1"',
        ),
        (
            QueryKind.COUNT,
            CountQuery,
            CountAnswerSetCertificate,
            '"schema":"spatial-count-trace-v2"',
        ),
    ],
)
@pytest.mark.parametrize("answer_mode", tuple(AnswerMode))
def test_generated_rows_round_trip_for_every_query_and_answer_mode(
    query_kind: QueryKind,
    query_type: type,
    certificate_type: type,
    domain_marker: str,
    answer_mode: AnswerMode,
) -> None:
    policy = GenerationPolicy(
        query_kind=query_kind,
        answer_mode=answer_mode,
        semantic_shape=SemanticShape.UNIQUE,
        menu_coverage=MenuCoverage.FULL,
        num_entities=5,
        num_premises=6,
        query_direction=(
            None if query_kind is QueryKind.DIRECTION else Direction.NORTH
        ),
        trace_format=TraceFormat.SYMBOLIC,
    )

    sample = SpatialGeneratorV2(seed=1701).generate(policy)
    row = sample.as_sft_row()
    parsed = SpatialTextAdapter().parse(row["messages"][1]["content"])
    checked_trace = parse_symbolic_training_trace(
        sample.problem,
        sample.trace,
        sample.menu_answer,
    )

    assert isinstance(parsed.problem.query, query_type)
    assert isinstance(sample.certificate, certificate_type)
    assert parsed.problem == sample.problem
    assert parsed.options == sample.options
    assert row["metadata"]["answer_mode"] == answer_mode.value
    assert row["metadata"]["round_trip_verified"] is True
    assert row["metadata"]["oracle_letters"] == sorted(sample.menu_answer.letters)
    assert checked_trace.decision.selected_letters == tuple(
        sorted(sample.menu_answer.letters)
    )
    assert row["messages"][2]["content"].endswith(
        "Answer: " + ", ".join(sorted(sample.menu_answer.letters))
    )
    assert '"schema":"spatial-answer-decision-v1"' in sample.trace
    assert domain_marker in sample.trace


def test_generator_supports_every_compass_direction_for_which_and_count() -> None:
    generator = SpatialGeneratorV2(seed=1703)

    for query_kind in (QueryKind.WHICH, QueryKind.COUNT):
        for direction in Direction:
            sample = generator.generate(
                GenerationPolicy(
                    query_kind=query_kind,
                    query_direction=direction,
                    semantic_shape=SemanticShape.UNIQUE,
                    num_entities=4,
                    num_premises=4,
                )
            )
            assert sample.problem.query.directions == frozenset({direction})


def test_generator_can_target_every_direction_answer() -> None:
    generator = SpatialGeneratorV2(seed=1717)

    for direction in Direction:
        sample = generator.generate(
            GenerationPolicy(
                query_kind=QueryKind.DIRECTION,
                target_direction=direction,
                semantic_shape=SemanticShape.UNIQUE,
                num_entities=5,
                num_premises=5,
            )
        )
        assert sample.analysis.possible_directions == (direction,)
        assert sample.as_sft_row()["metadata"]["target_direction"] == direction.value


def test_generated_rows_record_actual_construction_provenance() -> None:
    world_first = SpatialGeneratorV2(seed=1718).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            num_entities=3,
            num_premises=2,
        )
    )
    proof_template_first = SpatialGeneratorV2(seed=1719).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            num_entities=3,
            num_premises=2,
            omit_direct_query_relation=True,
            min_axis_depth=2,
            max_axis_depth=2,
        )
    )

    assert world_first.generation_provenance is GenerationProvenance.WORLD_FIRST
    assert (
        proof_template_first.generation_provenance
        is GenerationProvenance.PROOF_TEMPLATE_FIRST
    )
    assert world_first.as_sft_row()["metadata"]["generation_provenance"] == (
        "world-first"
    )
    assert (
        proof_template_first.as_sft_row()["metadata"]["generation_provenance"]
        == "proof-template-first"
    )


def test_structure_signature_is_direction_sensitive() -> None:
    def generated(seed: int, direction: Direction):
        return SpatialGeneratorV2(seed=seed).generate(
            GenerationPolicy(
                query_kind=QueryKind.DIRECTION,
                target_direction=direction,
                semantic_shape=SemanticShape.UNIQUE,
                num_entities=3,
                num_premises=2,
                omit_direct_query_relation=True,
                min_axis_depth=2,
                max_axis_depth=2,
            )
        )

    northeast = generated(1720, Direction.NORTHEAST)
    southwest = generated(1721, Direction.SOUTHWEST)

    assert northeast.problem.objects != southwest.problem.objects
    assert northeast.structure_signature != southwest.structure_signature
    assert len(northeast.structure_signature) == 64
    assert northeast.as_sft_row()["metadata"]["structure_signature"] == (
        northeast.structure_signature
    )


def test_symbolic_process_score_separates_reasoning_from_decision_errors() -> None:
    sample = SpatialGeneratorV2(seed=1723).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            answer_mode=AnswerMode.SINGLE,
            semantic_shape=SemanticShape.UNIQUE,
            trace_format=TraceFormat.SYMBOLIC,
            num_entities=3,
            num_premises=2,
        )
    )
    valid = score_symbolic_training_trace(
        sample.problem,
        sample.trace,
        sample.menu_answer,
    )
    reasoning, decision = sample.trace.splitlines()
    decision_payload = json.loads(decision)
    wrong = next(
        letter for letter in sample.options if letter not in sample.menu_answer.letters
    )
    decision_payload["select"] = [wrong]
    invalid = score_symbolic_training_trace(
        sample.problem,
        reasoning + "\n" + json.dumps(decision_payload),
        sample.menu_answer,
    )

    assert valid.fully_valid
    assert invalid.reasoning_valid
    assert not invalid.decision_valid
    assert not invalid.fully_valid
    assert invalid.error_stage == "decision"


def test_answer_only_and_corrupted_trace_controls_preserve_the_gold_answer() -> None:
    sample = SpatialGeneratorV2(seed=1724).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            answer_mode=AnswerMode.SINGLE,
            semantic_shape=SemanticShape.UNIQUE,
            trace_format=TraceFormat.SYMBOLIC,
            num_entities=3,
            num_premises=2,
        )
    )
    checked, answer_only, corrupted = build_supervision_variants(
        sample.as_sft_row(),
        (
            SupervisionArm.CHECKED_TRACE,
            SupervisionArm.ANSWER_ONLY,
            SupervisionArm.CORRUPTED_TRACE,
        ),
        problem=sample.problem,
        expected=sample.menu_answer,
    )
    expected_suffix = "Answer: " + ", ".join(sorted(sample.menu_answer.letters))

    assert checked["messages"][2]["content"].endswith(expected_suffix)
    assert answer_only["messages"][2]["content"] == expected_suffix
    assert corrupted["messages"][2]["content"].endswith(expected_suffix)
    assert "<think>" not in answer_only["messages"][2]["content"]
    corrupted_trace = (
        corrupted["messages"][2]["content"]
        .split("<think>\n", 1)[1]
        .split("\n</think>", 1)[0]
    )
    assert not score_symbolic_training_trace(
        sample.problem,
        corrupted_trace,
        sample.menu_answer,
    ).fully_valid
    assert checked["metadata"]["expected_process_valid"] is True
    assert corrupted["metadata"]["expected_process_valid"] is False
    tokenized = annotate_model_token_counts(
        (checked, answer_only, corrupted),
        lambda text: len(text.encode("utf-8")),
    )
    assert all(
        row["metadata"]["target_model_tokens"]
        == len(row["messages"][2]["content"].encode("utf-8"))
        for row in tokenized
    )


@pytest.mark.parametrize(
    ("shape", "rule", "minimum_uses"),
    [
        (BooleanShape.MODUS_PONENS, ProofRule.MODUS_PONENS, 1),
        (BooleanShape.IFF, ProofRule.IFF_ELIMINATION, 1),
        (BooleanShape.DOUBLE_NEGATION, ProofRule.DOUBLE_NEGATION, 1),
        (
            BooleanShape.DISJUNCTIVE_SYLLOGISM,
            ProofRule.DISJUNCTIVE_SYLLOGISM,
            1,
        ),
        (BooleanShape.CASE_SPLIT, ProofRule.CASE_SPLIT, 1),
        (BooleanShape.NESTED_CASE_SPLIT, ProofRule.CASE_SPLIT, 2),
    ],
)
@pytest.mark.parametrize("target_direction", list(Direction))
def test_generator_constructs_proof_first_boolean_curriculum(
    shape: BooleanShape,
    rule: ProofRule,
    minimum_uses: int,
    target_direction: Direction,
) -> None:
    sample = SpatialGeneratorV2(seed=1720).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            boolean_shape=shape,
            target_direction=target_direction,
            num_entities=6,
            num_premises=7,
        ),
        max_attempts=1,
    )

    possible = next(item for item in sample.certificate.candidates if item.possible)
    assert isinstance(possible.evidence, DirectionEntailmentCertificate)
    assert possible.direction is target_direction
    assert (
        sum(step.rule is rule for step in possible.evidence.proof.steps) >= minimum_uses
    )
    assert SpatialTextAdapter().parse(sample.prompt).problem == sample.problem
    assert sample.policy.boolean_shape is shape
    assert len(set(sample.problem.premise.operands)) == len(
        sample.problem.premise.operands
    )


def test_boolean_curriculum_rejects_ambiguous_semantics() -> None:
    with pytest.raises(ValueError, match="require unique"):
        GenerationPolicy(
            query_kind=QueryKind.WHICH,
            semantic_shape=SemanticShape.AMBIGUOUS,
            boolean_shape=BooleanShape.MODUS_PONENS,
        )


def test_boolean_curriculum_does_not_construct_premises_from_coordinates(
    monkeypatch,
) -> None:
    generator = SpatialGeneratorV2(seed=1720)

    def reject_world_first_generation(_objects):
        raise AssertionError("Boolean curriculum requested a privileged world")

    monkeypatch.setattr(generator, "_coordinates", reject_world_first_generation)

    sample = generator.generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            boolean_shape=BooleanShape.NESTED_CASE_SPLIT,
            num_entities=6,
            num_premises=7,
        ),
        max_attempts=1,
    )

    assert sample.analysis.possible_directions


@pytest.mark.parametrize("query_kind", (QueryKind.WHICH, QueryKind.COUNT))
@pytest.mark.parametrize(
    "shape",
    tuple(shape for shape in BooleanShape if shape is not BooleanShape.ATOMIC),
)
def test_boolean_curriculum_supports_membership_query_families(
    query_kind: QueryKind,
    shape: BooleanShape,
) -> None:
    sample = SpatialGeneratorV2(seed=1722).generate(
        GenerationPolicy(
            query_kind=query_kind,
            semantic_shape=SemanticShape.UNIQUE,
            boolean_shape=shape,
            query_direction=Direction.EAST,
            trace_format=TraceFormat.SYMBOLIC,
            num_entities=6,
            num_premises=7,
        ),
        max_attempts=1,
    )

    if query_kind is QueryKind.WHICH:
        assert sample.analysis.possible_entities == (
            sample.problem.query.candidates[0],
        )
        assert sample.analysis.entailed_entities == (
            sample.problem.query.candidates[0],
        )
        assert '"schema":"spatial-which-trace-v1"' in sample.trace
    else:
        assert sample.analysis.possible_counts == (1,)
        assert '"schema":"spatial-count-trace-v2"' in sample.trace


def test_workload_rejects_boolean_cross_product_with_incompatible_cells() -> None:
    with pytest.raises(ValueError, match="require unique"):
        WorkloadSpec(boolean_shapes=(BooleanShape.MODUS_PONENS,))


def test_direction_policy_enforces_transitive_axis_depth_without_direct_fact() -> None:
    sample = SpatialGeneratorV2(seed=1721).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            num_entities=7,
            num_premises=8,
            omit_direct_query_relation=True,
            min_axis_depth=2,
            max_axis_depth=4,
        ),
        max_attempts=2_000,
    )

    difficulty = sample.difficulty
    assert difficulty["direct_query_relation"] is False
    assert 2 <= difficulty["x_depth"] <= 4
    assert 2 <= difficulty["y_depth"] <= 4
    assert difficulty["supporting_premise_indices"]
    assert sample.as_sft_row()["metadata"]["difficulty"] == difficulty


def test_direction_policy_can_require_independent_axis_support() -> None:
    sample = SpatialGeneratorV2(seed=1723).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            num_entities=8,
            num_premises=10,
            omit_direct_query_relation=True,
            min_axis_depth=2,
            require_independent_axes=True,
        ),
        max_attempts=5_000,
    )

    assert sample.difficulty["axes_independent"] is True


def test_controlled_direction_proofs_are_constructed_without_rejection() -> None:
    generator = SpatialGeneratorV2(seed=1724)

    for direction in Direction:
        sample = generator.generate(
            GenerationPolicy(
                query_kind=QueryKind.DIRECTION,
                target_direction=direction,
                semantic_shape=SemanticShape.UNIQUE,
                num_entities=6,
                num_premises=7,
                omit_direct_query_relation=True,
                min_axis_depth=2,
                max_axis_depth=4,
                distractor_premises=5,
            )
        )
        assert sample.attempt == 0
        assert sample.rejection_counts == {}


@pytest.mark.parametrize(
    ("query_kind", "size", "seed"),
    [
        (QueryKind.DIRECTION, 2, 1725),
        (QueryKind.COUNT, 3, 1727),
    ],
)
def test_policy_can_target_exact_ambiguity_size(
    query_kind: QueryKind,
    size: int,
    seed: int,
) -> None:
    sample = SpatialGeneratorV2(seed=seed).generate(
        GenerationPolicy(
            query_kind=query_kind,
            semantic_shape=SemanticShape.AMBIGUOUS,
            ambiguity_size=size,
            num_entities=6,
            num_premises=5,
        ),
        max_attempts=5_000,
    )

    assert sample.difficulty["possibility_count"] == size


@pytest.mark.parametrize(
    ("query_kind", "seed"),
    [(QueryKind.WHICH, 1729), (QueryKind.COUNT, 1731)],
)
def test_membership_queries_can_require_transitive_positive_proofs(
    query_kind: QueryKind,
    seed: int,
) -> None:
    sample = SpatialGeneratorV2(seed=seed).generate(
        GenerationPolicy(
            query_kind=query_kind,
            semantic_shape=SemanticShape.UNIQUE,
            num_entities=7,
            num_premises=12,
            omit_direct_query_relation=True,
            min_membership_depth=2,
            max_membership_depth=4,
        ),
        max_attempts=5_000,
    )

    difficulty = sample.difficulty
    assert difficulty["direct_query_relation"] is False
    assert difficulty["membership_proofs"]
    assert all(
        2 <= proof[axis] <= 4
        for proof in difficulty["membership_proofs"]
        for axis in ("x_depth", "y_depth")
    )


@pytest.mark.parametrize("query_kind", (QueryKind.WHICH, QueryKind.COUNT))
def test_controlled_membership_proofs_are_constructed_without_rejection(
    query_kind: QueryKind,
) -> None:
    generator = SpatialGeneratorV2(seed=1732)

    for direction in Direction:
        sample = generator.generate(
            GenerationPolicy(
                query_kind=query_kind,
                query_direction=direction,
                semantic_shape=SemanticShape.UNIQUE,
                num_entities=7,
                num_premises=10,
                omit_direct_query_relation=True,
                min_membership_depth=2,
                max_membership_depth=4,
            )
        )
        assert sample.attempt == 0
        assert sample.rejection_counts == {}


def test_policy_can_require_exact_number_of_non_supporting_premises() -> None:
    sample = SpatialGeneratorV2(seed=1733).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            num_entities=7,
            num_premises=8,
            omit_direct_query_relation=True,
            min_axis_depth=2,
            distractor_premises=3,
        ),
        max_attempts=5_000,
    )

    assert sample.difficulty["num_distractor_premises"] == 3


def test_trace_variants_share_one_base_problem_and_have_distinct_ids() -> None:
    sample = SpatialGeneratorV2(seed=1735).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            trace_format=TraceFormat.NATURAL,
        )
    )
    variant = sample.with_trace(TraceFormat.SYMBOLIC)

    first = sample.as_sft_row()
    second = variant.as_sft_row()
    assert first["metadata"]["base_id"] == second["metadata"]["base_id"]
    assert first["id"] != second["id"]
    assert first["messages"][1] == second["messages"][1]
    assert first["metadata"]["oracle_letters"] == second["metadata"]["oracle_letters"]
    assert first["messages"][2] != second["messages"][2]


def test_answer_mode_variants_share_one_solved_problem() -> None:
    generator = SpatialGeneratorV2(seed=1745)
    sample = generator.generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            answer_mode=AnswerMode.SINGLE,
            semantic_shape=SemanticShape.AMBIGUOUS,
            ambiguity_size=2,
            num_entities=6,
            num_premises=5,
        )
    )
    complete = generator.with_answer_mode(
        sample,
        AnswerMode.ALL_POSSIBLE,
        MenuCoverage.FULL,
    )
    visible = generator.with_answer_mode(
        sample,
        AnswerMode.VISIBLE_POSSIBLE,
        MenuCoverage.PARTIAL,
    )

    rows = [variant.as_sft_row() for variant in (sample, complete, visible)]
    assert len({row["metadata"]["base_id"] for row in rows}) == 1
    assert len({row["id"] for row in rows}) == 3
    assert all(variant.problem == sample.problem for variant in (complete, visible))
    assert complete.resolution.values == sample.analysis.possible_directions
    assert 0 < len(visible.menu_answer.letters) < len(complete.menu_answer.letters)


@pytest.mark.parametrize("trace_format", tuple(TraceFormat))
def test_training_rows_never_contain_coordinate_witnesses(
    trace_format: TraceFormat,
) -> None:
    sample = SpatialGeneratorV2(seed=1705).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.UNIQUE,
            trace_format=trace_format,
        )
    )

    serialized = json.dumps(sample.as_sft_row())
    audit = sample.audit_metadata

    assert "coordinates" not in serialized.lower()
    assert "generation_witness" in audit
    assert audit["answer_certificate"]["type"].endswith("AnswerSetCertificate")
    assert set(audit["generation_witness"]) == set(sample.problem.objects)


def test_partial_all_possible_menu_maps_to_cannot_be_determined() -> None:
    sample = SpatialGeneratorV2(seed=1707).generate(
        GenerationPolicy(
            query_kind=QueryKind.COUNT,
            answer_mode=AnswerMode.ALL_POSSIBLE,
            semantic_shape=SemanticShape.AMBIGUOUS,
            menu_coverage=MenuCoverage.PARTIAL,
            num_entities=5,
            num_premises=4,
        )
    )

    assert sample.menu_answer.status == "incomplete-menu"
    assert "Cannot be determined" in sample.options.values()
    assert sample.menu_answer.letters


def test_zero_visible_possible_menu_maps_to_none_of_options() -> None:
    sample = SpatialGeneratorV2(seed=1709).generate(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            answer_mode=AnswerMode.VISIBLE_POSSIBLE,
            semantic_shape=SemanticShape.UNIQUE,
            menu_coverage=MenuCoverage.ZERO,
        )
    )

    assert sample.menu_answer.status == "none-of-options"
    assert "None of the Options" in sample.options.values()
    assert "The menu result is none-of-options" in sample.trace


def test_map_scoped_which_keeps_hidden_objects_in_candidate_universe() -> None:
    prompt = (
        "Consider a map with multiple locations:\n\n"
        "A is to the Northeast of R. B is to the Northwest of R. "
        "C is to the Southwest of R.\n\n"
        "Question: Select the complete set of possible answers. "
        "Which object in the map is in the Northeast of R? "
        "Available options: A. B, B. C, C. None of the Options"
    )

    parsed = SpatialTextAdapter().parse(prompt)
    analysis = SpatialSolverV2().analyze(parsed.problem)
    answer = encode_menu_answer(
        resolve_answer(analysis, AnswerMode.ALL_POSSIBLE),
        parsed.options,
    )

    assert isinstance(parsed.problem.query, WhichQuery)
    assert parsed.problem.query.candidates == ("A", "B", "C")
    assert analysis.possible_entities == ("A",)
    assert answer.status == "none-of-options"
    assert answer.raw == "C"


def test_generator_is_deterministic_per_seed_and_policy() -> None:
    policy = GenerationPolicy(
        query_kind=QueryKind.WHICH,
        answer_mode=AnswerMode.SINGLE,
        semantic_shape=SemanticShape.UNIQUE,
        query_direction=Direction.SOUTHWEST,
    )

    first = SpatialGeneratorV2(seed=1711).generate(policy).as_sft_row()
    second = SpatialGeneratorV2(seed=1711).generate(policy).as_sft_row()

    assert first == second


def test_policy_rejects_impossible_partial_singleton_contract() -> None:
    with pytest.raises(ValueError, match="partial menu coverage"):
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            answer_mode=AnswerMode.SINGLE,
            menu_coverage=MenuCoverage.PARTIAL,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"query_kind": QueryKind.COUNT, "min_axis_depth": 2},
        {"query_kind": QueryKind.DIRECTION, "min_membership_depth": 2},
        {
            "query_kind": QueryKind.DIRECTION,
            "semantic_shape": SemanticShape.AMBIGUOUS,
            "max_axis_depth": 3,
        },
        {
            "query_kind": QueryKind.WHICH,
            "semantic_shape": SemanticShape.UNIQUE,
            "distractor_premises": 1,
        },
    ],
)
def test_policy_rejects_incompatible_difficulty_controls(overrides: dict) -> None:
    with pytest.raises(ValueError):
        GenerationPolicy(**overrides)


def test_policy_rejects_impossible_ambiguity_size_before_sampling() -> None:
    with pytest.raises(ValueError, match="at most 8"):
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            semantic_shape=SemanticShape.AMBIGUOUS,
            ambiguity_size=9,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "semantic_shape": SemanticShape.UNIQUE,
            "num_entities": 5,
            "num_premises": 4,
            "min_axis_depth": 5,
        },
        {
            "semantic_shape": SemanticShape.UNIQUE,
            "num_entities": 3,
            "num_premises": 3,
            "omit_direct_query_relation": True,
        },
        {
            "semantic_shape": SemanticShape.UNIQUE,
            "num_entities": 6,
            "num_premises": 7,
            "min_axis_depth": 3,
            "require_independent_axes": True,
            "distractor_premises": 2,
        },
    ],
)
def test_policy_rejects_structurally_impossible_proof_budgets(
    overrides: dict,
) -> None:
    with pytest.raises(ValueError, match="cannot satisfy"):
        GenerationPolicy(**overrides)


def test_generator_records_rejection_reasons_before_acceptance() -> None:
    policy = GenerationPolicy(
        query_kind=QueryKind.WHICH,
        semantic_shape=SemanticShape.AMBIGUOUS,
        ambiguity_size=2,
        num_entities=6,
        num_premises=5,
    )

    sample = SpatialGeneratorV2(seed=1745).generate(policy, max_attempts=10)

    assert sample.attempt > 0
    assert sum(sample.rejection_counts.values()) == sample.attempt
    assert sample.as_sft_row()["metadata"]["rejection_counts"] == {
        key: sample.rejection_counts[key] for key in sorted(sample.rejection_counts)
    }


def test_generator_failure_reports_rejection_breakdown() -> None:
    policy = GenerationPolicy(
        query_kind=QueryKind.WHICH,
        semantic_shape=SemanticShape.AMBIGUOUS,
        ambiguity_size=2,
        num_entities=6,
        num_premises=5,
    )

    with pytest.raises(RuntimeError, match=r"rejections: .+=1"):
        SpatialGeneratorV2(seed=1745).generate(policy, max_attempts=1)


def test_balanced_workload_writes_unique_verified_rows_without_audit(tmp_path) -> None:
    train_path, dev_path, test_path = generate_workload(
        tmp_path / "workload.jsonl",
        WorkloadSpec(
            samples_per_cell=2,
            query_kinds=(QueryKind.DIRECTION, QueryKind.COUNT),
            answer_modes=(AnswerMode.SINGLE,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            menu_coverages=(MenuCoverage.FULL,),
            trace_formats=(TraceFormat.NATURAL, TraceFormat.SYMBOLIC),
            test_split=0.25,
            seed=1713,
        ),
    )

    rows = [
        json.loads(line)
        for path in (train_path, dev_path, test_path)
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 8
    assert len({row["id"] for row in rows}) == 8
    assert {row["metadata"]["query_kind"] for row in rows} == {
        "direction",
        "count",
    }
    assert {row["metadata"]["trace_format"] for row in rows} == {
        "natural",
        "symbolic",
    }
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["metadata"]["base_id"], []).append(row)
    assert len(grouped) == 4
    assert all(len(group) == 2 for group in grouped.values())
    assert all(
        len({row["messages"][1]["content"] for row in group}) == 1
        for group in grouped.values()
    )
    train_base_ids = {
        json.loads(line)["metadata"]["base_id"]
        for line in train_path.read_text().splitlines()
    }
    test_base_ids = {
        json.loads(line)["metadata"]["base_id"]
        for line in test_path.read_text().splitlines()
    }
    assert train_base_ids.isdisjoint(test_base_ids)
    assert all("audit" not in row for row in rows)


def test_workload_emits_each_requested_boolean_shape(tmp_path) -> None:
    train_path, _, _ = generate_workload(
        tmp_path / "boolean.jsonl",
        WorkloadSpec(
            samples_per_cell=1,
            query_kinds=(QueryKind.DIRECTION,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            trace_formats=(TraceFormat.SYMBOLIC,),
            boolean_shapes=(
                BooleanShape.MODUS_PONENS,
                BooleanShape.NESTED_CASE_SPLIT,
            ),
            target_directions=(Direction.NORTH,),
            num_entities=6,
            num_premises=7,
            test_split=0,
        ),
    )

    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert {row["metadata"]["boolean_shape"] for row in rows} == {
        "modus-ponens",
        "nested-case-split",
    }
    assert all(
        '"schema":"spatial-direction-trace-v2"' in row["messages"][2]["content"]
        for row in rows
    )


def test_cli_rejects_unknown_policy_dimensions(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--out",
            str(tmp_path / "bad.jsonl"),
            "--query-kinds",
            "direction,teleport",
        ],
    )

    assert result.exit_code != 0
    assert "unknown query kinds: teleport" in result.output


def test_workload_spec_rejects_empty_dimensions() -> None:
    with pytest.raises(ValueError, match="empty workload dimensions: query kind"):
        WorkloadSpec(query_kinds=())


def test_workload_requires_explicit_replace_for_existing_outputs(tmp_path) -> None:
    output = tmp_path / "replace.jsonl"
    spec = WorkloadSpec(
        samples_per_cell=1,
        query_kinds=(QueryKind.DIRECTION,),
        semantic_shapes=(SemanticShape.UNIQUE,),
        trace_formats=(TraceFormat.NATURAL,),
        target_directions=(Direction.NORTH,),
        test_split=0,
    )
    generate_workload(output, spec)

    with pytest.raises(FileExistsError, match="use replace"):
        generate_workload(output, spec)

    generate_workload(output, replace(spec, replace=True))


def test_workload_can_balance_which_queries_across_directions(tmp_path) -> None:
    train_path, _, _ = generate_workload(
        tmp_path / "directions.jsonl",
        WorkloadSpec(
            samples_per_cell=1,
            query_kinds=(QueryKind.WHICH,),
            answer_modes=(AnswerMode.SINGLE,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            menu_coverages=(MenuCoverage.FULL,),
            trace_formats=(TraceFormat.NATURAL,),
            query_directions=(Direction.NORTH, Direction.SOUTHWEST),
            test_split=0,
            seed=1715,
        ),
    )

    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert {row["metadata"]["query_direction"] for row in rows} == {
        "North",
        "Southwest",
    }


def test_workload_can_balance_direction_answers(tmp_path) -> None:
    train_path, _, _ = generate_workload(
        tmp_path / "answers.jsonl",
        WorkloadSpec(
            samples_per_cell=1,
            query_kinds=(QueryKind.DIRECTION,),
            answer_modes=(AnswerMode.SINGLE,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            menu_coverages=(MenuCoverage.FULL,),
            trace_formats=(TraceFormat.NATURAL,),
            target_directions=(Direction.EAST, Direction.NORTHWEST),
            test_split=0,
            seed=1741,
        ),
    )

    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert {row["metadata"]["target_direction"] for row in rows} == {
        "East",
        "Northwest",
    }


@pytest.mark.parametrize(
    "coverage,special",
    [
        (MenuCoverage.FULL, "Cannot be determined"),
        (MenuCoverage.ZERO, "None of the Options"),
    ],
)
def test_special_menu_options_participate_in_seeded_permutation(coverage, special):
    from spatial.v2.solver import DirectionAnalysis

    analysis = DirectionAnalysis(
        consistent=True,
        target="A",
        reference="B",
        possible_directions=(
            (Direction.NORTH, Direction.SOUTH)
            if coverage is MenuCoverage.FULL
            else (Direction.NORTH,)
        ),
    )
    policy = GenerationPolicy(menu_coverage=coverage)
    resolution = resolve_answer(analysis, policy.answer_mode)
    positions = set()
    for seed in range(40):
        first = SpatialGeneratorV2(seed)._menu(policy, analysis, resolution)
        assert first == SpatialGeneratorV2(seed)._menu(policy, analysis, resolution)
        positions.add(
            next(letter for letter, value in first.items() if value == special)
        )
        assert encode_menu_answer(resolution, first).letters == {
            next(letter for letter, value in first.items() if value == special)
        }
    assert positions == set("ABCDE")


@pytest.mark.parametrize("query_kind", tuple(QueryKind))
def test_premise_first_samples_without_world_or_proof_template(monkeypatch, query_kind):
    generator = SpatialGeneratorV2(712)

    def forbidden(*args):
        raise AssertionError("premise-first requested a privileged construction")

    for name in (
        "_coordinates",
        "_boolean_problem",
        "_controlled_direction_problem",
        "_controlled_membership_problem",
    ):
        monkeypatch.setattr(generator, name, forbidden)
    sample = generator.generate(
        GenerationPolicy(
            generation_mode="premise-first",
            query_kind=query_kind,
            num_entities=4,
            num_premises=3,
            trace_format=TraceFormat.SYMBOLIC,
        )
    )
    assert sample.generation_provenance is GenerationProvenance.PREMISE_FIRST
    parsed = SpatialTextAdapter().parse(sample.prompt)
    assert parsed.problem == sample.problem
    assert (
        encode_menu_answer(
            resolve_answer(
                SpatialSolverV2().analyze(parsed.problem), sample.policy.answer_mode
            ),
            parsed.options,
        )
        == sample.menu_answer
    )
    assert sample.certificate


def test_structure_key_ignores_renaming_and_commutative_orders():
    from spatial.v2.solver import (
        And,
        Iff,
        Implies,
        Not,
        Or,
        SpatialProblem,
        direction_constraint,
    )
    from spatial.v2.structure import canonical_structure_profile

    a = direction_constraint("A", "R", Direction.NORTH)
    b = direction_constraint("B", "C", Direction.EAST)
    c = direction_constraint("C", "R", Direction.SOUTHWEST)
    problem = SpatialProblem(
        ("A", "B", "C", "R"),
        And((Iff(a, b), Or((Not(b), c)), Implies(c, a))),
        WhichQuery(frozenset({Direction.NORTH}), "R", ("A", "B", "C")),
    )
    renamed_a = direction_constraint("Zulu", "Anchor", Direction.NORTH)
    renamed_b = direction_constraint("Beta", "Alpha", Direction.EAST)
    renamed_c = direction_constraint("Alpha", "Anchor", Direction.SOUTHWEST)
    renamed = SpatialProblem(
        ("Anchor", "Alpha", "Beta", "Zulu"),
        And(
            (
                Implies(renamed_c, renamed_a),
                Or((renamed_c, Not(renamed_b))),
                Iff(renamed_b, renamed_a),
            )
        ),
        WhichQuery(frozenset({Direction.NORTH}), "Anchor", ("Beta", "Zulu", "Alpha")),
    )
    assert canonical_structure_profile(problem) == canonical_structure_profile(renamed)
    reversed_implication = replace(
        problem, premise=And((Iff(a, b), Or((Not(b), c)), Implies(a, c)))
    )
    assert canonical_structure_profile(problem) != canonical_structure_profile(
        reversed_implication
    )


def test_structure_key_marks_bounded_coarse_fallback_and_keeps_invariance():
    from spatial.v2.solver import And, SpatialProblem, direction_constraint
    from spatial.v2.structure import canonical_structure_profile

    def star(names):
        reference, *candidates = names
        return SpatialProblem(
            tuple(names),
            And(
                tuple(
                    direction_constraint(name, reference, Direction.EAST)
                    for name in candidates
                )
            ),
            WhichQuery(frozenset({Direction.EAST}), reference, tuple(candidates)),
        )

    first = star(("R", "A", "B", "C", "D"))
    second = star(("Center", "Zulu", "Yankee", "Xray", "Whiskey"))
    exact = canonical_structure_profile(first)
    coarse = canonical_structure_profile(first, max_permutations=1)
    assert exact["canonicalization"] == "exact"
    assert coarse["canonicalization"] == "coarse-refinement"
    assert coarse == canonical_structure_profile(second, max_permutations=1)


@pytest.mark.parametrize("shape", tuple(BooleanShape))
def test_premise_first_boolean_families_replay_without_target_templates(shape):
    sample = SpatialGeneratorV2(713).generate(
        GenerationPolicy(
            generation_mode="premise-first",
            boolean_shape=shape,
            num_entities=4,
            num_premises=3,
            trace_format=TraceFormat.SYMBOLIC,
        ),
        max_attempts=100,
    )
    assert sample.generation_provenance is GenerationProvenance.PREMISE_FIRST
    assert score_symbolic_training_trace(
        sample.problem, sample.trace, sample.menu_answer
    ).fully_valid
