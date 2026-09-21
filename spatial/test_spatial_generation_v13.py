"""Public-interface tests for the deep v13 generation module."""

from __future__ import annotations

import json
import random
import re

import pytest

from spatial_generation_v13 import (
    CycleAxes,
    CyclePlacement,
    CycleSpec,
    CycleTopology,
    DepthRange,
    DistractorPolicy,
    DistractorSpec,
    GenerationCell,
    GenerationError,
    GenerationSpec,
    RelationMode,
    SemanticSubtype,
    SpatialGenerator,
    StructuralConstraints,
    TraceFormat,
    WorldConsistency,
    generate_dataset,
    render_trace,
)
from spatial_solver_v13 import SpatialSolverV13


def user_text(example) -> str:
    return next(
        message["content"]
        for message in example.messages
        if message["role"] == "user"
    )


def assistant_text(example) -> str:
    return next(
        message["content"]
        for message in example.messages
        if message["role"] == "assistant"
    )


def test_generate_accepts_one_coherent_spec() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.MIXED,
        constraints=StructuralConstraints(require_independent_axes=True),
        num_entities=10,
        num_relations=15,
    )

    row = SpatialGenerator().generate(spec, random.Random(1321))
    solved = SpatialSolverV13().solve_and_analyze(user_text(row))

    assert row.oracle_option == solved.grade.raw
    assert row.difficulty == solved.structure
    assert row.difficulty["semantic_subtype"] == "dir-1"
    assert row.difficulty["relation_mix"] == "mixed"
    assert row.difficulty["axes_independent"] is True
    assert row.to_row()["difficulty_schema_version"] == 4


@pytest.mark.parametrize("relation_mode", tuple(RelationMode))
@pytest.mark.parametrize("semantic_subtype", tuple(SemanticSubtype))
def test_every_semantic_subtype_and_relation_mode_round_trips(
    semantic_subtype: SemanticSubtype, relation_mode: RelationMode
) -> None:
    spec = GenerationSpec(
        semantic_subtype=semantic_subtype,
        relation_mode=relation_mode,
        num_entities=10,
        num_relations=15,
    )

    example = SpatialGenerator().generate(spec, random.Random(1313))
    solved = SpatialSolverV13().solve_and_analyze(user_text(example))

    assert example.oracle_option == solved.grade.raw
    assert example.difficulty == solved.structure
    assert example.difficulty["semantic_subtype"] == semantic_subtype.value
    assert example.difficulty["relation_mix"] == {
        RelationMode.DIAGONAL: "diagonal-only",
        RelationMode.CARDINAL: "cardinal-only",
        RelationMode.MIXED: "mixed",
    }[relation_mode]


def test_invalid_spec_fails_before_generation() -> None:
    with pytest.raises(ValueError, match="independent axes.*dir-1"):
        GenerationSpec(
            semantic_subtype=SemanticSubtype.COUNT_1,
            relation_mode=RelationMode.MIXED,
            constraints=StructuralConstraints(require_independent_axes=True),
        )


def test_generation_spec_can_use_an_explicit_larger_entity_pool() -> None:
    entity_pool = tuple(f"Location {index}" for index in range(30))
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.CARDINAL,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange.exact(10),
            y_depth=DepthRange.exact(10),
            distractors=DistractorSpec(policy=DistractorPolicy.QUERY_BRANCH, count=10),
        ),
        num_entities=30,
        num_relations=30,
        entity_pool=entity_pool,
    )

    example = SpatialGenerator().generate(spec, random.Random(13700))

    assert example.difficulty["num_entities"] == 30
    assert example.difficulty["x_depth"] == 10
    assert example.difficulty["y_depth"] == 10
    assert example.difficulty["num_query_branch_distractors"] == 10


def test_generation_error_reports_rejection_reasons() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.MIXED,
        constraints=StructuralConstraints(require_independent_axes=True),
        num_entities=2,
        num_relations=1,
        max_attempts=3,
    )

    with pytest.raises(GenerationError) as caught:
        SpatialGenerator().generate(spec, random.Random(7))

    assert caught.value.attempts == 3
    assert caught.value.rejections
    assert sum(caught.value.rejections.values()) == 3
    assert set(caught.value.rejections) <= {
        "no_qualifying_query",
        "solver_rejected",
        "wrong_semantic_subtype",
        "wrong_relation_mode",
        "axes_not_independent",
    }


def test_generate_dataset_accepts_explicit_cells(tmp_path) -> None:
    cells = [
        GenerationCell(
            name="cardinal-dir-2",
            spec=GenerationSpec(
                semantic_subtype=SemanticSubtype.DIR_2,
                relation_mode=RelationMode.CARDINAL,
            ),
            count=5,
        ),
        GenerationCell(
            name="mixed-count-omit",
            spec=GenerationSpec(
                semantic_subtype=SemanticSubtype.COUNT_OMIT,
                relation_mode=RelationMode.MIXED,
            ),
            count=5,
        ),
    ]

    train_path, test_path = generate_dataset(
        cells,
        output_file=tmp_path / "planned.jsonl",
        test_fraction=0.2,
        seed=13,
    )

    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    test = [json.loads(line) for line in test_path.read_text().splitlines()]
    assert len(train) == 8
    assert len(test) == 2
    assert {row["generation_cell"] for row in train + test} == {
        "cardinal-dir-2",
        "mixed-count-omit",
    }
    assert {row["generation_cell"] for row in test} == {
        "cardinal-dir-2",
        "mixed-count-omit",
    }


def test_depth_range_validates_bounds() -> None:
    assert DepthRange.exact(3).contains(3)
    assert not DepthRange(3, 5).contains(2)
    assert DepthRange(3, 5).contains(5)
    with pytest.raises(ValueError, match="minimum.*positive"):
        DepthRange(0, 2)
    with pytest.raises(ValueError, match="maximum.*minimum"):
        DepthRange(4, 3)


def test_depth_constraints_require_dir1() -> None:
    with pytest.raises(ValueError, match="proof depth.*dir-1"):
        GenerationSpec(
            semantic_subtype=SemanticSubtype.DIR_2,
            relation_mode=RelationMode.CARDINAL,
            constraints=StructuralConstraints(x_depth=DepthRange.exact(2)),
        )
    with pytest.raises(ValueError, match="proof depth.*independent axes"):
        GenerationSpec(
            semantic_subtype=SemanticSubtype.DIR_1,
            relation_mode=RelationMode.CARDINAL,
            constraints=StructuralConstraints(
                x_depth=DepthRange.exact(2),
                y_depth=DepthRange.exact(2),
            ),
        )


@pytest.mark.parametrize(
    "relation_mode,x_depth,y_depth",
    [
        (RelationMode.CARDINAL, 3, 4),
        (RelationMode.MIXED, 4, 3),
    ],
)
def test_generate_enforces_independent_axis_depths(
    relation_mode: RelationMode, x_depth: int, y_depth: int
) -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=relation_mode,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange.exact(x_depth),
            y_depth=DepthRange.exact(y_depth),
        ),
        num_entities=10,
        num_relations=12,
    )

    example = SpatialGenerator().generate(spec, random.Random(1401))

    assert example.difficulty["x_depth"] == x_depth
    assert example.difficulty["y_depth"] == y_depth
    assert example.difficulty["axes_independent"] is True
    assert example.difficulty["shared_supporting_statements"] == []


def test_generate_accepts_depth_ranges_not_only_exact_values() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.CARDINAL,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange(2, 3),
            y_depth=DepthRange(3, 4),
        ),
        num_entities=9,
        num_relations=10,
    )

    example = SpatialGenerator().generate(spec, random.Random(1403))

    assert 2 <= example.difficulty["x_depth"] <= 3
    assert 3 <= example.difficulty["y_depth"] <= 4


def test_clean_depth_generation_does_not_add_an_extra_filler_relation() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.CARDINAL,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange.exact(3),
            y_depth=DepthRange.exact(4),
        ),
        num_entities=10,
        num_relations=7,
    )

    example = SpatialGenerator().generate(spec, random.Random(1404))

    assert example.difficulty["num_relations"] == 7
    assert example.difficulty["num_distractor_relations"] == 0


def test_trace_initialization_contains_exactly_prompt_visible_entities() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.CARDINAL,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange.exact(1),
            y_depth=DepthRange.exact(1),
        ),
        num_entities=8,
        num_relations=10,
    )

    example = SpatialGenerator().generate(spec, random.Random(101))
    solved = SpatialSolverV13().solve_and_analyze(user_text(example))
    match = re.search(r"\*\*Entities Detected\*\*: (.+)", assistant_text(example))

    assert match
    trace_entities = {part.strip() for part in match.group(1).split(",")}
    assert trace_entities == set(solved.objects)


def test_delta_trace_keeps_updates_compact_and_renders_final_state_once() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.MIXED,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange.exact(4),
            y_depth=DepthRange.exact(4),
            distractors=DistractorSpec(
                policy=DistractorPolicy.QUERY_BRANCH, count=3
            ),
        ),
        num_entities=11,
        num_relations=11,
    )
    example = SpatialGenerator().generate(spec, random.Random(13777))
    solved = SpatialSolverV13().solve_and_analyze(user_text(example))

    full = render_trace(solved, TraceFormat.FULL_STATE)
    delta = render_trace(solved, TraceFormat.DELTA_STATE)

    assert delta.endswith(f"Answer: {solved.grade.pretty}")
    assert delta.count("**Final X-State**:") == 1
    assert delta.count("**Final Y-State**:") == 1
    assert delta.count("**Conflict Status**:") == len(solved.relations)
    assert "**X-State**:" not in delta
    assert "**Y-State**:" not in delta
    assert len(delta) < len(full)


def test_trace_final_deduction_shows_solver_verified_paths_and_composition() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.MIXED,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange.exact(3),
            y_depth=DepthRange.exact(4),
        ),
        num_entities=10,
        num_relations=12,
    )

    example = SpatialGenerator().generate(spec, random.Random(102))
    solved = SpatialSolverV13().solve_and_analyze(user_text(example))
    trace = assistant_text(example)
    x_path = " < ".join(solved.structure["x_path"])
    y_path = " < ".join(solved.structure["y_path"])

    assert f"**Target**: {solved.structure['target']}" in trace
    assert f"**Reference**: {solved.structure['reference']}" in trace
    assert f"**X-Proof**: {x_path}" in trace
    assert f"**Y-Proof**: {y_path}" in trace
    assert "**X-Conclusion**:" in trace
    assert "**Y-Conclusion**:" in trace
    assert "**Composition**:" in trace
    assert trace.index("**X-Proof**:") < trace.index("**Options**:")
    assert trace.index("**Y-Proof**:") < trace.index("**Options**:")


def test_trace_does_not_claim_a_path_for_an_unknown_axis() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_2,
        relation_mode=RelationMode.CARDINAL,
        num_entities=8,
        num_relations=10,
    )

    example = SpatialGenerator().generate(spec, random.Random(171))
    solved = SpatialSolverV13().solve_and_analyze(user_text(example))
    trace = assistant_text(example)

    assert (solved.structure["x_path"] is None) != (solved.structure["y_path"] is None)
    if solved.structure["x_path"] is None:
        assert "**X-Proof**: none (no ordering path)" in trace
    else:
        assert "**Y-Proof**: none (no ordering path)" in trace


def test_which_trace_shows_solver_verified_entity_proofs() -> None:
    example = SpatialGenerator().generate(
        GenerationSpec(
            semantic_subtype=SemanticSubtype.WHICH_2,
            relation_mode=RelationMode.MIXED,
        ),
        random.Random(177),
    )
    solved = SpatialSolverV13().solve_and_analyze(user_text(example))
    trace = assistant_text(example)

    assert f"**Reference**: {solved.structure['query_reference']}" in trace
    assert f"**Direction Query**: {solved.structure['query_direction']}" in trace
    assert "**Proven Entities**:" in trace
    for proof in solved.structure["proven_entity_proofs"]:
        for axis in solved.structure["required_axes"]:
            assert (
                f"**{proof['entity']} {axis.upper()}-Proof**: "
                f"{' < '.join(proof[f'{axis}_path'])}"
            ) in trace


def test_count_trace_shows_solver_verified_entity_set_and_count() -> None:
    example = SpatialGenerator().generate(
        GenerationSpec(
            semantic_subtype=SemanticSubtype.COUNT_OMIT,
            relation_mode=RelationMode.MIXED,
        ),
        random.Random(178),
    )
    solved = SpatialSolverV13().solve_and_analyze(user_text(example))
    trace = assistant_text(example)

    names = solved.structure["proven_entities"]
    rendered_names = ", ".join(names) if names else "none"
    assert f"**Reference**: {solved.structure['query_reference']}" in trace
    assert f"**Direction Query**: {solved.structure['query_direction']}" in trace
    assert f"**Proven Entities**: {rendered_names}" in trace
    assert f"**Count**: {len(names)}" in trace


@pytest.mark.parametrize(
    "policy,expected_key",
    [
        (DistractorPolicy.DISCONNECTED, "num_disconnected_distractors"),
        (DistractorPolicy.QUERY_BRANCH, "num_query_branch_distractors"),
    ],
)
def test_generate_exact_solver_classified_distractors_without_depth_drift(
    policy: DistractorPolicy, expected_key: str
) -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.DIR_1,
        relation_mode=RelationMode.MIXED,
        constraints=StructuralConstraints(
            require_independent_axes=True,
            x_depth=DepthRange.exact(3),
            y_depth=DepthRange.exact(4),
            distractors=DistractorSpec(policy=policy, count=3),
        ),
        num_entities=12 if policy is DistractorPolicy.DISCONNECTED else 10,
        num_relations=10,
    )

    example = SpatialGenerator().generate(spec, random.Random(1501))
    difficulty = example.difficulty

    assert difficulty["x_depth"] == 3
    assert difficulty["y_depth"] == 4
    assert difficulty["axes_independent"] is True
    assert difficulty["num_distractor_relations"] == 3
    assert difficulty[expected_key] == 3
    other_key = (
        "num_query_branch_distractors"
        if expected_key == "num_disconnected_distractors"
        else "num_disconnected_distractors"
    )
    assert difficulty[other_key] == 0


def test_distractor_constraints_require_depth_controlled_dir1() -> None:
    with pytest.raises(ValueError, match="distractors.*proof depth"):
        GenerationSpec(
            semantic_subtype=SemanticSubtype.DIR_1,
            relation_mode=RelationMode.MIXED,
            constraints=StructuralConstraints(
                require_independent_axes=True,
                distractors=DistractorSpec(
                    policy=DistractorPolicy.DISCONNECTED, count=2
                ),
            ),
        )


@pytest.mark.parametrize(
    "semantic_subtype,expected_family",
    [
        (SemanticSubtype.DIR_1, "direction"),
        (SemanticSubtype.WHICH_2, "which"),
        (SemanticSubtype.COUNT_1, "count"),
    ],
)
@pytest.mark.parametrize(
    "axes,topology,placement,length",
    [
        (CycleAxes.X, CycleTopology.DIRECT, CyclePlacement.QUERY_CONNECTED, 2),
        (CycleAxes.Y, CycleTopology.INDIRECT, CyclePlacement.DISCONNECTED, 3),
        (CycleAxes.BOTH, CycleTopology.INDIRECT, CyclePlacement.QUERY_CONNECTED, 4),
    ],
)
def test_global_cycle_generation_applies_to_every_question_family(
    semantic_subtype: SemanticSubtype,
    expected_family: str,
    axes: CycleAxes,
    topology: CycleTopology,
    placement: CyclePlacement,
    length: int,
) -> None:
    cycle = CycleSpec(
        axes=axes,
        topology=topology,
        placement=placement,
        length=length,
    )
    cycle_relations = length
    fresh_cycle_entities = length - (
        1 if placement is CyclePlacement.QUERY_CONNECTED else 0
    )
    spec = GenerationSpec(
        semantic_subtype=semantic_subtype,
        relation_mode=RelationMode.MIXED,
        cycle=cycle,
        num_entities=8 + fresh_cycle_entities,
        num_relations=10 + cycle_relations,
    )

    example = SpatialGenerator().generate(spec, random.Random(1801))
    difficulty = example.difficulty

    assert difficulty["world_consistency"] == "inconsistent"
    assert difficulty["question_family"] == expected_family
    assert difficulty["semantic_subtype"] == semantic_subtype.value
    assert difficulty["cycle_axes"] == (
        ["x", "y"] if axes is CycleAxes.BOTH else [axes.value]
    )
    assert difficulty["cycle_topology"] == topology.value
    assert difficulty["cycle_placement"] == placement.value
    assert example.oracle_option in {"A", "B", "C", "D", "E"}
    trace = assistant_text(example)
    assert "### Consistency Check" in trace
    assert "The complete map is inconsistent" in trace
    assert "### Final Deduction" not in trace
    assert "**Composition**:" not in trace
    row = example.to_row()
    assert row["semantic_subtype"] == semantic_subtype.value
    assert "base_semantic_subtype" not in row
    assert row["difficulty_schema_version"] == 4


def test_cycle_spec_validates_topology_and_length() -> None:
    with pytest.raises(ValueError, match="direct cycle.*length 2"):
        CycleSpec(
            axes=CycleAxes.X,
            topology=CycleTopology.DIRECT,
            placement=CyclePlacement.DISCONNECTED,
            length=3,
        )
    with pytest.raises(ValueError, match="indirect cycle.*at least 3"):
        CycleSpec(
            axes=CycleAxes.Y,
            topology=CycleTopology.INDIRECT,
            placement=CyclePlacement.DISCONNECTED,
            length=2,
        )


def test_cycle_is_not_a_semantic_subtype() -> None:
    assert not any(
        subtype.value.endswith("-cycle") for subtype in SemanticSubtype
    )


def test_near_cycle_control_preserves_base_answer_and_has_no_cycle() -> None:
    spec = GenerationSpec(
        semantic_subtype=SemanticSubtype.WHICH_2,
        relation_mode=RelationMode.MIXED,
        cycle=CycleSpec(
            axes=CycleAxes.Y,
            topology=CycleTopology.INDIRECT,
            placement=CyclePlacement.DISCONNECTED,
            length=4,
            world_consistency=WorldConsistency.CONSISTENT,
        ),
        num_entities=13,
        num_relations=14,
    )

    example = SpatialGenerator().generate(spec, random.Random(1807))

    assert example.difficulty["world_consistency"] == "consistent"
    assert example.difficulty["semantic_subtype"] == "which-2"
    assert example.difficulty["cycle_axes"] == []
    assert example.difficulty["cycle_topology"] == "none"
    assert example.difficulty["cycle_placement"] == "none"
    assert "### Consistency Check" not in assistant_text(example)
