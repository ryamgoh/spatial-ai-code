"""Active atomic generation boundary; historical curricula have separate callers."""

import ast
import importlib
from dataclasses import fields
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spatial.v2.generate_all import WorkloadSpec, app, generate_workload
from spatial.v2.generation import (
    GenerationMode,
    GenerationPolicy,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
    _system_prompt,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.legacy.generation import GenerationPolicy as LegacyPolicy
from spatial.v2.legacy.generation import _system_prompt as legacy_system_prompt
from spatial.v2.solver import And, RelationConstraint


@pytest.mark.parametrize("kind", tuple(QueryKind))
@pytest.mark.parametrize("mode", ("world-first", "premise-first", "proof-template-first"))
def test_active_routes_emit_only_atomic_premises(kind, mode):
    controls = {}
    if mode == "proof-template-first":
        controls = (
            {"min_axis_depth": 2, "omit_direct_query_relation": True}
            if kind is QueryKind.DIRECTION
            else {"min_membership_depth": 2}
        )
    sample = SpatialGeneratorV2(seed=42).generate(
        GenerationPolicy(
            query_kind=kind,
            generation_mode=mode,
            semantic_shape="unique",
            num_entities=7,
            num_premises=7,
            **controls,
        ),
        max_attempts=2000,
    )
    assert isinstance(sample.problem.premise, And)
    assert all(
        isinstance(premise, RelationConstraint) and len(premise.allowed) == 1
        for premise in sample.problem.premise.operands
    )
    assert sample.analysis.consistent and not sample.analysis.error
    assert sample.certificate is not None
    assert sample.as_sft_row()["metadata"]["round_trip_verified"]


@pytest.mark.parametrize(
    "name,value",
    [
        ("boolean_shape", "modus-ponens"),
        ("composition", "mp-trc"),
        ("possibility_skeleton", "direction-opposites"),
        ("require_joint_count_dependency", True),
    ],
)
def test_retired_controls_are_not_active_policy_fields(name, value):
    assert name not in {field.name for field in fields(GenerationPolicy)}
    with pytest.raises(TypeError, match=name):
        GenerationPolicy(**{name: value})


def test_legacy_policy_cannot_bypass_atomic_boundary():
    with pytest.raises(TypeError, match="atomic GenerationPolicy"):
        SpatialGeneratorV2(seed=42).generate(LegacyPolicy())
    assert not hasattr(SpatialGeneratorV2, "generate_randomized")


def test_frozen_corr_prompt_helper_is_unchanged():
    assert _system_prompt(AnswerMode.SINGLE) == legacy_system_prompt(AnswerMode.SINGLE)


def test_active_cli_has_no_boolean_curriculum_options():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "boolean-shapes" not in result.output
    assert "joint-count-dependency" not in result.output
    result = CliRunner().invoke(
        app, ["--out", "unused.jsonl", "--boolean-shapes", "modus-ponens"]
    )
    assert result.exit_code != 0


def test_atomic_workload_round_trip(tmp_path):
    outputs = generate_workload(
        tmp_path / "atomic.jsonl",
        WorkloadSpec(
            samples_per_cell=2,
            query_kinds=(QueryKind.DIRECTION,),
            semantic_shapes=(SemanticShape.UNIQUE,),
            generation_mode=GenerationMode.WORLD_FIRST,
            test_split=0,
        ),
    )
    assert all(path.exists() for path in outputs)


def test_active_modules_do_not_import_legacy_curricula():
    root = Path(__file__).parents[1]
    for name in ("generation", "generate_all"):
        tree = ast.parse((root / f"{name}.py").read_text())
        imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        assert not any(".legacy" in module for module in imports)
    for name in ("random_proof", "random_proof_generation", "composition"):
        assert importlib.util.find_spec(f"spatial.v2.{name}") is None
