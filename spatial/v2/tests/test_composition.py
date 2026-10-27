from dataclasses import replace

import pytest

from spatial.v2.legacy.composition import COMPOSITIONS, assess_composition
from spatial.v2.legacy.generation import GenerationPolicy, SpatialGeneratorV2
from spatial.v2.proofs import ProofConstructionError
from spatial.v2.solver import And, DirectionQuery, SpatialSolverV2, direction_constraint
from spatial.v2.symbolic_trace_codec import score_symbolic_training_trace


@pytest.mark.parametrize("name", COMPOSITIONS)
def test_composition_requires_replayed_order_and_semantic_dependence(name):
    kind, _ = COMPOSITIONS[name]
    sample = SpatialGeneratorV2(1911).generate(
        GenerationPolicy(
            query_kind=kind,
            semantic_shape="unique",
            composition=name,
            num_entities=7 if name == "mp-trc-mp" else 6,
            num_premises=12 if name == "mp-trc-mp" else 10,
            trace_format="symbolic",
        ),
        max_attempts=1,
    )
    assert sample.difficulty["composition"]["intervention_premise_indices"]
    assert score_symbolic_training_trace(
        sample.problem, sample.trace, sample.menu_answer
    ).fully_valid


def test_direct_shortcut_is_rejected_as_composition():
    sample = SpatialGeneratorV2(42).generate(
        GenerationPolicy(
            semantic_shape="unique",
            composition="mp-trc",
            num_entities=6,
            num_premises=10,
        )
    )
    query = sample.problem.query
    assert isinstance(query, DirectionQuery)
    shortcut = direction_constraint(
        query.target, query.reference, sample.analysis.possible_directions[0]
    )
    problem = replace(
        sample.problem, premise=And((*sample.problem.premise.operands, shortcut))
    )
    with pytest.raises(ProofConstructionError):
        assess_composition(problem, "mp-trc", SpatialSolverV2())


def test_composition_rejects_unknown_family_and_world_mode():
    with pytest.raises(ValueError, match="unknown composition"):
        GenerationPolicy(composition="unknown")
    with pytest.raises(ValueError, match="mode"):
        GenerationPolicy(
            composition="mp-trc", semantic_shape="unique", generation_mode="world-first"
        )


def test_depth_filler_topologies_vary_without_shortening_query_proof():
    generator = SpatialGeneratorV2(617)
    signatures = set()
    for _ in range(15):
        sample = generator.generate(
            GenerationPolicy(
                semantic_shape="unique",
                min_axis_depth=3,
                max_axis_depth=3,
                omit_direct_query_relation=True,
                num_entities=9,
                num_premises=11,
            )
        )
        assert sample.difficulty["x_depth"] == sample.difficulty["y_depth"] == 3
        signatures.add(sample.structure_signature)
    assert len(signatures) > 9


@pytest.mark.parametrize(
    "name",
    ["mp-trc", "iff-trc", "trc-mp", "dje-trc", "mp-trc-mp", "mp-dje-trc", "trc-mp-trc"],
)
@pytest.mark.parametrize("kind", ["direction", "which", "count"])
def test_balanced_hybrid_families_replay_across_query_contracts(name, kind):
    sample = SpatialGeneratorV2(1911).generate(
        GenerationPolicy(
            query_kind=kind,
            semantic_shape="unique",
            composition=name,
            num_entities=7 if name == "mp-trc-mp" else 6,
            num_premises=12 if name == "mp-trc-mp" else 10,
            trace_format="symbolic",
        ),
        max_attempts=1,
    )
    assert score_symbolic_training_trace(
        sample.problem, sample.trace, sample.menu_answer
    ).fully_valid
    evidence = sample.difficulty["composition"]
    assert len(evidence["linked_steps"]) == len(evidence["required_rules"])


def test_spatial_antecedent_recomposition_rejects_corrupted_axis_evidence():
    from spatial.v2.proofs import (
        ProofCheckError,
        ProofRule,
        build_direction_proof,
        check_direction_proof,
    )

    sample = SpatialGeneratorV2(77).generate(
        GenerationPolicy(
            query_kind="direction",
            semantic_shape="unique",
            composition="trc-mp",
            num_entities=6,
            num_premises=10,
        )
    )
    proof = build_direction_proof(sample.problem)
    bridge = next(
        step for step in proof.steps if step.rule is ProofRule.ATOM_RECOMPOSITION
    )
    bad = replace(bridge, inputs=(bridge.inputs[0], bridge.inputs[0]))
    damaged = replace(
        proof, steps=tuple(bad if s.id == bridge.id else s for s in proof.steps)
    )
    with pytest.raises(ProofCheckError):
        check_direction_proof(damaged)
