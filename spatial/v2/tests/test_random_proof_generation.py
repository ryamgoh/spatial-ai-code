import random

import pytest

from spatial.v2.generation import (
    ENTITY_NAMES,
    GenerationPolicy,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.possibility import construct_possibility
from spatial.v2.random_proof import (
    RandomProofPolicy,
    SemanticRule,
    TransitionPartition,
    constructively_supported_transitions,
)
from spatial.v2.random_proof_generation import (
    RandomProofConstructionError,
    construct_random_proof_candidate,
    lift_problem_with_random_path,
    possibility_backbone_length,
)
from spatial.v2.solver import SpatialSolverV2


@pytest.mark.parametrize("query_kind", ("direction", "which", "count"))
def test_constructs_certified_single_step_candidates(query_kind: str) -> None:
    policy = RandomProofPolicy(
        semantic_length=1,
        distractor_count=0,
        allowed_rules=(SemanticRule.MP,),
    )

    candidate = construct_random_proof_candidate(
        query_kind=query_kind,
        objects=ENTITY_NAMES[:16],
        policy=policy,
        seed=42,
    )

    assert candidate.complexity.semantic_length == 1
    assert ("modus-ponens",) in candidate.complexity.critical_signatures
    assert candidate.complexity.removable_premises == 0


def test_constructs_requested_ordered_path_and_exact_distractors() -> None:
    partition = TransitionPartition(
        frozenset(
            {
                (SemanticRule.IFF, SemanticRule.DJE),
                (SemanticRule.DJE, SemanticRule.MP),
            }
        ),
        frozenset(),
    )
    policy = RandomProofPolicy(
        semantic_length=3,
        distractor_count=2,
        allowed_rules=(SemanticRule.IFF, SemanticRule.DJE, SemanticRule.MP),
        transition_partition=partition,
    )

    candidate = construct_random_proof_candidate(
        query_kind="direction",
        objects=ENTITY_NAMES[:20],
        policy=policy,
        seed=17,
    )

    assert candidate.requested_path == (
        SemanticRule.IFF,
        SemanticRule.DJE,
        SemanticRule.MP,
    )
    assert candidate.complexity.semantic_length == 3
    assert candidate.complexity.removable_premises == 2


def test_infeasible_transition_is_rejected() -> None:
    policy = RandomProofPolicy(
        semantic_length=2,
        distractor_count=0,
        allowed_rules=(SemanticRule.TRC, SemanticRule.DN),
        transition_partition=TransitionPartition(
            frozenset({(SemanticRule.TRC, SemanticRule.DN)}), frozenset()
        ),
    )

    with pytest.raises(RandomProofConstructionError):
        construct_random_proof_candidate(
            query_kind="direction",
            objects=ENTITY_NAMES[:16],
            policy=policy,
            seed=3,
        )


def test_dje_backbone_length_avoids_recertifying_every_base_retry() -> None:
    assert possibility_backbone_length("direction", "dje", 4) == 1
    assert possibility_backbone_length("which", "dje", 4) == 0
    assert possibility_backbone_length("count", "dje", 4) == 4
    assert possibility_backbone_length("count", "csr", 4) is None


def test_standard_generator_renders_fixed_menu_randomized_single() -> None:
    sample = SpatialGeneratorV2(42).generate_randomized(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            answer_mode=AnswerMode.SINGLE,
            semantic_shape=SemanticShape.UNIQUE,
            num_entities=16,
            num_premises=15,
            menu_option_count=5,
        ),
        RandomProofPolicy(3, 0, (SemanticRule.MP,)),
    )

    assert len(sample.options) == 5
    assert sample.difficulty["proof_complexity"]["semantic_length"] == 3
    assert sample.difficulty["requested_semantic_length"] == 3


def test_possibility_skeleton_can_be_lifted_to_longer_randomized_proof() -> None:
    sample = SpatialGeneratorV2(43).generate_randomized(
        GenerationPolicy(
            query_kind=QueryKind.DIRECTION,
            answer_mode=AnswerMode.ALL_POSSIBLE,
            semantic_shape=SemanticShape.AMBIGUOUS,
            num_entities=16,
            num_premises=15,
            possibility_skeleton="csr",
            possibility_gold_size=3,
            ordinary_option_target=5,
            menu_option_count=5,
        ),
        RandomProofPolicy(6, 0, (SemanticRule.MP,)),
    )

    assert len(sample.options) == 5
    assert len(sample.analysis.possible_directions) == 3
    assert sample.difficulty["proof_complexity"]["semantic_length"] == 6


def test_randomized_lift_preserves_requested_selection_gold_size() -> None:
    sample = SpatialGeneratorV2(17209467846799471187).generate_randomized(
        GenerationPolicy(
            query_kind=QueryKind.WHICH,
            answer_mode=AnswerMode.ALL_POSSIBLE,
            semantic_shape=SemanticShape.AMBIGUOUS,
            ambiguity_size=2,
            possibility_skeleton="exclude",
            possibility_gold_size=2,
            possibility_candidate_count=5,
            num_entities=24,
            num_premises=23,
            ordinary_option_target=5,
            menu_option_count=5,
        ),
        RandomProofPolicy(
            2,
            0,
            forbidden_final_rules=(SemanticRule.TRC,),
            transition_partition=TransitionPartition(
                constructively_supported_transitions(), frozenset()
            ),
        ),
        max_attempts=2_000,
    )

    assert len(sample.analysis.possible_entities) == 2
    assert len(sample.menu_answer.letters) == 2


def test_proof_prefix_does_not_change_base_possibility_set() -> None:
    seed = 17209467846799471187
    rng = random.Random(seed)
    policy = GenerationPolicy(
        query_kind=QueryKind.WHICH,
        answer_mode=AnswerMode.ALL_POSSIBLE,
        semantic_shape=SemanticShape.AMBIGUOUS,
        ambiguity_size=2,
        possibility_skeleton="exclude",
        possibility_gold_size=2,
        possibility_candidate_count=5,
        num_entities=24,
        num_premises=23,
        ordinary_option_target=5,
        menu_option_count=5,
    )
    objects = tuple(sorted(rng.sample(ENTITY_NAMES, policy.num_entities)))
    base = construct_possibility(policy, objects, rng)
    lifted = lift_problem_with_random_path(
        base,
        RandomProofPolicy(
            2,
            0,
            forbidden_final_rules=(SemanticRule.TRC,),
            transition_partition=TransitionPartition(
                constructively_supported_transitions(), frozenset()
            ),
        ),
        seed=seed * 1_000_003,
    )
    solver = SpatialSolverV2()

    assert solver.analyze(lifted.problem).possible_entities == (
        solver.analyze(base).possible_entities
    )
