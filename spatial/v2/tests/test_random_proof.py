import random
from collections import Counter

import pytest

from spatial.v2.random_proof import (
    RandomProofPolicy,
    SemanticRule,
    TransitionPartition,
    count_rule_paths,
    deterministic_transition_partition,
    exponential_quotas,
    sample_rule_path,
)


def test_exponential_quotas_match_frozen_study_allocations() -> None:
    assert exponential_quotas(range(1, 6), 120, 1.15) == {
        1: 18,
        2: 20,
        3: 24,
        4: 27,
        5: 31,
    }
    assert exponential_quotas(range(6, 9), 60, 1.15) == {6: 17, 7: 20, 8: 23}
    assert exponential_quotas(range(0, 9), 300, 1.10) == {
        0: 22,
        1: 24,
        2: 27,
        3: 30,
        4: 32,
        5: 36,
        6: 39,
        7: 43,
        8: 47,
    }


def test_completion_sampler_is_uniform_over_complete_paths() -> None:
    rules = (SemanticRule.MP, SemanticRule.IFF, SemanticRule.TRC)
    seen = frozenset(
        {
            (SemanticRule.MP, SemanticRule.IFF),
            (SemanticRule.MP, SemanticRule.TRC),
            (SemanticRule.IFF, SemanticRule.TRC),
        }
    )
    partition = TransitionPartition(seen, frozenset())
    assert count_rule_paths(2, rules, partition) == 3

    rng = random.Random(42)
    counts = Counter(sample_rule_path(rng, 2, rules, partition) for _ in range(6000))

    assert set(counts) == {
        (SemanticRule.MP, SemanticRule.IFF),
        (SemanticRule.MP, SemanticRule.TRC),
        (SemanticRule.IFF, SemanticRule.TRC),
    }
    assert all(1800 <= count <= 2200 for count in counts.values())


def test_sampler_can_require_exactly_one_heldout_transition() -> None:
    rules = (SemanticRule.MP, SemanticRule.IFF, SemanticRule.TRC)
    partition = TransitionPartition(
        frozenset(
            {
                (SemanticRule.MP, SemanticRule.IFF),
                (SemanticRule.IFF, SemanticRule.TRC),
                (SemanticRule.TRC, SemanticRule.MP),
            }
        ),
        frozenset({(SemanticRule.MP, SemanticRule.TRC)}),
    )

    path = sample_rule_path(
        random.Random(7),
        4,
        rules,
        partition,
        require_exactly_one_heldout=True,
    )

    transitions = list(zip(path, path[1:]))
    assert sum(transition in partition.heldout for transition in transitions) == 1
    assert all(transition in partition.eligible for transition in transitions)


def test_random_proof_policy_validates_requested_bounds() -> None:
    assert RandomProofPolicy(8, 8).semantic_length == 8
    with pytest.raises(ValueError, match="semantic_length"):
        RandomProofPolicy(9, 0)
    with pytest.raises(ValueError, match="held-out"):
        RandomProofPolicy(1, 0, require_heldout_transition=True)


def test_deterministic_partition_retains_seen_l8_and_heldout_edges() -> None:
    partition = deterministic_transition_partition(42)

    assert len(partition.heldout) >= 2
    assert count_rule_paths(8, tuple(SemanticRule), partition) > 0
    assert all(
        any(first is rule for first, _ in partition.seen) for rule in SemanticRule
    )
