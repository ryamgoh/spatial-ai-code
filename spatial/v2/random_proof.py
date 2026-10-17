"""Randomized proof-path policies and unbiased bounded sequence sampling."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Iterable


class SemanticRule(str, Enum):
    MP = "MP"
    IFF = "IFF"
    DJE = "DJE"
    CSR = "CSR"
    DN = "DN"
    TRC = "TRC"


Transition = tuple[SemanticRule, SemanticRule]

PROOF_RULE_LABELS = {
    SemanticRule.MP: "modus-ponens",
    SemanticRule.IFF: "iff-elimination",
    SemanticRule.DJE: "disjunctive-syllogism",
    SemanticRule.CSR: "case-split",
    SemanticRule.DN: "double-negation",
    SemanticRule.TRC: "axis-transitivity",
}


@dataclass(frozen=True)
class RandomProofPolicy:
    """Requested realised structure for one randomized proof-graph candidate."""

    semantic_length: int
    distractor_count: int
    allowed_rules: tuple[SemanticRule, ...] = tuple(SemanticRule)
    require_heldout_transition: bool = False
    transition_partition: TransitionPartition | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.semantic_length <= 8:
            raise ValueError("semantic_length must be between 0 and 8")
        if not 0 <= self.distractor_count <= 8:
            raise ValueError("distractor_count must be between 0 and 8")
        if not self.allowed_rules:
            raise ValueError("allowed_rules must not be empty")
        normalized = tuple(SemanticRule(rule) for rule in self.allowed_rules)
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed_rules must be unique")
        object.__setattr__(self, "allowed_rules", normalized)
        if self.require_heldout_transition and self.semantic_length < 2:
            raise ValueError("a held-out transition requires semantic_length >= 2")

    def resolved_partition(self) -> TransitionPartition:
        if self.transition_partition is not None:
            return self.transition_partition
        rules = frozenset(self.allowed_rules)
        return TransitionPartition(
            frozenset((first, second) for first in rules for second in rules),
            frozenset(),
        )


@dataclass(frozen=True)
class TransitionPartition:
    seen: frozenset[Transition]
    heldout: frozenset[Transition]

    def __post_init__(self) -> None:
        if self.seen & self.heldout:
            raise ValueError("seen and held-out transitions must be disjoint")

    @property
    def eligible(self) -> frozenset[Transition]:
        return self.seen | self.heldout


def _weighted_choice(
    rng: random.Random,
    values: list[SemanticRule],
    weights: list[int],
) -> SemanticRule:
    total = sum(weights)
    if total <= 0:
        raise ValueError("cannot sample from an empty proof-path distribution")
    ticket = rng.randrange(total)
    for value, weight in zip(values, weights, strict=True):
        if ticket < weight:
            return value
        ticket -= weight
    raise AssertionError("weighted choice did not consume its ticket")


def _completion_counter(
    allowed: tuple[SemanticRule, ...],
    partition: TransitionPartition,
    require_exactly_one_heldout: bool,
):
    @lru_cache(maxsize=None)
    def completions(
        current: SemanticRule, remaining: int, heldout_used: int
    ) -> int:
        if remaining == 0:
            return int(bool(heldout_used) == require_exactly_one_heldout)
        total = 0
        for following in allowed:
            transition = (current, following)
            if transition in partition.seen:
                total += completions(following, remaining - 1, heldout_used)
            elif transition in partition.heldout and heldout_used == 0:
                total += completions(following, remaining - 1, 1)
        return total

    return completions


def count_rule_paths(
    length: int,
    rules: Iterable[SemanticRule],
    partition: TransitionPartition,
    *,
    require_exactly_one_heldout: bool = False,
) -> int:
    """Count complete rule sequences under one transition constraint."""
    allowed = tuple(dict.fromkeys(SemanticRule(rule) for rule in rules))
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return int(not require_exactly_one_heldout)
    if not allowed:
        return 0

    completions = _completion_counter(
        allowed, partition, require_exactly_one_heldout
    )
    return sum(completions(rule, length - 1, 0) for rule in allowed)


def sample_rule_path(
    rng: random.Random,
    length: int,
    rules: Iterable[SemanticRule],
    partition: TransitionPartition,
    *,
    require_exactly_one_heldout: bool = False,
) -> tuple[SemanticRule, ...]:
    """Uniformly sample a complete feasible sequence using completion counts."""
    allowed = tuple(dict.fromkeys(SemanticRule(rule) for rule in rules))
    if length == 0:
        if require_exactly_one_heldout:
            raise ValueError("length zero cannot contain a held-out transition")
        return ()
    if length < 0 or not allowed:
        raise ValueError("length must be positive and rules must not be empty")

    completions = _completion_counter(
        allowed, partition, require_exactly_one_heldout
    )
    starts = list(allowed)
    start_weights = [completions(rule, length - 1, 0) for rule in starts]
    if not any(start_weights):
        description = "exactly one held-out" if require_exactly_one_heldout else "seen"
        raise ValueError(f"no {description} rule path exists at length {length}")
    current = _weighted_choice(rng, starts, start_weights)
    path = [current]
    heldout_used = 0
    for remaining in range(length - 1, 0, -1):
        candidates: list[SemanticRule] = []
        weights: list[int] = []
        next_states: list[int] = []
        for following in allowed:
            transition = (current, following)
            next_heldout = heldout_used
            if transition in partition.seen:
                pass
            elif transition in partition.heldout and heldout_used == 0:
                next_heldout = 1
            else:
                continue
            weight = completions(following, remaining - 1, next_heldout)
            if weight:
                candidates.append(following)
                weights.append(weight)
                next_states.append(next_heldout)
        following = _weighted_choice(rng, candidates, weights)
        selected = candidates.index(following)
        heldout_used = next_states[selected]
        path.append(following)
        current = following
    return tuple(path)


def exponential_quotas(
    values: Iterable[int],
    total: int,
    ratio: float,
) -> dict[int, int]:
    """Largest-remainder integer allocation for a truncated exponential tilt."""
    selected = tuple(values)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("values must be a non-empty sequence of unique integers")
    if total < 0:
        raise ValueError("total must be non-negative")
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be a positive finite number")
    weights = [ratio**index for index in range(len(selected))]
    denominator = sum(weights)
    raw = [total * weight / denominator for weight in weights]
    quotas = [math.floor(value) for value in raw]
    remaining = total - sum(quotas)
    order = sorted(
        range(len(selected)),
        key=lambda index: (-(raw[index] - quotas[index]), index),
    )
    for index in order[:remaining]:
        quotas[index] += 1
    return dict(zip(selected, quotas, strict=True))


def stable_transition_order(
    transitions: Iterable[Transition], *, seed: int
) -> tuple[Transition, ...]:
    """Return a reproducible hash order without depending on Python hash seeds."""

    def key(transition: Transition) -> tuple[str, str, str]:
        text = f"{seed}:{transition[0].value}:{transition[1].value}"
        return hashlib.sha256(text.encode()).hexdigest(), *(
            transition[0].value,
            transition[1].value,
        )

    return tuple(sorted(set(transitions), key=key))


def constructively_supported_transitions() -> frozenset[Transition]:
    """Transitions supported by the backward proof-chain constructor.

    A TRC step can only produce an exact spatial atom.  DJE, CSR, and DN expose
    compound inputs to the preceding step, so those three transitions cannot
    immediately follow TRC in the constructive grammar.
    """
    blocked_after_trc = {SemanticRule.DJE, SemanticRule.CSR, SemanticRule.DN}
    return frozenset(
        (first, second)
        for first in SemanticRule
        for second in SemanticRule
        if first is not SemanticRule.TRC or second not in blocked_after_trc
    )


def deterministic_transition_partition(seed: int = 42) -> TransitionPartition:
    """Create the pre-model 75/25 transition split used by the v2 capability run."""
    ordered = stable_transition_order(constructively_supported_transitions(), seed=seed)
    heldout_target = max(2, round(len(ordered) * 0.25))
    heldout = set(ordered[:heldout_target])
    seen = set(ordered[heldout_target:])
    # A training rule must retain at least one outgoing edge. Move the first
    # stable held-out edge back if the raw hash split isolates its source.
    for rule in SemanticRule:
        if not any(first is rule for first, _ in seen):
            transition = next(item for item in ordered if item[0] is rule)
            heldout.discard(transition)
            seen.add(transition)
    if len(heldout) < 2:
        raise ValueError("transition partition needs at least two held-out edges")
    partition = TransitionPartition(frozenset(seen), frozenset(heldout))
    if count_rule_paths(8, tuple(SemanticRule), partition) == 0:
        raise ValueError("seen transition graph cannot construct L8")
    return partition
