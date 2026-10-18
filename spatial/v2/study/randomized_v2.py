"""Frozen design and diagnostic preparation for randomized SpatialEntail v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from spatial.v2.consistency import generate_consistency_pair
from spatial.v2.generation import (
    ENTITY_NAMES,
    GenerationPolicy,
    MenuCoverage,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.random_proof import (
    PROOF_RULE_LABELS,
    RandomProofPolicy,
    SemanticRule,
    TransitionPartition,
    constructively_supported_transitions,
    deterministic_transition_partition,
    exponential_quotas,
)
from spatial.v2.random_proof_generation import possibility_backbone_length


class BenchmarkPartition(str, Enum):
    MATCHED = "matched"
    LENGTH_OOD = "length-ood"
    NOISE_OOD = "noise-ood"
    COMPOSITION_OOD = "composition-ood"


@dataclass(frozen=True)
class PartitionQuota:
    partition: BenchmarkPartition
    count: int
    length: dict[int, int]
    distractors: dict[int, int]
    transition_contract: str
    pair_caps: tuple[tuple[int, int, int], ...] = ()
    pair_cap_basis: str | None = None


@dataclass(frozen=True)
class RandomizedStudyDesign:
    """Frozen train/OOD boundaries and maximum-entropy tilt parameters."""

    train_max_l: int = 6
    train_max_x: int = 6
    eval_max_l: int = 8
    eval_max_x: int = 8
    benchmark_l_ratio: float = 1.15
    benchmark_x_ratio: float = 1.10
    training_l_ratio: float = 1.40
    training_x_ratio: float = 1.25

    def __post_init__(self) -> None:
        if not 4 <= self.train_max_l < self.eval_max_l <= 8:
            raise ValueError("require 4 <= train_max_l < eval_max_l <= 8")
        if not 0 <= self.train_max_x < self.eval_max_x <= 8:
            raise ValueError("require 0 <= train_max_x < eval_max_x <= 8")
        if any(
            ratio <= 0
            for ratio in (
                self.benchmark_l_ratio,
                self.benchmark_x_ratio,
                self.training_l_ratio,
                self.training_x_ratio,
            )
        ):
            raise ValueError("all exponential ratios must be positive")


DEFAULT_DESIGN = RandomizedStudyDesign()
_CONSTRUCTION_OBJECT_COUNT = 24
_POSSIBILITY_CANDIDATE_COUNT = 5


def minimum_semantic_length(query_kind: QueryKind, gold_size: int) -> int:
    """Audited lower bound for the complete certificate, not generator intent."""
    if gold_size == 1 or query_kind is not QueryKind.COUNT:
        return 1
    # Complete Count evidence refutes assignments for excluded counts.  The
    # current independently replayed certificate has a G-dependent lower bound.
    return gold_size


def minimum_composition_length(query_kind: QueryKind, gold_size: int) -> int:
    """Reserve two linear prefix steps for one held-out ordered transition."""
    if gold_size == 1:
        base_length = 0
    else:
        skeleton = "dje" if query_kind is QueryKind.COUNT else "exclude"
        base_length = possibility_backbone_length(
            query_kind.value, skeleton, gold_size
        )
        if base_length is None:
            raise ValueError("composition requires a known possibility backbone")
    return max(minimum_semantic_length(query_kind, gold_size), base_length + 2)


def calibration_length_quotas(
    query_kind: QueryKind,
    gold_size: int,
    *,
    total: int = 200,
    design: RandomizedStudyDesign = DEFAULT_DESIGN,
) -> dict[int, int]:
    minimum = minimum_semantic_length(query_kind, gold_size)
    return exponential_quotas(range(minimum, design.eval_max_l + 1), total, 1.0)


def benchmark_partition_quotas(
    query_kind: QueryKind,
    gold_size: int,
    design: RandomizedStudyDesign = DEFAULT_DESIGN,
) -> tuple[PartitionQuota, ...]:
    minimum = minimum_semantic_length(query_kind, gold_size)
    matched_lengths = range(minimum, design.train_max_l + 1)
    composition_lengths = range(
        minimum_composition_length(query_kind, gold_size),
        design.train_max_l + 1,
    )
    if not tuple(matched_lengths) or not tuple(composition_lengths):
        raise ValueError("query/gold combination has no feasible L allocation")
    return (
        PartitionQuota(
            BenchmarkPartition.MATCHED,
            120,
            exponential_quotas(
                matched_lengths, 120, design.benchmark_l_ratio
            ),
            exponential_quotas(
                range(0, design.train_max_x + 1),
                120,
                design.benchmark_x_ratio,
            ),
            "seen-only",
            (
                ((minimum, 0, 4),)
                if query_kind is QueryKind.COUNT and gold_size > 1
                else ()
            ),
            (
                "empirical-operational"
                if query_kind is QueryKind.COUNT and gold_size > 1
                else None
            ),
        ),
        PartitionQuota(
            BenchmarkPartition.LENGTH_OOD,
            60,
            exponential_quotas(
                range(design.train_max_l + 1, design.eval_max_l + 1),
                60,
                design.benchmark_l_ratio,
            ),
            exponential_quotas(
                range(0, design.train_max_x + 1),
                60,
                design.benchmark_x_ratio,
            ),
            "seen-only",
        ),
        PartitionQuota(
            BenchmarkPartition.NOISE_OOD,
            60,
            exponential_quotas(matched_lengths, 60, design.benchmark_l_ratio),
            exponential_quotas(
                range(design.train_max_x + 1, design.eval_max_x + 1),
                60,
                design.benchmark_x_ratio,
            ),
            "seen-only",
        ),
        PartitionQuota(
            BenchmarkPartition.COMPOSITION_OOD,
            60,
            exponential_quotas(
                composition_lengths, 60, design.benchmark_l_ratio
            ),
            exponential_quotas(
                range(0, design.train_max_x + 1),
                60,
                design.benchmark_x_ratio,
            ),
            "exactly-one-heldout",
        ),
    )


def pair_marginal_quotas(
    first: dict[int, int],
    second: dict[int, int],
    *,
    seed: int,
    pair_caps: dict[tuple[int, int], int] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Pair two exact marginals without declaring a full Cartesian matrix."""
    if sum(first.values()) != sum(second.values()):
        raise ValueError("marginal totals must match")
    caps = pair_caps or {}
    if any(cap < 0 for cap in caps.values()):
        raise ValueError("pair capacities must be non-negative")
    left = [value for value, count in sorted(first.items()) for _ in range(count)]
    right = [value for value, count in sorted(second.items()) for _ in range(count)]
    for retry in range(1_000):
        shuffled = list(right)
        random.Random(seed + retry * 1_000_003).shuffle(shuffled)
        pairs = tuple(zip(left, shuffled, strict=True))
        counts = Counter(pairs)
        if all(counts[pair] <= cap for pair, cap in caps.items()):
            return pairs
    raise ValueError("could not pair marginals within declared capacities")


def training_assignments(
    contract: str,
    *,
    total: int,
    seed: int,
    development: bool = False,
    design: RandomizedStudyDesign = DEFAULT_DESIGN,
) -> tuple[tuple[int, int, int], ...]:
    """Assign a monotone exponential L distribution within every gold size."""
    if contract not in {"single", "multi"}:
        raise ValueError("contract must be single or multi")
    if total <= 0 or (contract == "multi" and total % 4):
        raise ValueError("total must be positive and MULTI totals divisible by four")
    gold_counts = {1: total} if contract == "single" else {
        gold: total // 4 for gold in range(1, 5)
    }
    rng = random.Random(seed)
    assignments: list[tuple[int, int, int]] = []
    for gold, count in sorted(gold_counts.items()):
        minimum = 1 if contract == "single" else gold
        length_quotas = exponential_quotas(
            range(minimum, design.train_max_l + 1),
            count,
            1.0 if development else design.training_l_ratio,
        )
        distractor_quotas = exponential_quotas(
            range(0, design.train_max_x + 1),
            count,
            1.0 if development else design.training_x_ratio,
        )
        lengths = [
            length
            for length, length_count in length_quotas.items()
            for _ in range(length_count)
        ]
        distractors = [
            value
            for value, distractor_count in distractor_quotas.items()
            for _ in range(distractor_count)
        ]
        rng.shuffle(lengths)
        rng.shuffle(distractors)
        assignments.extend(
            (gold, length, distractor)
            for length, distractor in zip(lengths, distractors, strict=True)
        )
    rng.shuffle(assignments)
    return tuple(assignments)


def _transition_payload() -> dict[str, Any]:
    partition = deterministic_transition_partition(42)
    render = lambda transition: f"{transition[0].value}->{transition[1].value}"
    return {
        "seed": 42,
        "seen": [render(item) for item in sorted(partition.seen, key=render)],
        "heldout": [render(item) for item in sorted(partition.heldout, key=render)],
        "capability_probe_attempts": 20,
        "capability_success_threshold": _capability_success_threshold(20),
    }


def study_manifest(
    design: RandomizedStudyDesign = DEFAULT_DESIGN,
) -> dict[str, Any]:
    benchmark = {}
    for query in QueryKind:
        for gold in range(1, 5):
            key = f"{query.value}-g{gold}"
            benchmark[key] = [
                {
                    **asdict(quota),
                    "partition": quota.partition.value,
                    "length": {str(k): v for k, v in quota.length.items()},
                    "distractors": {
                        str(k): v for k, v in quota.distractors.items()
                    },
                }
                for quota in benchmark_partition_quotas(query, gold, design)
            ]
    return {
        "schema": "spatialentail-randomized-v2-study-v1",
        "seed": 42,
        "design": asdict(design),
        "contracts": {
            "single": {"gold_sizes": [1], "headline_examples": 900},
            "all-possible": {
                "gold_sizes": [2, 3, 4],
                "headline_examples": 2700,
            },
            "visible-possible": {
                "capability_only": True,
                "visible_gold_size": "0..G",
            },
        },
        "fixed_spatial_menu_size": 5,
        "calibration": {
            "examples": 2400,
            "examples_per_query_gold": 200,
            "lengths": list(range(1, design.eval_max_l + 1)),
            "distractors": list(range(0, design.eval_max_x + 1)),
            "acceptance_floor": 0.05,
            "median_generation_seconds_ceiling": 60,
            "query_gold_length_quotas": {
                f"{query.value}-g{gold}": calibration_length_quotas(
                    query, gold, design=design
                )
                for query in QueryKind
                for gold in range(1, 5)
            },
        },
        "transitions": _transition_payload(),
        "benchmark": benchmark,
        "training": {
            "examples_per_contract": 8000,
            "development_per_contract": 800,
            "single_length": exponential_quotas(
                range(1, design.train_max_l + 1),
                8000,
                design.training_l_ratio,
            ),
            "multi_length_by_gold": {
                str(gold): exponential_quotas(
                    range(gold, design.train_max_l + 1),
                    2000,
                    design.training_l_ratio,
                )
                for gold in range(1, 5)
            },
            "single_distractors": exponential_quotas(
                range(0, design.train_max_x + 1),
                8000,
                design.training_x_ratio,
            ),
            "multi_distractors_by_gold": {
                str(gold): exponential_quotas(
                    range(0, design.train_max_x + 1),
                    2000,
                    design.training_x_ratio,
                )
                for gold in range(1, 5)
            },
            "single_gold": {"1": 8000},
            "multi_gold": {str(gold): 2000 for gold in range(1, 5)},
            "seeds": [42, 43],
            "supervision_by_contract": {
                "single": ["natural", "answer-only"],
                "multi": ["natural"],
            },
        },
        "diagnostics": {
            "special_response_examples": 300,
            "consistency_pairs": 280,
            "consistency_prompts": 560,
            "consistency_cycle_lengths": list(range(2, 9)),
            "consistency_axes": ["x", "y"],
        },
    }


def write_or_verify_manifest(path: Path, manifest: dict[str, Any]) -> str:
    """Write a manifest once, or reuse an identical manifest after cancellation."""
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(
                f"manifest exists with a different study design: {path}"
            )
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return "written"


def write_consistency_diagnostic(
    output: Path, *, seed: int = 42, pairs_per_cycle_axis: int = 20
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    if pairs_per_cycle_axis <= 0:
        raise ValueError("pairs_per_cycle_axis must be positive")
    output.mkdir(parents=True)
    rows = []
    pair_count = 0
    for cycle_length in range(2, 9):
        for axis in ("x", "y"):
            for sample_index in range(pairs_per_cycle_axis):
                distractors = sample_index % 6
                pair = generate_consistency_pair(
                    names=ENTITY_NAMES,
                    cycle_length=cycle_length,
                    distractor_count=distractors,
                    axis=axis,
                    seed=seed + cycle_length * 10_000 + sample_index * 10 + int(axis == "y"),
                )
                rows.extend(pair.rows())
                pair_count += 1
    content = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    (output / "consistency.jsonl").write_text(content)
    receipt = {
        "schema": "spatialentail-consistency-diagnostic-v1",
        "seed": seed,
        "pairs": pair_count,
        "rows": len(rows),
        "cycle_lengths": list(range(2, 9)),
        "axes": ["x", "y"],
        "distractors": list(range(0, 6)),
        "pairs_per_cycle_axis": pairs_per_cycle_axis,
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def _tag_diagnostic(row: dict[str, Any], category: str) -> dict[str, Any]:
    row.setdefault("metadata", {})["diagnostic_category"] = category
    row["metadata"]["headline"] = False
    return row


def write_special_response_diagnostic(
    output: Path,
    *,
    seed: int = 42,
    l0_per_query: int = 30,
    ambiguous_per_query: int = 30,
    no_match_count: int = 30,
    visible_per_query_gold: int = 10,
) -> dict[str, Any]:
    """Write the non-headline L0 and special-answer capability suite."""
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    counts = (
        l0_per_query,
        ambiguous_per_query,
        no_match_count,
        visible_per_query_gold,
    )
    if any(count <= 0 for count in counts):
        raise ValueError("special-response diagnostic counts must be positive")
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    cursor = 0
    for query in QueryKind:
        for _ in range(l0_per_query):
            generator = SpatialGeneratorV2(seed + cursor)
            sample = generator.generate_randomized(
                GenerationPolicy(
                    query_kind=query,
                    answer_mode=AnswerMode.SINGLE,
                    semantic_shape=SemanticShape.UNIQUE,
                    num_entities=16,
                    num_premises=15,
                    menu_option_count=5,
                ),
                RandomProofPolicy(0, 0),
            )
            rows.append(
                _tag_diagnostic(
                    sample.as_sft_row(include_audit=True), "direct-l0"
                )
            )
            cursor += 1
        for _ in range(ambiguous_per_query):
            generator = SpatialGeneratorV2(seed + cursor)
            sample = generator.generate(
                GenerationPolicy(
                    query_kind=query,
                    answer_mode=AnswerMode.SINGLE,
                    semantic_shape=SemanticShape.AMBIGUOUS,
                    ambiguity_size=2,
                    num_entities=8,
                    num_premises=7,
                    menu_option_count=5,
                ),
                max_attempts=5_000,
            )
            rows.append(
                _tag_diagnostic(
                    sample.as_sft_row(include_audit=True), "single-ambiguous"
                )
            )
            cursor += 1
    for _ in range(no_match_count):
        generator = SpatialGeneratorV2(seed + cursor)
        sample = generator.generate(
            GenerationPolicy(
                query_kind=QueryKind.WHICH,
                answer_mode=AnswerMode.SINGLE,
                semantic_shape=SemanticShape.NO_MATCH,
                num_entities=8,
                num_premises=7,
                menu_option_count=5,
            ),
            max_attempts=5_000,
        )
        rows.append(
            _tag_diagnostic(
                sample.as_sft_row(include_audit=True), "selection-no-match"
            )
        )
        cursor += 1
    for query in QueryKind:
        for gold in range(2, 5):
            for _ in range(visible_per_query_gold):
                generator = SpatialGeneratorV2(seed + cursor)
                base = generator.generate(
                    GenerationPolicy(
                        query_kind=query,
                        answer_mode=AnswerMode.ALL_POSSIBLE,
                        semantic_shape=SemanticShape.AMBIGUOUS,
                        possibility_skeleton="dje",
                        possibility_gold_size=gold,
                        possibility_candidate_count=(
                            gold + 4
                            if query is QueryKind.WHICH
                            else gold + 3
                            if query is QueryKind.COUNT
                            else 5
                        ),
                        num_entities=14,
                        num_premises=13,
                        ordinary_option_target=5,
                        menu_option_count=5,
                    ),
                    max_attempts=5_000,
                )
                visible = generator.with_answer_mode(
                    base,
                    AnswerMode.VISIBLE_POSSIBLE,
                    MenuCoverage.ZERO,
                    visible_gold_size=0,
                    menu_option_count=5,
                )
                rows.append(
                    _tag_diagnostic(
                        visible.as_sft_row(include_audit=True), "visible-zero"
                    )
                )
                cursor += 1
    signatures = [row["metadata"]["structure_signature"] for row in rows]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("special-response diagnostic contains duplicate structures")
    content = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    (output / "special-responses.jsonl").write_text(content)
    receipt = {
        "schema": "spatialentail-special-response-diagnostic-v1",
        "seed": seed,
        "rows": len(rows),
        "categories": {
            category: sum(
                row["metadata"]["diagnostic_category"] == category for row in rows
            )
            for category in (
                "direct-l0",
                "single-ambiguous",
                "selection-no-match",
                "visible-zero",
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def _calibration_seed(
    base_seed: int,
    query: QueryKind,
    gold: int,
    length: int,
    sample_index: int,
) -> int:
    payload = f"{base_seed}:{query.value}:{gold}:{length}:{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _duplicate_structure_indices(rows: list[dict[str, Any]]) -> list[int]:
    seen: set[str] = set()
    duplicates = []
    for index, row in enumerate(rows):
        signature = row.get("metadata", {}).get("structure_signature")
        if not isinstance(signature, str) or not signature:
            raise ValueError(f"row {index} lacks a canonical structure signature")
        if signature in seen:
            duplicates.append(index)
        else:
            seen.add(signature)
    return duplicates


def _replacement_seed(original: int, retry: int) -> int:
    payload = f"{original}:canonical-replacement:{retry}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _replace_headline_duplicates(
    rows: list[dict[str, Any]],
    work: list[tuple[str, int, str, int, int, int]],
    *,
    workers: int,
    max_rounds: int = 50,
) -> int:
    """Regenerate only repeated final-benchmark structures."""
    replacement_count = 0
    retries: dict[int, int] = {}
    for _ in range(max_rounds):
        duplicates = _duplicate_structure_indices(rows)
        if not duplicates:
            return replacement_count
        replacement_count += len(duplicates)
        replacement_work = []
        for index in duplicates:
            retries[index] = retries.get(index, 0) + 1
            original = work[index]
            replacement_work.append(
                (
                    index,
                    (*original[:-1], _replacement_seed(original[-1], retries[index])),
                )
            )
        if workers == 1:
            for index, item in replacement_work:
                rows[index] = _headline_row(*item)
        else:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(replacement_work))
            ) as executor:
                futures = {
                    executor.submit(_headline_row, *item): index
                    for index, item in replacement_work
                }
                for future in as_completed(futures):
                    rows[futures[future]] = future.result()
    remaining = len(_duplicate_structure_indices(rows))
    raise RuntimeError(
        f"could not obtain a unique benchmark after {max_rounds} replacement "
        f"rounds; {remaining} duplicate rows remain"
    )


def _randomized_policy(query: QueryKind, gold: int) -> GenerationPolicy:
    if gold == 1:
        return GenerationPolicy(
            query_kind=query,
            answer_mode=AnswerMode.SINGLE,
            semantic_shape=SemanticShape.UNIQUE,
            num_entities=_CONSTRUCTION_OBJECT_COUNT,
            num_premises=_CONSTRUCTION_OBJECT_COUNT - 1,
            menu_option_count=5,
        )
    skeleton = "dje" if query is QueryKind.COUNT else "exclude"
    return GenerationPolicy(
        query_kind=query,
        answer_mode=AnswerMode.ALL_POSSIBLE,
        semantic_shape=SemanticShape.AMBIGUOUS,
        ambiguity_size=gold,
        possibility_skeleton=skeleton,
        possibility_gold_size=gold,
        possibility_candidate_count=_POSSIBILITY_CANDIDATE_COUNT,
        num_entities=_CONSTRUCTION_OBJECT_COUNT,
        num_premises=_CONSTRUCTION_OBJECT_COUNT - 1,
        ordinary_option_target=5,
        menu_option_count=5,
    )


def _calibration_row(
    query_value: str,
    gold: int,
    length: int,
    distractors: int,
    seed: int,
) -> dict[str, Any]:
    query = QueryKind(query_value)
    generator = SpatialGeneratorV2(seed)
    policy = _randomized_policy(query, gold)
    transitions = constructively_supported_transitions()
    sample = generator.generate_randomized(
        policy,
        RandomProofPolicy(
            length,
            distractors,
            forbidden_final_rules=(SemanticRule.TRC,) if gold > 1 else (),
            transition_partition=TransitionPartition(transitions, frozenset()),
        ),
        max_attempts=2_000,
    )
    row = sample.as_sft_row(include_audit=True)
    row["metadata"].update(
        diagnostic_category="randomized-v2-calibration",
        calibration_gold_size=gold,
        calibration_length=length,
        calibration_distractors=distractors,
        headline=False,
    )
    return row


def _transition_probe_attempt(
    query_value: str, first_value: str, second_value: str, seed: int
) -> bool:
    first = SemanticRule(first_value)
    second = SemanticRule(second_value)
    edge = (first, second)
    generator = SpatialGeneratorV2(seed)
    try:
        generator.generate_randomized(
            GenerationPolicy(
                query_kind=QueryKind(query_value),
                answer_mode=AnswerMode.SINGLE,
                semantic_shape=SemanticShape.UNIQUE,
                num_entities=16,
                num_premises=15,
                menu_option_count=5,
            ),
            RandomProofPolicy(
                2,
                0,
                allowed_rules=tuple(dict.fromkeys(edge)),
                transition_partition=TransitionPartition(frozenset({edge}), frozenset()),
            ),
            max_attempts=20,
        )
    except (RuntimeError, ValueError):
        return False
    return True


def _capability_success_threshold(attempts: int) -> int:
    return (4 * attempts + 4) // 5


def probe_transition_capability(
    *,
    seed: int = 42,
    attempts: int = 20,
    workers: int = 1,
) -> dict[str, Any]:
    """Probe every ordered pair per query before model-dependent work begins."""
    if attempts <= 0 or workers <= 0:
        raise ValueError("transition probe attempts and workers must be positive")
    threshold = _capability_success_threshold(attempts)
    transitions = [
        (first, second) for first in SemanticRule for second in SemanticRule
    ]
    work = [
        (
            query.value,
            first.value,
            second.value,
            seed + query_index * 100_000 + edge_index * 1_000 + attempt,
        )
        for query_index, query in enumerate(QueryKind)
        for edge_index, (first, second) in enumerate(transitions)
        for attempt in range(attempts)
    ]
    results: list[bool]
    if workers == 1:
        results = [_transition_probe_attempt(*item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(work))) as executor:
            results = list(executor.map(_transition_probe_attempt, *zip(*work)))
    cursor = 0
    query_results = {}
    constructively_supported = constructively_supported_transitions()
    for query in QueryKind:
        edges = {}
        eligible_count = 0
        for first, second in transitions:
            successes = sum(results[cursor : cursor + attempts])
            cursor += attempts
            name = f"{first.value}->{second.value}"
            eligible = successes >= threshold
            eligible_count += int(eligible)
            edges[name] = {
                "successes": successes,
                "attempts": attempts,
                "eligible": eligible,
                "constructively_supported": (
                    (first, second) in constructively_supported
                ),
            }
        if eligible_count < 6:
            raise RuntimeError(
                f"transition capability gate found only {eligible_count} reliable "
                f"edges for {query.value}; at least six are required"
            )
        query_results[query.value] = {
            "eligible_transition_count": eligible_count,
            "edges": edges,
        }
    return {
        "schema": "spatialentail-randomized-v2-transition-probe-v1",
        "seed": seed,
        "attempts_per_transition": attempts,
        "success_threshold": threshold,
        "queries": query_results,
    }


def _heldout_profile_edges(profile: dict[str, Any]) -> set[tuple[str, str]]:
    partition = deterministic_transition_partition(42)
    heldout = {
        (PROOF_RULE_LABELS[first], PROOF_RULE_LABELS[second])
        for first, second in partition.heldout
    }
    actual = {
        tuple(edge)
        for edge in profile.get("ordered_transitions", [])
        if isinstance(edge, (list, tuple)) and len(edge) == 2
    }
    return actual & heldout


def _headline_row(
    query_value: str,
    gold: int,
    partition_value: str,
    length: int,
    distractors: int,
    seed: int,
) -> dict[str, Any]:
    query = QueryKind(query_value)
    benchmark_partition = BenchmarkPartition(partition_value)
    transition_partition = deterministic_transition_partition(42)
    require_heldout = benchmark_partition is BenchmarkPartition.COMPOSITION_OOD
    composition_minimum = minimum_composition_length(query, gold)
    if require_heldout and length < composition_minimum:
        raise ValueError(
            f"{query.value} G{gold} composition requires at least "
            f"L{composition_minimum}"
        )
    proof_partition = (
        transition_partition
        if require_heldout
        else TransitionPartition(transition_partition.seen, frozenset())
    )
    last_error = "no candidate attempted"
    for retry in range(100):
        candidate_seed = seed + retry * 1_000_003
        generator = SpatialGeneratorV2(candidate_seed)
        policy = _randomized_policy(query, gold)
        try:
            sample = generator.generate_randomized(
                policy,
                RandomProofPolicy(
                    length,
                    distractors,
                    require_heldout_transition=require_heldout,
                    transition_partition=proof_partition,
                    forbidden_final_rules=(SemanticRule.TRC,) if gold > 1 else (),
                ),
                max_attempts=2_000,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            continue
        profile = sample.difficulty["proof_complexity"]
        actual_heldout = _heldout_profile_edges(profile)
        if require_heldout and len(actual_heldout) != 1:
            last_error = "composition candidate did not realise exactly one held-out edge"
            continue
        if not require_heldout and actual_heldout:
            last_error = "seen candidate realised a held-out edge"
            continue
        row = sample.as_sft_row(include_audit=True)
        row["metadata"].update(
            benchmark_partition=benchmark_partition.value,
            gold_size=gold,
            headline=True,
            heldout_transitions=["->".join(edge) for edge in sorted(actual_heldout)],
        )
        return row
    raise RuntimeError(
        f"could not generate {query.value} G{gold} {partition_value} "
        f"L{length} X{distractors}: {last_error}"
    )


def prepare_calibration(
    output: Path,
    *,
    seed: int = 42,
    examples_per_query_gold: int = 200,
    workers: int | None = None,
    query_kinds: tuple[QueryKind, ...] = tuple(QueryKind),
    gold_sizes: tuple[int, ...] = (1, 2, 3, 4),
    run_transition_probe: bool = True,
    design: RandomizedStudyDesign = DEFAULT_DESIGN,
) -> dict[str, Any]:
    """Generate the structural capability pool with deterministic work items."""
    if output.exists():
        raise FileExistsError(f"calibration output already exists: {output}")
    if examples_per_query_gold <= 0:
        raise ValueError("examples_per_query_gold must be positive")
    work = []
    for query in query_kinds:
        for gold in gold_sizes:
            quotas = calibration_length_quotas(
                query,
                gold,
                total=examples_per_query_gold,
                design=design,
            )
            cursor = 0
            for length, count in quotas.items():
                for local_index in range(count):
                    work.append(
                        (
                            query.value,
                            gold,
                            length,
                            cursor % (design.eval_max_x + 1),
                            _calibration_seed(seed, query, gold, length, local_index),
                        )
                    )
                    cursor += 1
    requested_workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if requested_workers <= 0:
        raise ValueError("workers must be positive")
    worker_count = min(requested_workers, len(work))
    transition_probe = (
        probe_transition_capability(
            seed=seed,
            attempts=20,
            workers=worker_count,
        )
        if run_transition_probe
        else None
    )
    rows: list[dict[str, Any] | None] = [None] * len(work)
    started = time.perf_counter()
    print(
        f"Generating {len(work)} randomized calibration examples with "
        f"{worker_count} worker process(es)",
        file=sys.stderr,
        flush=True,
    )
    if worker_count == 1:
        for index, item in enumerate(work):
            rows[index] = _calibration_row(*item)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_calibration_row, *item): index
                for index, item in enumerate(work)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                rows[index] = future.result()
                if completed % 50 == 0 or completed == len(futures):
                    print(
                        f"[{completed}/{len(futures)}] randomized calibration",
                        file=sys.stderr,
                        flush=True,
                    )
    admitted = [row for row in rows if row is not None]
    if len(admitted) != len(work):
        raise RuntimeError("calibration generation left empty work slots")
    duplicate_indices = _duplicate_structure_indices(admitted)
    output.mkdir(parents=True)
    content = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in admitted
    )
    (output / "source-natural.jsonl").write_text(content)
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": "spatialentail-randomized-v2-calibration-v1",
        "seed": seed,
        "rows": len(admitted),
        "examples_per_query_gold": examples_per_query_gold,
        "query_kinds": [query.value for query in query_kinds],
        "gold_sizes": list(gold_sizes),
        "workers": worker_count,
        "elapsed_seconds": elapsed,
        "unique_structures": len(admitted) - len(duplicate_indices),
        "duplicate_structure_rows": len(duplicate_indices),
        "transition_probe": transition_probe,
        "study": study_manifest(design),
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def prepare_headline_benchmark(
    output: Path,
    *,
    calibration_manifest: Path,
    seed: int = 42,
    workers: int | None = None,
) -> dict[str, Any]:
    """Generate the frozen candidate after a complete calibration receipt exists."""
    if output.exists():
        raise FileExistsError(f"benchmark output already exists: {output}")
    calibration = json.loads(calibration_manifest.read_text())
    if calibration.get("schema") != "spatialentail-randomized-v2-calibration-v1":
        raise ValueError("benchmark requires a randomized-v2 calibration manifest")
    if calibration.get("rows") != 2400:
        raise ValueError("benchmark requires the complete 2,400-row calibration")
    if not isinstance(calibration.get("transition_probe"), dict):
        raise ValueError("benchmark requires the completed transition capability probe")
    try:
        design = RandomizedStudyDesign(**calibration["study"]["design"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibration manifest lacks a valid frozen study design") from exc
    expected_study = json.loads(json.dumps(study_manifest(design)))
    if calibration["study"] != expected_study:
        raise ValueError(
            "calibration uses a superseded study manifest; regenerate calibration"
        )
    work = []
    for query in QueryKind:
        for gold in range(1, 5):
            for quota in benchmark_partition_quotas(query, gold, design):
                pairs = pair_marginal_quotas(
                    quota.length,
                    quota.distractors,
                    seed=_calibration_seed(seed, query, gold, quota.count, 0),
                    pair_caps={
                        (length, distractors): cap
                        for length, distractors, cap in quota.pair_caps
                    },
                )
                for sample_index, (length, distractors) in enumerate(pairs):
                    work.append(
                        (
                            query.value,
                            gold,
                            quota.partition.value,
                            length,
                            distractors,
                            _calibration_seed(
                                seed,
                                query,
                                gold,
                                length * 100 + distractors,
                                sample_index,
                            ),
                        )
                    )
    requested_workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    worker_count = min(requested_workers, len(work))
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    rows: list[dict[str, Any] | None] = [None] * len(work)
    started = time.perf_counter()
    if worker_count == 1:
        for index, item in enumerate(work):
            rows[index] = _headline_row(*item)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_headline_row, *item): index
                for index, item in enumerate(work)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                rows[futures[future]] = future.result()
                if completed % 50 == 0 or completed == len(futures):
                    print(
                        f"[{completed}/{len(futures)}] randomized headline benchmark",
                        file=sys.stderr,
                        flush=True,
                    )
    admitted = [row for row in rows if row is not None]
    if len(admitted) != 3600:
        raise RuntimeError("headline benchmark did not produce 3,600 rows")
    replacement_count = _replace_headline_duplicates(
        admitted,
        work,
        workers=worker_count,
    )
    output.mkdir(parents=True)
    (output / "source-natural.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in admitted
        )
    )
    receipt = {
        "schema": "spatialentail-randomized-v2-benchmark-v1",
        "seed": seed,
        "rows": len(admitted),
        "workers": worker_count,
        "elapsed_seconds": time.perf_counter() - started,
        "canonical_replacements": replacement_count,
        "unique_structures": len(admitted),
        "calibration_manifest_sha256": hashlib.sha256(
            calibration_manifest.read_bytes()
        ).hexdigest(),
        "study": study_manifest(design),
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--consistency-out", type=Path)
    parser.add_argument("--special-out", type=Path)
    parser.add_argument("--calibration-out", type=Path)
    parser.add_argument("--benchmark-out", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--examples-per-query-gold", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-max-l", type=int, default=DEFAULT_DESIGN.train_max_l)
    parser.add_argument("--train-max-x", type=int, default=DEFAULT_DESIGN.train_max_x)
    parser.add_argument("--eval-max-l", type=int, default=DEFAULT_DESIGN.eval_max_l)
    parser.add_argument("--eval-max-x", type=int, default=DEFAULT_DESIGN.eval_max_x)
    args = parser.parse_args()
    design = RandomizedStudyDesign(
        train_max_l=args.train_max_l,
        train_max_x=args.train_max_x,
        eval_max_l=args.eval_max_l,
        eval_max_x=args.eval_max_x,
    )
    manifest = study_manifest(design)
    if args.manifest_out is not None:
        write_or_verify_manifest(args.manifest_out, manifest)
    if args.consistency_out is not None:
        write_consistency_diagnostic(args.consistency_out, seed=args.seed)
    if args.special_out is not None:
        write_special_response_diagnostic(args.special_out, seed=args.seed)
    if args.calibration_out is not None:
        prepare_calibration(
            args.calibration_out,
            seed=args.seed,
            examples_per_query_gold=args.examples_per_query_gold,
            workers=args.workers,
            design=design,
        )
    if args.benchmark_out is not None:
        if args.calibration_manifest is None:
            parser.error("--benchmark-out requires --calibration-manifest")
        prepare_headline_benchmark(
            args.benchmark_out,
            calibration_manifest=args.calibration_manifest,
            seed=args.seed,
            workers=args.workers,
        )
    if (
        args.manifest_out is None
        and args.consistency_out is None
        and args.special_out is None
        and args.calibration_out is None
        and args.benchmark_out is None
    ):
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
