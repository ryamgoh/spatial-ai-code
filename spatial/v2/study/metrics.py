"""Paired fixed-suite cell-macro reporting; intervals condition on checkpoints."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from statistics import mean


def paired_macro_difference(
    cells: Mapping[str, Sequence[str]],
    first: Mapping[str, bool],
    second: Mapping[str, bool],
    *,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict:
    ids = [item for items in cells.values() for item in items]
    if (
        not cells
        or any(not items for items in cells.values())
        or len(ids) != len(set(ids))
    ):
        raise ValueError("cells require unique nonempty base IDs")
    if set(first) != set(ids) or set(second) != set(ids):
        raise ValueError("paired predictions must cover exactly the frozen base IDs")
    if (
        any(type(v) is not bool for v in (*first.values(), *second.values()))
        or repetitions < 100
    ):
        raise ValueError(
            "predictions must be Boolean and bootstrap repetitions at least 100"
        )
    differences = {
        cell: [int(second[i]) - int(first[i]) for i in items]
        for cell, items in cells.items()
    }
    rng = random.Random(seed)
    draws = sorted(
        mean(
            mean(rng.choices(values, k=len(values))) for values in differences.values()
        )
        for _ in range(repetitions)
    )
    return {
        "schema": "fixed-suite-paired-macro-v1",
        "difference": mean(mean(values) for values in differences.values()),
        "interval95": [
            draws[int(0.025 * repetitions)],
            draws[min(repetitions - 1, int(0.975 * repetitions))],
        ],
        "cell_differences": {
            cell: mean(values) for cell, values in differences.items()
        },
        "bases": len(ids),
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "scope": "conditional on checkpoints and frozen cells; not training-seed or new-skeleton uncertainty",
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    if not pvalues or any(not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("p-values must lie in [0,1]")
    adjusted = {}
    previous = 0.0
    for rank, (name, value) in enumerate(
        sorted(pvalues.items(), key=lambda item: item[1])
    ):
        previous = max(previous, min(1.0, value * (len(pvalues) - rank)))
        adjusted[name] = previous
    return adjusted


def paired_macro_permutation(cells, first, second, *, repetitions=10000, seed=42):
    """Two-sided paired randomization p-value for the fixed cell-macro contrast."""
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("permutation repetitions must be positive")
    report = paired_macro_difference(cells, first, second, repetitions=100, seed=seed)
    rng = random.Random(seed)
    observed = abs(report["difference"])
    values = [[int(second[i]) - int(first[i]) for i in ids] for ids in cells.values()]
    extreme = 0
    for _ in range(repetitions):
        statistic = mean(
            mean(v * rng.choice((-1, 1)) for v in group) for group in values
        )
        extreme += abs(statistic) >= observed
    return (extreme + 1) / (repetitions + 1)
