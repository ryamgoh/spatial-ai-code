"""Distribution reporting and fail-closed validation for V2 workloads."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any


def _value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return "none" if value is None else str(value)


def _distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(_value(row["metadata"].get(field)) for row in rows)
    return dict(sorted(counts.items()))


def _difficulty_distribution(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    counts = Counter(
        _value(row["metadata"].get("difficulty", {}).get(field)) for row in rows
    )
    return dict(sorted(counts.items()))


def _distributions(
    rows: Sequence[Mapping[str, Any]],
    fields: Iterable[str],
    *,
    difficulty: bool = False,
) -> dict[str, dict[str, int]]:
    if not rows:
        return {}
    summarize = _difficulty_distribution if difficulty else _distribution
    distributions = {field: summarize(rows, field) for field in fields}
    empty = {"none": len(rows)}
    return {field: counts for field, counts in distributions.items() if counts != empty}


def _user_prompt(row: Mapping[str, Any]) -> str:
    return next(
        str(message["content"])
        for message in row["messages"]
        if message.get("role") == "user"
    )


def build_workload_manifest(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    expected_trace_variants: Iterable[tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a split workload and return a JSON-compatible manifest."""
    train_rows = tuple(train_rows)
    test_rows = tuple(test_rows)
    rows = (*train_rows, *test_rows)
    ids = [str(row.get("id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row IDs")
    if any(not row.get("metadata", {}).get("round_trip_verified") for row in rows):
        raise ValueError("all rows must be round-trip verified")

    train_base_ids = {str(row["metadata"]["base_id"]) for row in train_rows}
    test_base_ids = {str(row["metadata"]["base_id"]) for row in test_rows}
    leaked_base_ids = train_base_ids & test_base_ids
    if leaked_base_ids:
        raise ValueError("cross-split base IDs: " + ", ".join(sorted(leaked_base_ids)))

    train_prompts = {_user_prompt(row) for row in train_rows}
    test_prompts = {_user_prompt(row) for row in test_rows}
    if train_prompts & test_prompts:
        raise ValueError("cross-split prompts")

    prompt_bases: dict[str, set[str]] = defaultdict(set)
    variants_by_base: dict[str, set[tuple[str, str]]] = defaultdict(set)
    rows_by_base: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        metadata = row["metadata"]
        base_id = str(metadata["base_id"])
        prompt_bases[_user_prompt(row)].add(base_id)
        rows_by_base.setdefault(base_id, row)
        variants_by_base[base_id].add(
            (str(metadata["trace_format"]), str(metadata["state_mode"]))
        )
    if any(len(base_ids) != 1 for base_ids in prompt_bases.values()):
        raise ValueError("duplicate prompts belong to different base IDs")

    expected = (
        {(_value(first), _value(second)) for first, second in expected_trace_variants}
        if expected_trace_variants is not None
        else None
    )
    if expected is not None:
        incomplete = sorted(
            base_id
            for base_id, variants in variants_by_base.items()
            if variants != expected
        )
        if incomplete:
            raise ValueError(
                "incomplete trace variants for base IDs: " + ", ".join(incomplete)
            )

    base_distribution_fields = (
        "query_kind",
        "query_direction",
        "target_direction",
        "answer_mode",
        "semantic_shape",
        "menu_coverage",
        "menu_status",
    )
    trace_distribution_fields = ("trace_format", "state_mode")
    difficulty_fields = (
        "possibility_count",
        "direct_query_relation",
        "x_depth",
        "y_depth",
        "axes_independent",
        "min_membership_x_depth",
        "max_membership_x_depth",
        "min_membership_y_depth",
        "max_membership_y_depth",
        "num_distractor_premises",
    )
    base_rows = tuple(rows_by_base.values())
    train_base_rows = tuple(
        row for base_id, row in rows_by_base.items() if base_id in train_base_ids
    )
    test_base_rows = tuple(
        row for base_id, row in rows_by_base.items() if base_id in test_base_ids
    )
    return {
        "schema": "spatial-v2-workload-manifest",
        "total_rows": len(rows),
        "base_problems": len(variants_by_base),
        "splits": {"train": len(train_rows), "test": len(test_rows)},
        "base_splits": {
            "train": len(train_base_ids),
            "test": len(test_base_ids),
        },
        "trace_distributions": _distributions(rows, trace_distribution_fields),
        "base_distributions": _distributions(base_rows, base_distribution_fields),
        "split_base_distributions": {
            "train": _distributions(train_base_rows, base_distribution_fields),
            "test": _distributions(test_base_rows, base_distribution_fields),
        },
        "difficulty": _distributions(base_rows, difficulty_fields, difficulty=True),
        "trace_variants_per_base": dict(
            sorted(Counter(len(value) for value in variants_by_base.values()).items())
        ),
        "validation": {"status": "passed"},
    }
