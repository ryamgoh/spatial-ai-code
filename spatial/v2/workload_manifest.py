"""Distribution reporting and fail-closed validation for V2 workloads."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
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


def workload_output_paths(output_file: str | Path) -> tuple[Path, Path, Path]:
    path = Path(output_file)
    suffix = path.suffix or ".jsonl"
    base = path.with_suffix("") if path.suffix else path
    return (
        base.with_name(base.name + "_train").with_suffix(suffix),
        base.with_name(base.name + "_test").with_suffix(suffix),
        base.with_name(base.name + "_manifest").with_suffix(".json"),
    )


def check_output_paths(paths: Iterable[Path], *, replace: bool) -> None:
    """Fail on existing generated outputs unless replacement is authorized."""
    paths = tuple(paths)
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing and not replace:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists: {rendered}; use replace")


def remove_output_paths(paths: Iterable[Path]) -> None:
    """Remove only exact output paths after replacement has been authorized."""
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def build_workload_manifest(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    expected_trace_variants: Iterable[Any] | None = None,
    expected_variants_by_base: Mapping[str, Iterable[tuple[Any, Any, Any]]]
    | None = None,
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
    variants_by_base: dict[str, set[str]] = defaultdict(set)
    row_variants_by_base: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    rows_by_base: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        metadata = row["metadata"]
        base_id = str(metadata["base_id"])
        prompt_bases[_user_prompt(row)].add(base_id)
        rows_by_base.setdefault(base_id, row)
        variants_by_base[base_id].add(str(metadata["trace_format"]))
        row_variants_by_base[base_id].add(
            (
                str(metadata["answer_mode"]),
                str(metadata["menu_coverage"]),
                str(metadata["trace_format"]),
            )
        )
    if any(len(base_ids) != 1 for base_ids in prompt_bases.values()):
        raise ValueError("duplicate prompts belong to different base IDs")

    expected = (
        {_value(variant) for variant in expected_trace_variants}
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
    if expected_variants_by_base is not None:
        expected_by_base = {
            base_id: {tuple(_value(item) for item in variant) for variant in variants}
            for base_id, variants in expected_variants_by_base.items()
        }
        if set(expected_by_base) != set(row_variants_by_base):
            raise ValueError(
                "expected variant base IDs do not match generated base IDs"
            )
        incomplete = sorted(
            base_id
            for base_id, variants in row_variants_by_base.items()
            if variants != expected_by_base[base_id]
        )
        if incomplete:
            raise ValueError(
                "incomplete answer/trace variants for base IDs: "
                + ", ".join(incomplete)
            )

    base_distribution_fields = (
        "matrix_cell",
        "query_kind",
        "query_direction",
        "target_direction",
        "semantic_shape",
    )
    answer_distribution_fields = (
        "matrix_answer_variant",
        "answer_mode",
        "menu_coverage",
        "menu_status",
    )
    trace_distribution_fields = ("trace_format",)
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
    rejection_reasons: Counter[str] = Counter()
    rejected_candidates = 0
    for row in base_rows:
        metadata = row["metadata"]
        attempt = int(metadata.get("attempt", 0))
        rejection_counts = metadata.get("rejection_counts", {})
        if sum(rejection_counts.values()) != attempt:
            raise ValueError("rejection counts do not match accepted attempt")
        rejected_candidates += attempt
        rejection_reasons.update(rejection_counts)
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
        "answer_distributions": _distributions(rows, answer_distribution_fields),
        "base_distributions": _distributions(base_rows, base_distribution_fields),
        "split_base_distributions": {
            "train": _distributions(train_base_rows, base_distribution_fields),
            "test": _distributions(test_base_rows, base_distribution_fields),
        },
        "difficulty": _distributions(base_rows, difficulty_fields, difficulty=True),
        "generation_efficiency": {
            "accepted_base_problems": len(base_rows),
            "rejected_candidates": rejected_candidates,
            "total_candidate_attempts": len(base_rows) + rejected_candidates,
            "acceptance_rate": (
                len(base_rows) / (len(base_rows) + rejected_candidates)
                if base_rows
                else 0.0
            ),
            "max_rejections_before_accept": max(
                (int(row["metadata"].get("attempt", 0)) for row in base_rows),
                default=0,
            ),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "trace_variants_per_base": dict(
            sorted(Counter(len(value) for value in variants_by_base.values()).items())
        ),
        "answer_variants_per_base": dict(
            sorted(
                Counter(
                    len({variant[:2] for variant in variants})
                    for variants in row_variants_by_base.values()
                ).items()
            )
        ),
        "validation": {"status": "passed"},
    }


def write_workload(
    output_file: str | Path,
    row_groups: list[list[dict]],
    *,
    test_split: float,
    seed: int,
    expected_trace_variants: Iterable[tuple[Any, Any]],
    manifest_metadata: dict[str, Any],
    expected_variants_by_base: Mapping[str, Iterable[tuple[Any, Any, Any, Any]]]
    | None = None,
) -> tuple[Path, Path, Path]:
    """Validate, split, and write already-generated base-problem groups."""
    random.Random(seed).shuffle(row_groups)
    test_size = int(len(row_groups) * test_split)
    test_rows = [row for group in row_groups[:test_size] for row in group]
    train_rows = [row for group in row_groups[test_size:] for row in group]
    manifest = build_workload_manifest(
        train_rows,
        test_rows,
        expected_trace_variants=expected_trace_variants,
        expected_variants_by_base=expected_variants_by_base,
    )
    manifest.update(manifest_metadata)
    train_path, test_path, manifest_path = workload_output_paths(output_file)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, selected in ((train_path, train_rows), (test_path, test_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return train_path, test_path, manifest_path
