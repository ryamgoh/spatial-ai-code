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

from spatial.v2.context_budget import validate_context_admission


class SplitStrategy(str, Enum):
    RANDOM = "random"
    STRUCTURAL = "structural"
    HOLDOUT = "holdout"


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


def _supervision_volume(
    rows: Sequence[Mapping[str, Any]],
    *,
    by_variant: bool = False,
) -> dict[str, dict[str, int | float]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        metadata = row.get("metadata", {})
        arm = metadata.get("supervision_arm")
        if arm is None:
            continue
        if by_variant:
            arm = "/".join(
                str(metadata.get(field, "none"))
                for field in (
                    "answer_mode",
                    "menu_coverage",
                    "trace_format",
                    "supervision_arm",
                )
            )
        totals[str(arm)].update(
            rows=1,
            target_characters=int(metadata.get("target_characters", 0)),
            target_whitespace_tokens=int(metadata.get("target_whitespace_tokens", 0)),
        )
        if metadata.get("target_model_tokens") is not None:
            totals[str(arm)].update(
                model_token_rows=1,
                target_model_tokens=int(metadata["target_model_tokens"]),
            )
    result = {}
    max_characters = max(
        (counts["target_characters"] for counts in totals.values()),
        default=0,
    )
    max_whitespace_tokens = max(
        (counts["target_whitespace_tokens"] for counts in totals.values()),
        default=0,
    )
    complete_model_counts = {
        arm: counts
        for arm, counts in totals.items()
        if counts["model_token_rows"] == counts["rows"]
    }
    max_model_tokens = max(
        (counts["target_model_tokens"] for counts in complete_model_counts.values()),
        default=0,
    )
    for arm, counts in sorted(totals.items()):
        row_count = counts["rows"]
        result[arm] = {
            **dict(counts),
            "mean_target_characters": counts["target_characters"] / row_count,
            "mean_target_whitespace_tokens": (
                counts["target_whitespace_tokens"] / row_count
            ),
            "character_multiplier_to_max": (
                max_characters / counts["target_characters"]
                if counts["target_characters"]
                else 0.0
            ),
            "whitespace_token_multiplier_to_max": (
                max_whitespace_tokens / counts["target_whitespace_tokens"]
                if counts["target_whitespace_tokens"]
                else 0.0
            ),
        }
        if arm in complete_model_counts:
            result[arm].update(
                mean_target_model_tokens=(counts["target_model_tokens"] / row_count),
                model_token_multiplier_to_max=(
                    max_model_tokens / counts["target_model_tokens"]
                    if counts["target_model_tokens"]
                    else 0.0
                ),
            )
    return result


def _user_prompt(row: Mapping[str, Any]) -> str:
    return next(
        str(message["content"])
        for message in row["messages"]
        if message.get("role") == "user"
    )


def _answer_positions(
    rows: Sequence[Mapping[str, Any]],
    *,
    marginal: bool = False,
) -> dict[str, dict[str, int]]:
    """Count each menu once, independent of trace/control copies."""
    seen: set[tuple[str, str, str]] = set()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        metadata = row["metadata"]
        key = tuple(
            str(metadata[field])
            for field in ("base_id", "answer_mode", "menu_coverage")
        )
        if key in seen:
            continue
        seen.add(key)
        letters = metadata.get("oracle_letters")
        if letters is not None:
            counts[str(metadata.get("matrix_cell", "all"))].update(
                letters if marginal else [",".join(sorted(letters))]
            )
    return {
        cell: dict(sorted(values.items())) for cell, values in sorted(counts.items())
    }


def workload_output_paths(output_file: str | Path) -> tuple[Path, Path, Path, Path]:
    path = Path(output_file)
    suffix = path.suffix or ".jsonl"
    base = path.with_suffix("") if path.suffix else path
    return (
        base.with_name(base.name + "_train").with_suffix(suffix),
        base.with_name(base.name + "_dev").with_suffix(suffix),
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
    dev_rows: Sequence[Mapping[str, Any]] = (),
    expected_trace_variants: Iterable[Any] | None = None,
    expected_variants_by_base: Mapping[str, Iterable[tuple[Any, Any, Any, Any]]]
    | None = None,
    split_strategy: SplitStrategy | str = SplitStrategy.RANDOM,
) -> dict[str, Any]:
    """Validate a split workload and return a JSON-compatible manifest."""
    split_strategy = SplitStrategy(split_strategy)
    train_rows = tuple(train_rows)
    test_rows = tuple(test_rows)
    splits = {"train": train_rows, "dev": tuple(dev_rows), "test": test_rows}
    rows = tuple(row for selected in splits.values() for row in selected)
    ids = [str(row.get("id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row IDs")
    if any(not row.get("metadata", {}).get("round_trip_verified") for row in rows):
        raise ValueError("all rows must be round-trip verified")

    base_ids = {
        name: {str(row["metadata"]["base_id"]) for row in selected}
        for name, selected in splits.items()
    }
    prompts = {
        name: {_user_prompt(row) for row in selected}
        for name, selected in splits.items()
    }
    structures = {
        name: {
            str(row["metadata"]["structure_signature"])
            for row in selected
            if row["metadata"].get("structure_signature") is not None
        }
        for name, selected in splits.items()
    }
    structural_overlap: set[str] = set()
    split_names = tuple(splits)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if base_ids[left] & base_ids[right]:
                raise ValueError(f"cross-split base IDs: {left}/{right}")
            if prompts[left] & prompts[right]:
                raise ValueError(f"cross-split prompts: {left}/{right}")
            structural_overlap.update(structures[left] & structures[right])
    if split_strategy in {SplitStrategy.STRUCTURAL, SplitStrategy.HOLDOUT}:
        if any(not row["metadata"].get("structure_signature") for row in rows):
            raise ValueError("structural split requires every row to have a signature")
        if structural_overlap:
            raise ValueError(
                "cross-split structure signatures: "
                + ", ".join(sorted(structural_overlap))
            )

    prompt_bases: dict[str, set[str]] = defaultdict(set)
    variants_by_base: dict[str, set[str]] = defaultdict(set)
    row_variants_by_base: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    rows_by_base: dict[str, Mapping[str, Any]] = {}
    menus: dict[tuple[str, str, str], tuple[str, tuple[str, ...], str | None]] = {}
    for row in rows:
        metadata = row["metadata"]
        base_id = str(metadata["base_id"])
        prompt_bases[_user_prompt(row)].add(base_id)
        first_metadata = rows_by_base.setdefault(base_id, row)["metadata"]
        for field in (
            "structure_signature",
            "generation_provenance",
            "query_kind",
            "semantic_shape",
            "boolean_shape",
            "possible_values",
            "matrix_cell",
            "query_direction",
            "target_direction",
            "difficulty",
            "num_entities",
            "num_premises",
            "structure_profile",
            "seed",
            "sample_index",
            "attempt",
            "rejection_counts",
        ):
            if metadata.get(field) != first_metadata.get(field):
                raise ValueError(
                    f"paired variants disagree on {field} for base ID: {base_id}"
                )
        variants_by_base[base_id].add(str(metadata["trace_format"]))
        variant = (
            str(metadata["answer_mode"]),
            str(metadata["menu_coverage"]),
            str(metadata["trace_format"]),
            str(metadata.get("supervision_arm", "checked-trace")),
        )
        if variant in row_variants_by_base[base_id]:
            raise ValueError(
                f"duplicate answer/trace/control variant for base ID: {base_id}"
            )
        row_variants_by_base[base_id].add(variant)
        menu_key = (base_id, *variant[:2])
        letters = metadata.get("oracle_letters")
        if (
            not isinstance(letters, list)
            or not letters
            or any(
                not isinstance(letter, str)
                or len(letter) != 1
                or not "A" <= letter <= "Z"
                for letter in letters
            )
            or letters != sorted(set(letters))
        ):
            raise ValueError(f"invalid oracle_letters for base ID: {base_id}")
        menu = (_user_prompt(row), tuple(letters), metadata.get("evaluation_prompt"))
        if menus.setdefault(menu_key, menu) != menu:
            raise ValueError(
                f"paired variants disagree on prompt or answer for base ID: {base_id}"
            )
    if any(len(base_ids) != 1 for base_ids in prompt_bases.values()):
        raise ValueError("duplicate prompts belong to different base IDs")
    for row in rows:
        if (
            "evaluation_prompt" in row["metadata"]
            or "context_group_accepted" in row["metadata"]
        ):
            validate_context_admission(row)

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
        "generation_provenance",
        "query_kind",
        "query_direction",
        "target_direction",
        "semantic_shape",
        "boolean_shape",
    )
    answer_distribution_fields = (
        "matrix_answer_variant",
        "answer_mode",
        "menu_coverage",
        "menu_status",
    )
    trace_distribution_fields = (
        "trace_format",
        "supervision_arm",
        "expected_process_valid",
    )
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
        "support_checks_complete",
        "support_semantics",
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
    split_base_rows = {
        name: tuple(
            row for base_id, row in rows_by_base.items() if base_id in selected_ids
        )
        for name, selected_ids in base_ids.items()
    }
    return {
        "schema": "spatial-v2-workload-manifest",
        "total_rows": len(rows),
        "base_problems": len(variants_by_base),
        "splits": {name: len(selected) for name, selected in splits.items()},
        "base_splits": {name: len(selected) for name, selected in base_ids.items()},
        "trace_distributions": _distributions(rows, trace_distribution_fields),
        "supervision_volume": _supervision_volume(rows),
        "supervision_volume_by_variant": _supervision_volume(rows, by_variant=True),
        "split_supervision_volume": {
            name: _supervision_volume(selected) for name, selected in splits.items()
        },
        "split_supervision_volume_by_variant": {
            name: _supervision_volume(selected, by_variant=True)
            for name, selected in splits.items()
        },
        "answer_distributions": _distributions(rows, answer_distribution_fields),
        "answer_positions_by_cell": _answer_positions(rows),
        "marginal_answer_positions_by_cell": _answer_positions(rows, marginal=True),
        "split_answer_positions_by_cell": {
            name: _answer_positions(selected) for name, selected in splits.items()
        },
        "base_distributions": _distributions(base_rows, base_distribution_fields),
        "split_base_distributions": {
            name: _distributions(selected, base_distribution_fields)
            for name, selected in split_base_rows.items()
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
        "validation": {
            "status": "passed",
            "split_strategy": split_strategy.value,
            "structural_overlap": sorted(structural_overlap),
        },
    }


def _group_structure_signature(group: Sequence[Mapping[str, Any]]) -> str:
    signatures = {
        str(row.get("metadata", {}).get("structure_signature", "")) for row in group
    }
    if len(signatures) != 1 or not next(iter(signatures)):
        raise ValueError(
            "every row variant in a structural split group needs one shared signature"
        )
    return next(iter(signatures))


def _split_row_groups(
    row_groups: Sequence[list[dict]],
    test_split: float,
    seed: int,
    strategy: SplitStrategy,
    *,
    dev_split: float = 0,
    holdout_cells: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[list[dict]], list[list[dict]], list[list[dict]]]:
    if not 0 <= test_split < 1 or not 0 <= dev_split < 1 or test_split + dev_split >= 1:
        raise ValueError(
            "dev_split and test_split must be nonnegative and sum to less than one"
        )
    groups = list(row_groups)
    if any(not group for group in groups):
        raise ValueError("split groups must not be empty")
    random.Random(seed).shuffle(groups)
    if strategy is SplitStrategy.HOLDOUT:
        if not holdout_cells or set(holdout_cells) - {"dev", "test"}:
            raise ValueError("holdout_cells must name dev and/or test cells")
        dev_cells = set(holdout_cells.get("dev", ()))
        test_cells = set(holdout_cells.get("test", ()))
        if dev_cells & test_cells:
            raise ValueError("holdout dev/test cells must be disjoint")
        if bool(dev_cells) != bool(dev_split) or bool(test_cells) != bool(test_split):
            raise ValueError(
                "holdout cells and requested dev/test fractions must agree"
            )
        cells = []
        for group in groups:
            group_cells = {row["metadata"].get("matrix_cell") for row in group}
            if len(group_cells) != 1 or None in group_cells:
                raise ValueError("holdout requires one matrix cell per base group")
            cells.append(next(iter(group_cells)))
        unknown = (dev_cells | test_cells) - set(cells)
        if unknown:
            raise ValueError(
                "unknown or empty holdout cells: " + ", ".join(sorted(unknown))
            )
        selected = {"train": [], "dev": [], "test": []}
        for group, cell in zip(groups, cells, strict=True):
            name = (
                "dev"
                if cell in dev_cells
                else "test"
                if cell in test_cells
                else "train"
            )
            selected[name].append(group)
        if (
            not selected["train"]
            or (dev_split and not selected["dev"])
            or (test_split and not selected["test"])
        ):
            raise ValueError("holdout must leave every requested split nonempty")
        return selected["train"], selected["dev"], selected["test"]
    if holdout_cells:
        raise ValueError("holdout_cells requires the holdout split strategy")
    if not groups:
        if dev_split or test_split:
            raise ValueError("not enough base groups for requested splits")
        return [], [], []

    requested_splits = 1 + bool(dev_split) + bool(test_split)
    clusters: list[list[list[dict]]]
    if strategy is SplitStrategy.STRUCTURAL:
        by_structure: dict[str, list[list[dict]]] = defaultdict(list)
        for group in groups:
            by_structure[_group_structure_signature(group)].append(group)
        clusters = list(by_structure.values())
    else:
        clusters = [[group] for group in groups]
    if len(clusters) < requested_splits:
        raise ValueError(
            "not enough independent groups or structure signatures for requested splits"
        )

    def take_closest(target: int, reserve: int) -> list[list[dict]]:
        # Whole clusters only. Reserve at least one for each later nonempty split.
        chosen: list[list[dict]] = []
        while len(clusters) > reserve:
            current = len(chosen)
            best = min(
                range(len(clusters)),
                key=lambda i: abs(current + len(clusters[i]) - target),
            )
            if chosen and abs(current + len(clusters[best]) - target) >= abs(
                current - target
            ):
                break
            chosen.extend(clusters.pop(best))
        return chosen

    test = (
        take_closest(max(1, int(len(groups) * test_split)), 1 + bool(dev_split))
        if test_split
        else []
    )
    dev = take_closest(max(1, int(len(groups) * dev_split)), 1) if dev_split else []
    train = [group for cluster in clusters for group in cluster]
    return train, dev, test


def write_workload(
    output_file: str | Path,
    row_groups: list[list[dict]],
    *,
    test_split: float,
    seed: int,
    dev_split: float = 0,
    split_strategy: SplitStrategy | str = SplitStrategy.RANDOM,
    holdout_cells: Mapping[str, Sequence[str]] | None = None,
    expected_trace_variants: Iterable[Any],
    manifest_metadata: dict[str, Any],
    expected_variants_by_base: Mapping[str, Iterable[tuple[Any, Any, Any, Any]]]
    | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Validate and write paired train/dev/test data and its manifest."""
    split_strategy = SplitStrategy(split_strategy)
    train_groups, dev_groups, test_groups = _split_row_groups(
        row_groups,
        test_split,
        seed,
        split_strategy,
        dev_split=dev_split,
        holdout_cells=holdout_cells,
    )
    train_rows, dev_rows, test_rows = (
        [row for group in groups for row in group]
        for groups in (train_groups, dev_groups, test_groups)
    )
    manifest = build_workload_manifest(
        train_rows,
        test_rows,
        dev_rows=dev_rows,
        expected_trace_variants=expected_trace_variants,
        expected_variants_by_base=expected_variants_by_base,
        split_strategy=split_strategy,
    )
    overlap = set(manifest) & set(manifest_metadata)
    if overlap:
        raise ValueError(
            "manifest metadata cannot overwrite validation fields: "
            + ", ".join(sorted(overlap))
        )
    manifest.update(manifest_metadata)
    if holdout_cells:
        manifest["validation"]["holdout_cells"] = {
            name: list(cells) for name, cells in holdout_cells.items()
        }
    train_path, dev_path, test_path, manifest_path = workload_output_paths(output_file)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, selected in (
        (train_path, train_rows),
        (dev_path, dev_rows),
        (test_path, test_rows),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return train_path, dev_path, test_path, manifest_path
