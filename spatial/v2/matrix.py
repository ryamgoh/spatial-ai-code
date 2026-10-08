"""YAML experiment matrices for reproducible SpatialMap V2 ablations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, TypeVar

import yaml

from spatial.v2.generation import (
    BooleanShape,
    GenerationPolicy,
    MenuCoverage,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.solver import Direction
from spatial.v2.trace import TraceFormat
from spatial.v2.workload_manifest import (
    check_output_paths,
    remove_output_paths,
    workload_output_paths,
    write_workload,
)

EnumValue = TypeVar("EnumValue")
_ROOT_KEYS = {
    "version",
    "seed",
    "test_split",
    "include_audit",
    "defaults",
    "variants",
    "cells",
}
_CELL_KEYS = {
    "name",
    "count",
    "query_kind",
    "semantic_shape",
    "boolean_shape",
    "depth",
    "ambiguity_size",
    "num_entities",
    "num_premises",
    "omit_direct_query_relation",
    "require_independent_axes",
    "distractor_premises",
    "max_attempts_per_sample",
    "query_directions",
    "target_directions",
    "answer_variants",
}
_DEFAULT_KEYS = _CELL_KEYS - {"name", "count", "query_kind", "semantic_shape"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class AnswerVariant:
    name: str
    mode: AnswerMode
    menu_coverage: MenuCoverage


@dataclass(frozen=True)
class TraceVariant:
    trace_format: TraceFormat


@dataclass(frozen=True)
class MatrixCell:
    name: str
    count: int
    query_kind: QueryKind
    semantic_shape: SemanticShape
    boolean_shape: BooleanShape
    depth: int | None
    ambiguity_size: int | None
    num_entities: int
    num_premises: int
    omit_direct_query_relation: bool
    require_independent_axes: bool
    distractor_premises: int
    max_attempts_per_sample: int
    query_directions: tuple[Direction, ...] | None
    target_directions: tuple[Direction, ...] | None
    answer_variants: tuple[str, ...] | None


@dataclass(frozen=True)
class ExperimentMatrix:
    seed: int
    test_split: float
    include_audit: bool
    answer_variants: tuple[AnswerVariant, ...]
    trace_variants: tuple[TraceVariant, ...]
    cells: tuple[MatrixCell, ...]

    def manifest_config(self) -> dict[str, Any]:
        def serialize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: serialize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serialize(item) for item in value]
            return value

        return serialize(asdict(self))


def _enum(value: Any, enum_type: type[EnumValue], field: str) -> EnumValue:
    normalized = str(value).strip().lower()
    for member in enum_type:
        if str(member.value).lower() == normalized:
            return member
    raise ValueError(f"unknown {field}: {value}")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")  # noqa: TRY004
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")  # noqa: TRY004
    return value


def _directions(value: Any, field: str) -> tuple[Direction, ...] | None:
    if value is None:
        return None
    return tuple(_enum(item, Direction, field) for item in _sequence(value, field))


def _name(value: Any, field: str) -> str:
    name = str(value or "").strip()
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            f"{field} must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    return name


def _answer_variants(raw: dict[str, Any]) -> tuple[AnswerVariant, ...]:
    items = _sequence(raw.get("answers"), "variants.answers")
    variants = []
    for item in items:
        item = _mapping(item, "answer variant")
        unknown = set(item) - {"name", "mode", "menu_coverage"}
        if unknown:
            raise ValueError(
                "unknown answer variant keys: " + ", ".join(sorted(unknown))
            )
        mode = _enum(item.get("mode"), AnswerMode, "answer mode")
        coverage = _enum(
            item.get("menu_coverage", "full"), MenuCoverage, "menu coverage"
        )
        name = _name(
            item.get("name") or f"{mode.value}-{coverage.value}",
            "answer variant name",
        )
        variants.append(AnswerVariant(name, mode, coverage))
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("duplicate answer variant names")
    if len({(variant.mode, variant.menu_coverage) for variant in variants}) != len(
        variants
    ):
        raise ValueError("duplicate answer variants")
    return tuple(variants)


def _trace_variants(raw: dict[str, Any]) -> tuple[TraceVariant, ...]:
    items = _sequence(raw.get("traces"), "variants.traces")
    variants = []
    for item in items:
        item = _mapping(item, "trace variant")
        unknown = set(item) - {"format"}
        if unknown:
            raise ValueError(
                "unknown trace variant keys: " + ", ".join(sorted(unknown))
            )
        variants.append(
            TraceVariant(_enum(item.get("format"), TraceFormat, "trace format"))
        )
    if len(set(variants)) != len(variants):
        raise ValueError("duplicate trace variants")
    return tuple(variants)


def _cell(raw: dict[str, Any], defaults: dict[str, Any]) -> MatrixCell:
    unknown = set(raw) - _CELL_KEYS
    if unknown:
        raise ValueError("unknown cell keys: " + ", ".join(sorted(unknown)))
    values = {**defaults, **raw}
    name = _name(values.get("name"), "cell name")
    count = int(values.get("count", 0))
    if count <= 0:
        raise ValueError(f"cell {name} count must be positive")
    query_kind = _enum(values.get("query_kind"), QueryKind, "query kind")
    semantic_shape = _enum(
        values.get("semantic_shape"), SemanticShape, "semantic shape"
    )
    depth = int(values["depth"]) if values.get("depth") is not None else None
    if depth is not None and depth <= 0:
        raise ValueError(f"cell {name} depth must be positive")
    return MatrixCell(
        name=name,
        count=count,
        query_kind=query_kind,
        semantic_shape=semantic_shape,
        boolean_shape=_enum(
            values.get("boolean_shape", "atomic"),
            BooleanShape,
            "Boolean shape",
        ),
        depth=depth,
        ambiguity_size=(
            int(values["ambiguity_size"])
            if values.get("ambiguity_size") is not None
            else None
        ),
        num_entities=int(values.get("num_entities", 6)),
        num_premises=int(values.get("num_premises", 7)),
        omit_direct_query_relation=_boolean(
            values.get("omit_direct_query_relation", False),
            "omit_direct_query_relation",
        ),
        require_independent_axes=_boolean(
            values.get("require_independent_axes", False),
            "require_independent_axes",
        ),
        distractor_premises=int(values.get("distractor_premises", 0)),
        max_attempts_per_sample=int(values.get("max_attempts_per_sample", 2_000)),
        query_directions=_directions(
            values.get("query_directions"), "query directions"
        ),
        target_directions=_directions(
            values.get("target_directions"), "target directions"
        ),
        answer_variants=(
            tuple(
                str(name)
                for name in _sequence(values["answer_variants"], "cell answer_variants")
            )
            if values.get("answer_variants") is not None
            else None
        ),
    )


def load_experiment_matrix(path: str | Path) -> ExperimentMatrix:
    """Load and strictly validate a version-1 ablation matrix."""
    try:
        source = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid matrix YAML: {exc}") from exc
    root = _mapping(source, "matrix")
    unknown = set(root) - _ROOT_KEYS
    if unknown:
        raise ValueError("unknown matrix keys: " + ", ".join(sorted(unknown)))
    if root.get("version") != 1:
        raise ValueError("matrix version must be 1")
    defaults = _mapping(root.get("defaults", {}), "defaults")
    unknown_defaults = set(defaults) - _DEFAULT_KEYS
    if unknown_defaults:
        raise ValueError("unknown default keys: " + ", ".join(sorted(unknown_defaults)))
    variants = _mapping(root.get("variants"), "variants")
    unknown_variants = set(variants) - {"answers", "traces"}
    if unknown_variants:
        raise ValueError("unknown variant keys: " + ", ".join(sorted(unknown_variants)))
    cells = tuple(
        _cell(_mapping(item, "cell"), defaults)
        for item in _sequence(root.get("cells"), "cells")
    )
    names = [cell.name for cell in cells]
    if len(names) != len(set(names)):
        raise ValueError("duplicate cell names")
    test_split = float(root.get("test_split", 0.2))
    if not 0 <= test_split < 1:
        raise ValueError("test_split must be in [0, 1)")
    matrix = ExperimentMatrix(
        seed=int(root.get("seed", 42)),
        test_split=test_split,
        include_audit=_boolean(root.get("include_audit", False), "include_audit"),
        answer_variants=_answer_variants(variants),
        trace_variants=_trace_variants(variants),
        cells=cells,
    )
    _validate_matrix_cells(matrix)
    return matrix


def _direction_pairs(
    cell: MatrixCell,
) -> tuple[tuple[Direction | None, Direction | None], ...]:
    if cell.query_kind is QueryKind.DIRECTION:
        return (
            tuple((None, direction) for direction in cell.target_directions)
            if cell.target_directions
            else ((None, None),)
        )
    return (
        tuple((direction, None) for direction in cell.query_directions)
        if cell.query_directions
        else ((None, None),)
    )


def _policy(
    cell: MatrixCell,
    answer: AnswerVariant,
    trace: TraceVariant,
    query_direction: Direction | None,
    target_direction: Direction | None,
) -> GenerationPolicy:
    depth_args = {}
    if cell.depth is not None:
        depth_args = (
            {"min_axis_depth": cell.depth, "max_axis_depth": cell.depth}
            if cell.query_kind is QueryKind.DIRECTION
            else {
                "min_membership_depth": cell.depth,
                "max_membership_depth": cell.depth,
            }
        )
    return GenerationPolicy(
        query_kind=cell.query_kind,
        answer_mode=answer.mode,
        semantic_shape=cell.semantic_shape,
        boolean_shape=cell.boolean_shape,
        menu_coverage=answer.menu_coverage,
        trace_format=trace.trace_format,
        num_entities=cell.num_entities,
        num_premises=cell.num_premises,
        query_direction=query_direction,
        target_direction=target_direction,
        omit_direct_query_relation=cell.omit_direct_query_relation,
        require_independent_axes=cell.require_independent_axes,
        ambiguity_size=cell.ambiguity_size,
        distractor_premises=cell.distractor_premises,
        **depth_args,
    )


def _answers_for_cell(
    matrix: ExperimentMatrix,
    cell: MatrixCell,
) -> tuple[AnswerVariant, ...]:
    if cell.answer_variants is None:
        return matrix.answer_variants
    by_name = {variant.name: variant for variant in matrix.answer_variants}
    unknown = sorted(set(cell.answer_variants) - set(by_name))
    if unknown:
        raise ValueError(
            f"cell {cell.name}: unknown answer variants: {', '.join(unknown)}"
        )
    if len(cell.answer_variants) != len(set(cell.answer_variants)):
        raise ValueError(f"cell {cell.name}: duplicate answer variant references")
    return tuple(by_name[name] for name in cell.answer_variants)


def _validate_matrix_cells(matrix: ExperimentMatrix) -> None:
    first_trace = matrix.trace_variants[0]
    for cell in matrix.cells:
        answers = _answers_for_cell(matrix, cell)
        if cell.query_kind is QueryKind.DIRECTION and cell.query_directions:
            raise ValueError(f"cell {cell.name}: query_directions apply to Which/Count")
        if cell.query_kind is not QueryKind.DIRECTION and cell.target_directions:
            raise ValueError(f"cell {cell.name}: target_directions apply to Direction")
        if cell.semantic_shape is SemanticShape.UNIQUE and any(
            variant.menu_coverage is MenuCoverage.PARTIAL for variant in answers
        ):
            raise ValueError(
                f"cell {cell.name}: partial menu coverage requires ambiguous semantics"
            )
        for answer, directions in product(
            answers,
            _direction_pairs(cell),
        ):
            query_direction, target_direction = directions
            try:
                _policy(
                    cell,
                    answer,
                    first_trace,
                    query_direction,
                    target_direction,
                )
            except ValueError as exc:
                raise ValueError(f"cell {cell.name}: {exc}") from exc


def generate_matrix(
    matrix_path: str | Path,
    output_file: str | Path,
    *,
    replace: bool = False,
) -> tuple[Path, Path, Path]:
    """Generate all requested matrix cells and their paired variants."""
    matrix = load_experiment_matrix(matrix_path)
    output_paths = (*workload_output_paths(output_file), _view_root(output_file))
    check_output_paths(output_paths, replace=replace)
    generator = SpatialGeneratorV2(matrix.seed)
    row_groups: list[list[dict]] = []
    cell_counts: dict[str, int] = {}
    first_trace = matrix.trace_variants[0]
    expected_variants_by_base: dict[
        str, set[tuple[AnswerMode, MenuCoverage, TraceFormat]]
    ] = {}

    for cell in matrix.cells:
        answers = _answers_for_cell(matrix, cell)
        first_answer = answers[0]
        generated = 0
        for query_direction, target_direction in _direction_pairs(cell):
            base_policy = _policy(
                cell,
                first_answer,
                first_trace,
                query_direction,
                target_direction,
            )
            for _ in range(cell.count):
                base = generator.generate(
                    base_policy,
                    max_attempts=cell.max_attempts_per_sample,
                )
                rows = []
                for answer, trace in product(
                    answers,
                    matrix.trace_variants,
                ):
                    answer_sample = generator.with_answer_mode(
                        base,
                        answer.mode,
                        answer.menu_coverage,
                    )
                    variant = answer_sample.with_trace(trace.trace_format)
                    row = variant.as_sft_row(include_audit=matrix.include_audit)
                    row["metadata"]["matrix_cell"] = cell.name
                    row["metadata"]["matrix_answer_variant"] = answer.name
                    rows.append(row)
                row_groups.append(rows)
                expected_variants_by_base[base.base_id] = {
                    (
                        answer.mode,
                        answer.menu_coverage,
                        trace.trace_format,
                    )
                    for answer, trace in product(answers, matrix.trace_variants)
                }
                generated += 1
        cell_counts[cell.name] = generated

    trace_variants = tuple(variant.trace_format for variant in matrix.trace_variants)
    matrix_config = matrix.manifest_config()
    matrix_config.update(
        {
            "requested_base_problems": sum(cell_counts.values()),
            "generated_rows": sum(len(group) for group in row_groups),
            "cells": cell_counts,
        }
    )
    if replace:
        remove_output_paths(output_paths)
    paths = write_workload(
        output_file,
        row_groups,
        test_split=matrix.test_split,
        seed=matrix.seed,
        expected_trace_variants=trace_variants,
        expected_variants_by_base=expected_variants_by_base,
        manifest_metadata={"matrix": matrix_config},
    )
    _materialize_views(*paths, _view_root(output_file))
    return paths


def _view_root(output_file: str | Path) -> Path:
    path = Path(output_file)
    base = path.with_suffix("") if path.suffix else path
    return base.with_name(base.name + "_views")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _variant_key(row: dict[str, Any]) -> str:
    metadata = row["metadata"]
    return "__".join(
        (
            metadata["matrix_answer_variant"],
            metadata["trace_format"],
        )
    )


def _split_fingerprint(rows: list[dict[str, Any]]) -> str:
    base_ids = "\n".join(sorted({row["metadata"]["base_id"] for row in rows}))
    return hashlib.sha256(base_ids.encode()).hexdigest()


def _materialize_views(
    train_path: Path,
    test_path: Path,
    manifest_path: Path,
    view_root: Path,
) -> None:
    splits = {"train": _read_jsonl(train_path), "test": _read_jsonl(test_path)}
    by_variant: dict[str, dict[str, list[dict[str, Any]]]] = {}
    by_cell: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for split, rows in splits.items():
        for row in rows:
            variant = _variant_key(row)
            cell = row["metadata"]["matrix_cell"]
            by_variant.setdefault(variant, {"train": [], "test": []})[split].append(row)
            by_cell.setdefault(cell, {}).setdefault(variant, {"train": [], "test": []})[
                split
            ].append(row)

    def write_views(
        directory: Path,
        groups: dict[str, dict[str, list[dict[str, Any]]]],
    ) -> dict[str, dict[str, Any]]:
        outputs = {}
        for name, rows_by_split in sorted(groups.items()):
            paths = {
                split: directory / f"{name}_{split}.jsonl"
                for split in ("train", "test")
            }
            for split, path in paths.items():
                _write_jsonl(path, rows_by_split[split])
            outputs[name] = {
                "train": str(paths["train"].relative_to(manifest_path.parent)),
                "test": str(paths["test"].relative_to(manifest_path.parent)),
                "train_rows": len(rows_by_split["train"]),
                "test_rows": len(rows_by_split["test"]),
            }
        return outputs

    views = {
        "root": str(view_root.relative_to(manifest_path.parent)),
        "split_fingerprint": {
            split: _split_fingerprint(rows) for split, rows in splits.items()
        },
        "by_variant": write_views(view_root / "by_variant", by_variant),
        "by_cell": {
            cell: write_views(view_root / "by_cell" / cell, groups)
            for cell, groups in sorted(by_cell.items())
        },
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["views"] = views
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
