"""Generate balanced, solver-verified SpatialMap V2 JSONL workloads."""

from __future__ import annotations

import json
import random
from enum import Enum
from itertools import product
from pathlib import Path
from typing import TypeVar

import typer
from spatial_explanation_renderers_v2 import StateMode, TraceFormat
from spatial_generation_v2 import (
    GenerationPolicy,
    MenuCoverage,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
)
from spatial_grading_v2 import AnswerMode
from spatial_solver_v2 import Direction
from spatial_workload_manifest_v2 import build_workload_manifest

app = typer.Typer(add_completion=False)
EnumType = TypeVar("EnumType", bound=Enum)


def _enum_values(
    raw: str, enum_type: type[EnumType], label: str
) -> tuple[EnumType, ...]:
    requested = tuple(value.strip() for value in raw.split(",") if value.strip())
    known = {item.value.lower(): item for item in enum_type}
    unknown = sorted(value for value in requested if value.lower() not in known)
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    if not requested:
        raise ValueError(f"at least one {label} is required")
    return tuple(known[value.lower()] for value in requested)


def _split_paths(output_file: str | Path) -> tuple[Path, Path]:
    path = Path(output_file)
    suffix = path.suffix or ".jsonl"
    base = path.with_suffix("") if path.suffix else path
    return (
        base.with_name(base.name + "_train").with_suffix(suffix),
        base.with_name(base.name + "_test").with_suffix(suffix),
    )


def _manifest_path(output_file: str | Path) -> Path:
    path = Path(output_file)
    base = path.with_suffix("") if path.suffix else path
    return base.with_name(base.name + "_manifest").with_suffix(".json")


def generate_workload(
    output_file: str | Path,
    *,
    samples_per_cell: int,
    query_kinds: tuple[QueryKind, ...] = tuple(QueryKind),
    answer_modes: tuple[AnswerMode, ...] = (AnswerMode.SINGLE,),
    semantic_shapes: tuple[SemanticShape, ...] = (
        SemanticShape.UNIQUE,
        SemanticShape.AMBIGUOUS,
    ),
    menu_coverages: tuple[MenuCoverage, ...] = (MenuCoverage.FULL,),
    trace_formats: tuple[TraceFormat, ...] = tuple(TraceFormat),
    state_modes: tuple[StateMode, ...] = (StateMode.DELTA,),
    query_directions: tuple[Direction, ...] | None = None,
    target_directions: tuple[Direction, ...] | None = None,
    num_entities: int = 6,
    num_premises: int = 7,
    omit_direct_query_relation: bool = False,
    min_axis_depth: int = 1,
    max_axis_depth: int | None = None,
    require_independent_axes: bool = False,
    ambiguity_size: int | None = None,
    min_membership_depth: int = 1,
    max_membership_depth: int | None = None,
    distractor_premises: int = 0,
    max_attempts_per_sample: int = 2_000,
    test_split: float = 0.2,
    seed: int = 42,
    include_audit: bool = False,
) -> tuple[Path, Path]:
    """Generate each requested policy cell equally, then shuffle and split."""
    if samples_per_cell < 0:
        raise ValueError("samples_per_cell cannot be negative")
    if max_attempts_per_sample <= 0:
        raise ValueError("max_attempts_per_sample must be positive")
    if not 0 <= test_split < 1:
        raise ValueError("test_split must be in [0, 1)")
    generator = SpatialGeneratorV2(seed=seed)
    if not trace_formats or not state_modes:
        raise ValueError("at least one trace format and state mode is required")
    row_groups: list[list[dict]] = []
    for dimensions in product(
        query_kinds,
        answer_modes,
        semantic_shapes,
        menu_coverages,
    ):
        query_kind, answer_mode, shape, coverage = dimensions
        if (
            query_kind is QueryKind.DIRECTION
            and shape is SemanticShape.UNIQUE
            and target_directions
        ):
            direction_pairs = tuple(
                (None, direction) for direction in target_directions
            )
        elif query_kind in {QueryKind.WHICH, QueryKind.COUNT} and query_directions:
            direction_pairs = tuple((direction, None) for direction in query_directions)
        else:
            direction_pairs = ((None, None),)
        for query_direction, target_direction in direction_pairs:
            try:
                policy = GenerationPolicy(
                    query_kind=query_kind,
                    answer_mode=answer_mode,
                    semantic_shape=shape,
                    menu_coverage=coverage,
                    trace_format=trace_formats[0],
                    state_mode=state_modes[0],
                    num_entities=num_entities,
                    num_premises=num_premises,
                    query_direction=query_direction,
                    target_direction=target_direction,
                    omit_direct_query_relation=omit_direct_query_relation,
                    min_axis_depth=min_axis_depth,
                    max_axis_depth=max_axis_depth,
                    require_independent_axes=require_independent_axes,
                    ambiguity_size=ambiguity_size,
                    min_membership_depth=min_membership_depth,
                    max_membership_depth=max_membership_depth,
                    distractor_premises=distractor_premises,
                )
            except ValueError as exc:
                raise ValueError(
                    "invalid workload cell "
                    f"{query_kind.value}/{answer_mode.value}/{shape.value}/"
                    f"{coverage.value}: {exc}"
                ) from exc
            for _ in range(samples_per_cell):
                sample = generator.generate(
                    policy, max_attempts=max_attempts_per_sample
                )
                row_groups.append(
                    [
                        sample.with_trace(trace_format, state_mode).as_sft_row(
                            include_audit=include_audit
                        )
                        for trace_format, state_mode in product(
                            trace_formats, state_modes
                        )
                    ]
                )

    random.Random(seed).shuffle(row_groups)
    test_size = int(len(row_groups) * test_split)
    test_rows = [row for group in row_groups[:test_size] for row in group]
    train_rows = [row for group in row_groups[test_size:] for row in group]
    expected_variants = tuple(product(trace_formats, state_modes))
    manifest = build_workload_manifest(
        train_rows,
        test_rows,
        expected_trace_variants=expected_variants,
    )
    manifest["generation"] = {
        "samples_per_cell": samples_per_cell,
        "query_kinds": [value.value for value in query_kinds],
        "answer_modes": [value.value for value in answer_modes],
        "semantic_shapes": [value.value for value in semantic_shapes],
        "menu_coverages": [value.value for value in menu_coverages],
        "trace_formats": [value.value for value in trace_formats],
        "state_modes": [value.value for value in state_modes],
        "query_directions": (
            [value.value for value in query_directions]
            if query_directions is not None
            and any(kind in {QueryKind.WHICH, QueryKind.COUNT} for kind in query_kinds)
            else None
        ),
        "target_directions": (
            [value.value for value in target_directions]
            if target_directions is not None and QueryKind.DIRECTION in query_kinds
            else None
        ),
        "num_entities": num_entities,
        "num_premises": num_premises,
        "omit_direct_query_relation": omit_direct_query_relation,
        "min_axis_depth": min_axis_depth,
        "max_axis_depth": max_axis_depth,
        "require_independent_axes": require_independent_axes,
        "ambiguity_size": ambiguity_size,
        "min_membership_depth": min_membership_depth,
        "max_membership_depth": max_membership_depth,
        "distractor_premises": distractor_premises,
        "max_attempts_per_sample": max_attempts_per_sample,
        "test_split": test_split,
        "seed": seed,
        "include_audit": include_audit,
    }
    train_path, test_path = _split_paths(output_file)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, selected in ((train_path, train_rows), (test_path, test_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _manifest_path(output_file).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return train_path, test_path


@app.command()
def main(
    out: Path = typer.Option(..., help="Output JSONL base path."),  # noqa: B008
    samples_per_cell: int = typer.Option(100, min=0),
    query_kinds: str = typer.Option("direction,which,count"),
    answer_modes: str = typer.Option("single"),
    semantic_shapes: str = typer.Option("unique,ambiguous"),
    menu_coverages: str = typer.Option("full"),
    trace_formats: str = typer.Option("natural,symbolic"),
    state_modes: str = typer.Option("delta"),
    query_directions: str = typer.Option(
        "north,northeast,east,southeast,south,southwest,west,northwest",
        help="Balanced Which/Count query directions; ignored for Direction queries.",
    ),
    target_directions: str = typer.Option(
        "north,northeast,east,southeast,south,southwest,west,northwest",
        help="Balanced Direction answers; ignored for Which/Count queries.",
    ),
    num_entities: int = typer.Option(6, min=2, max=30),
    num_premises: int = typer.Option(7, min=1),
    omit_direct_query_relation: bool = typer.Option(False),
    min_axis_depth: int = typer.Option(1, min=1),
    max_axis_depth: int | None = typer.Option(None, min=1),
    require_independent_axes: bool = typer.Option(False),
    ambiguity_size: int | None = typer.Option(None, min=2),
    min_membership_depth: int = typer.Option(1, min=1),
    max_membership_depth: int | None = typer.Option(None, min=1),
    distractor_premises: int = typer.Option(0, min=0),
    max_attempts_per_sample: int = typer.Option(2_000, min=1),
    test_split: float = typer.Option(0.2, min=0.0, max=0.999999),
    seed: int = typer.Option(42),
    include_audit: bool = typer.Option(
        False,
        help="Include coordinate-bearing audit metadata. Never enable for SFT data.",
    ),
) -> None:
    """Write balanced train/test workloads over the requested policy cells."""
    try:
        train_path, test_path = generate_workload(
            out,
            samples_per_cell=samples_per_cell,
            query_kinds=_enum_values(query_kinds, QueryKind, "query kinds"),
            answer_modes=_enum_values(answer_modes, AnswerMode, "answer modes"),
            semantic_shapes=_enum_values(
                semantic_shapes, SemanticShape, "semantic shapes"
            ),
            menu_coverages=_enum_values(menu_coverages, MenuCoverage, "menu coverages"),
            trace_formats=_enum_values(trace_formats, TraceFormat, "trace formats"),
            state_modes=_enum_values(state_modes, StateMode, "state modes"),
            query_directions=_enum_values(
                query_directions, Direction, "query directions"
            ),
            target_directions=_enum_values(
                target_directions, Direction, "target directions"
            ),
            num_entities=num_entities,
            num_premises=num_premises,
            omit_direct_query_relation=omit_direct_query_relation,
            min_axis_depth=min_axis_depth,
            max_axis_depth=max_axis_depth,
            require_independent_axes=require_independent_axes,
            ambiguity_size=ambiguity_size,
            min_membership_depth=min_membership_depth,
            max_membership_depth=max_membership_depth,
            distractor_premises=distractor_premises,
            max_attempts_per_sample=max_attempts_per_sample,
            test_split=test_split,
            seed=seed,
            include_audit=include_audit,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"train: {train_path}")
    typer.echo(f"test: {test_path}")
    typer.echo(f"manifest: {_manifest_path(out)}")


if __name__ == "__main__":
    app()
