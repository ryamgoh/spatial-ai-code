"""Generate balanced, solver-verified SpatialMap V2 JSONL workloads."""

from __future__ import annotations

import json
import random
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

app = typer.Typer(add_completion=False)
EnumType = TypeVar("EnumType")


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
    num_entities: int = 6,
    num_premises: int = 7,
    test_split: float = 0.2,
    seed: int = 42,
    include_audit: bool = False,
) -> tuple[Path, Path]:
    """Generate each requested policy cell equally, then shuffle and split."""
    if samples_per_cell < 0:
        raise ValueError("samples_per_cell cannot be negative")
    if not 0 <= test_split < 1:
        raise ValueError("test_split must be in [0, 1)")
    generator = SpatialGeneratorV2(seed=seed)
    rows = []
    for dimensions in product(
        query_kinds,
        answer_modes,
        semantic_shapes,
        menu_coverages,
        trace_formats,
        state_modes,
    ):
        query_kind, answer_mode, shape, coverage, trace_format, state_mode = dimensions
        directions = (
            query_directions
            if query_kind in {QueryKind.WHICH, QueryKind.COUNT} and query_directions
            else (None,)
        )
        for query_direction in directions:
            try:
                policy = GenerationPolicy(
                    query_kind=query_kind,
                    answer_mode=answer_mode,
                    semantic_shape=shape,
                    menu_coverage=coverage,
                    trace_format=trace_format,
                    state_mode=state_mode,
                    num_entities=num_entities,
                    num_premises=num_premises,
                    query_direction=query_direction,
                )
            except ValueError as exc:
                raise ValueError(
                    "invalid workload cell "
                    f"{query_kind.value}/{answer_mode.value}/{shape.value}/"
                    f"{coverage.value}: {exc}"
                ) from exc
            for _ in range(samples_per_cell):
                rows.append(
                    generator.generate(policy).as_sft_row(include_audit=include_audit)
                )

    random.Random(seed).shuffle(rows)
    test_size = int(len(rows) * test_split)
    test_rows = rows[:test_size]
    train_rows = rows[test_size:]
    train_path, test_path = _split_paths(output_file)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, selected in ((train_path, train_rows), (test_path, test_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
    num_entities: int = typer.Option(6, min=2, max=30),
    num_premises: int = typer.Option(7, min=1),
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
            num_entities=num_entities,
            num_premises=num_premises,
            test_split=test_split,
            seed=seed,
            include_audit=include_audit,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"train: {train_path}")
    typer.echo(f"test: {test_path}")


if __name__ == "__main__":
    app()
