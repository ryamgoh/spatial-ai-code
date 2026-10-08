"""Generate balanced, solver-verified SpatialMap V2 JSONL workloads."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from itertools import product
from pathlib import Path
from typing import TypeVar

import typer

from spatial.v2.generation import (
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

app = typer.Typer(add_completion=False)
EnumType = TypeVar("EnumType", bound=Enum)


@dataclass(frozen=True)
class WorkloadSpec:
    samples_per_cell: int = 100
    query_kinds: tuple[QueryKind, ...] = tuple(QueryKind)
    answer_modes: tuple[AnswerMode, ...] = (AnswerMode.SINGLE,)
    semantic_shapes: tuple[SemanticShape, ...] = (
        SemanticShape.UNIQUE,
        SemanticShape.AMBIGUOUS,
    )
    menu_coverages: tuple[MenuCoverage, ...] = (MenuCoverage.FULL,)
    trace_formats: tuple[TraceFormat, ...] = tuple(TraceFormat)
    query_directions: tuple[Direction, ...] | None = None
    target_directions: tuple[Direction, ...] | None = None
    num_entities: int = 6
    num_premises: int = 7
    omit_direct_query_relation: bool = False
    min_axis_depth: int = 1
    max_axis_depth: int | None = None
    require_independent_axes: bool = False
    ambiguity_size: int | None = None
    min_membership_depth: int = 1
    max_membership_depth: int | None = None
    distractor_premises: int = 0
    max_attempts_per_sample: int = 2_000
    test_split: float = 0.2
    seed: int = 42
    include_audit: bool = False
    replace: bool = False

    def __post_init__(self) -> None:
        if self.samples_per_cell < 0:
            raise ValueError("samples_per_cell cannot be negative")
        if self.max_attempts_per_sample <= 0:
            raise ValueError("max_attempts_per_sample must be positive")
        if not 0 <= self.test_split < 1:
            raise ValueError("test_split must be in [0, 1)")
        dimensions = {
            "query kind": self.query_kinds,
            "answer mode": self.answer_modes,
            "semantic shape": self.semantic_shapes,
            "menu coverage": self.menu_coverages,
            "trace format": self.trace_formats,
        }
        empty = [name for name, values in dimensions.items() if not values]
        if empty:
            raise ValueError("empty workload dimensions: " + ", ".join(empty))

    def manifest_config(self) -> dict:
        def serialize(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return [serialize(item) for item in value]
            return value

        config = {
            field.name: serialize(getattr(self, field.name)) for field in fields(self)
        }
        if not any(
            kind in {QueryKind.WHICH, QueryKind.COUNT} for kind in self.query_kinds
        ):
            config["query_directions"] = None
        if QueryKind.DIRECTION not in self.query_kinds:
            config["target_directions"] = None
        return config


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


def generate_workload(
    output_file: str | Path,
    spec: WorkloadSpec,
) -> tuple[Path, Path]:
    """Generate each requested policy cell equally, then shuffle and split."""
    output_paths = workload_output_paths(output_file)
    check_output_paths(output_paths, replace=spec.replace)
    generator = SpatialGeneratorV2(seed=spec.seed)
    row_groups: list[list[dict]] = []
    for dimensions in product(
        spec.query_kinds,
        spec.answer_modes,
        spec.semantic_shapes,
        spec.menu_coverages,
    ):
        query_kind, answer_mode, shape, coverage = dimensions
        if (
            query_kind is QueryKind.DIRECTION
            and shape is SemanticShape.UNIQUE
            and spec.target_directions
        ):
            direction_pairs = tuple(
                (None, direction) for direction in spec.target_directions
            )
        elif query_kind in {QueryKind.WHICH, QueryKind.COUNT} and spec.query_directions:
            direction_pairs = tuple(
                (direction, None) for direction in spec.query_directions
            )
        else:
            direction_pairs = ((None, None),)
        for query_direction, target_direction in direction_pairs:
            try:
                policy = GenerationPolicy(
                    query_kind=query_kind,
                    answer_mode=answer_mode,
                    semantic_shape=shape,
                    menu_coverage=coverage,
                    trace_format=spec.trace_formats[0],
                    num_entities=spec.num_entities,
                    num_premises=spec.num_premises,
                    query_direction=query_direction,
                    target_direction=target_direction,
                    omit_direct_query_relation=spec.omit_direct_query_relation,
                    min_axis_depth=spec.min_axis_depth,
                    max_axis_depth=spec.max_axis_depth,
                    require_independent_axes=spec.require_independent_axes,
                    ambiguity_size=spec.ambiguity_size,
                    min_membership_depth=spec.min_membership_depth,
                    max_membership_depth=spec.max_membership_depth,
                    distractor_premises=spec.distractor_premises,
                )
            except ValueError as exc:
                raise ValueError(
                    "invalid workload cell "
                    f"{query_kind.value}/{answer_mode.value}/{shape.value}/"
                    f"{coverage.value}: {exc}"
                ) from exc
            for _ in range(spec.samples_per_cell):
                sample = generator.generate(
                    policy, max_attempts=spec.max_attempts_per_sample
                )
                row_groups.append(
                    [
                        sample.with_trace(trace_format).as_sft_row(
                            include_audit=spec.include_audit
                        )
                        for trace_format in spec.trace_formats
                    ]
                )

    if spec.replace:
        remove_output_paths(output_paths)
    train_path, test_path, _manifest_path = write_workload(
        output_file,
        row_groups,
        test_split=spec.test_split,
        seed=spec.seed,
        expected_trace_variants=spec.trace_formats,
        manifest_metadata={"generation": spec.manifest_config()},
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
    replace: bool = typer.Option(
        False, help="Replace this workload's existing outputs."
    ),
) -> None:
    """Write balanced train/test workloads over the requested policy cells."""
    try:
        train_path, test_path = generate_workload(
            out,
            WorkloadSpec(
                samples_per_cell=samples_per_cell,
                query_kinds=_enum_values(query_kinds, QueryKind, "query kinds"),
                answer_modes=_enum_values(answer_modes, AnswerMode, "answer modes"),
                semantic_shapes=_enum_values(
                    semantic_shapes, SemanticShape, "semantic shapes"
                ),
                menu_coverages=_enum_values(
                    menu_coverages, MenuCoverage, "menu coverages"
                ),
                trace_formats=_enum_values(trace_formats, TraceFormat, "trace formats"),
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
                replace=replace,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"train: {train_path}")
    typer.echo(f"test: {test_path}")
    typer.echo(f"manifest: {workload_output_paths(out)[2]}")


if __name__ == "__main__":
    app()
