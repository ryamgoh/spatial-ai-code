"""Generate balanced, solver-verified SpatialMap V2 JSONL workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from itertools import product
from pathlib import Path
from typing import TypeVar

import typer

from spatial.v2.context_budget import (
    ContextBudget,
    admit_row_group,
    load_tokenizer,
    rejection_output_path,
    summarize_admissions,
    write_rejection_report,
)
from spatial.v2.generation import (
    BooleanShape,
    GenerationMode,
    GenerationPolicy,
    MenuCoverage,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.solver import Direction
from spatial.v2.supervision_controls import (
    SupervisionArm,
    build_supervision_variants,
)
from spatial.v2.trace import TraceFormat
from spatial.v2.workload_manifest import (
    SplitStrategy,
    check_output_paths,
    remove_output_paths,
    workload_output_paths,
    write_workload,
)

app = typer.Typer(add_completion=False)
EnumType = TypeVar("EnumType", bound=Enum)


@dataclass(frozen=True)
class WorkloadSpec:
    generation_mode: GenerationMode = GenerationMode.AUTO
    samples_per_cell: int = 100
    query_kinds: tuple[QueryKind, ...] = tuple(QueryKind)
    answer_modes: tuple[AnswerMode, ...] = (AnswerMode.SINGLE,)
    semantic_shapes: tuple[SemanticShape, ...] = (
        SemanticShape.UNIQUE,
        SemanticShape.AMBIGUOUS,
    )
    menu_coverages: tuple[MenuCoverage, ...] = (MenuCoverage.FULL,)
    trace_formats: tuple[TraceFormat, ...] = tuple(TraceFormat)
    supervision_arms: tuple[SupervisionArm, ...] = (SupervisionArm.CHECKED_TRACE,)
    boolean_shapes: tuple[BooleanShape, ...] = (BooleanShape.ATOMIC,)
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
    dev_split: float = 0
    context_budget: ContextBudget | None = None
    test_split: float = 0.2
    split_strategy: SplitStrategy = SplitStrategy.RANDOM
    seed: int = 42
    include_audit: bool = False
    replace: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_strategy", SplitStrategy(self.split_strategy))
        object.__setattr__(
            self, "generation_mode", GenerationMode(self.generation_mode)
        )
        if self.samples_per_cell < 0:
            raise ValueError("samples_per_cell cannot be negative")
        if self.max_attempts_per_sample <= 0:
            raise ValueError("max_attempts_per_sample must be positive")
        if not 0 <= self.dev_split < 1 or self.dev_split + self.test_split >= 1:
            raise ValueError("dev_split and test_split must sum to less than 1")
        if not 0 <= self.test_split < 1:
            raise ValueError("test_split must be in [0, 1)")
        dimensions = {
            "query kind": self.query_kinds,
            "answer mode": self.answer_modes,
            "semantic shape": self.semantic_shapes,
            "menu coverage": self.menu_coverages,
            "trace format": self.trace_formats,
            "supervision arm": self.supervision_arms,
            "Boolean shape": self.boolean_shapes,
        }
        empty = [name for name, values in dimensions.items() if not values]
        if empty:
            raise ValueError("empty workload dimensions: " + ", ".join(empty))
        normalized_arms = tuple(SupervisionArm(arm) for arm in self.supervision_arms)
        if len(normalized_arms) != len(set(normalized_arms)):
            raise ValueError("supervision arms must be unique")
        object.__setattr__(self, "supervision_arms", normalized_arms)
        if (
            SupervisionArm.CORRUPTED_TRACE in normalized_arms
            and TraceFormat.SYMBOLIC not in self.trace_formats
        ):
            raise ValueError("corrupted-trace requires a symbolic trace variant")
        if (
            self.generation_mode != "premise-first"
            and any(shape is not BooleanShape.ATOMIC for shape in self.boolean_shapes)
            and (set(self.semantic_shapes) != {SemanticShape.UNIQUE})
        ):
            raise ValueError("non-atomic Boolean shapes require unique semantic cells")

    def manifest_config(self) -> dict:
        def serialize(value):
            if isinstance(value, ContextBudget):
                return asdict(value)
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
) -> tuple[Path, Path, Path]:
    """Generate each requested policy cell equally, then shuffle and split."""
    output_paths = (
        *workload_output_paths(output_file),
        rejection_output_path(output_file),
    )
    check_output_paths(output_paths, replace=spec.replace)
    tokenizer = load_tokenizer(spec.context_budget) if spec.context_budget else None
    admissions = []
    generator = SpatialGeneratorV2(seed=spec.seed)
    row_groups: list[list[dict]] = []
    expected_variants_by_base = {}
    for dimensions in product(
        spec.query_kinds,
        spec.answer_modes,
        spec.semantic_shapes,
        spec.menu_coverages,
        spec.boolean_shapes,
    ):
        query_kind, answer_mode, shape, coverage, boolean_shape = dimensions
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
                    generation_mode=spec.generation_mode,
                    query_kind=query_kind,
                    answer_mode=answer_mode,
                    semantic_shape=shape,
                    menu_coverage=coverage,
                    trace_format=spec.trace_formats[0],
                    boolean_shape=boolean_shape,
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
                checked_rows = [
                    sample.with_trace(trace_format).as_sft_row(
                        include_audit=spec.include_audit
                    )
                    for trace_format in spec.trace_formats
                ]
                variants = []
                for index, row in enumerate(checked_rows):
                    arms = tuple(
                        arm
                        for arm in spec.supervision_arms
                        if (arm is not SupervisionArm.ANSWER_ONLY or index == 0)
                        and (
                            arm is not SupervisionArm.CORRUPTED_TRACE
                            or row["metadata"]["trace_format"] == "symbolic"
                        )
                    )
                    if arms:
                        variants.extend(
                            build_supervision_variants(
                                row,
                                arms,
                                problem=sample.problem,
                                expected=sample.menu_answer,
                            )
                        )
                if spec.context_budget:
                    admission = admit_row_group(
                        variants, tokenizer, spec.context_budget
                    )
                    admissions.append(admission)
                    if not admission.accepted:
                        write_rejection_report(
                            output_file,
                            admissions,
                            spec.context_budget,
                            cell="/".join(str(value.value) for value in dimensions),
                        )
                        raise ValueError(
                            f"cell {dimensions}: paired context admission failed: {admission.row_rejections}"
                        )
                    variants = list(admission.rows)
                row_groups.append(variants)
                expected_variants_by_base[sample.base_id] = {
                    (answer_mode, coverage, trace, arm)
                    for index, trace in enumerate(spec.trace_formats)
                    for arm in spec.supervision_arms
                    if (arm is not SupervisionArm.ANSWER_ONLY or index == 0)
                    and (
                        arm is not SupervisionArm.CORRUPTED_TRACE
                        or trace is TraceFormat.SYMBOLIC
                    )
                }

    train_path, dev_path, test_path, _manifest_path = write_workload(
        output_file,
        row_groups,
        test_split=spec.test_split,
        dev_split=spec.dev_split,
        seed=spec.seed,
        split_strategy=spec.split_strategy,
        expected_trace_variants={
            variant[2]
            for variants in expected_variants_by_base.values()
            for variant in variants
        },
        expected_variants_by_base=expected_variants_by_base,
        manifest_metadata={
            "generation": spec.manifest_config(),
            **(
                {
                    "context_admission": summarize_admissions(
                        admissions, spec.context_budget
                    )
                }
                if spec.context_budget
                else {}
            ),
        },
    )
    if spec.replace:
        remove_output_paths((rejection_output_path(output_file),))
    return train_path, dev_path, test_path


@app.command()
def main(
    out: Path = typer.Option(..., help="Output JSONL base path."),  # noqa: B008
    generation_mode: GenerationMode = typer.Option(GenerationMode.AUTO),  # noqa: B008
    samples_per_cell: int = typer.Option(100, min=0),
    query_kinds: str = typer.Option("direction,which,count"),
    answer_modes: str = typer.Option("single"),
    semantic_shapes: str = typer.Option("unique,ambiguous"),
    menu_coverages: str = typer.Option("full"),
    trace_formats: str = typer.Option("natural,symbolic"),
    supervision_arms: str = typer.Option("checked-trace"),
    boolean_shapes: str = typer.Option("atomic"),
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
    dev_split: float = typer.Option(0, min=0.0, max=0.999999),
    tokenizer_name: str | None = typer.Option(None),
    train_max_tokens: int = typer.Option(4096, min=1),
    eval_max_tokens: int = typer.Option(8192, min=1),
    max_new_tokens: int = typer.Option(4096, min=1),
    test_split: float = typer.Option(0.2, min=0.0, max=0.999999),
    split_strategy: SplitStrategy = typer.Option(SplitStrategy.RANDOM),  # noqa: B008
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
        train_path, dev_path, test_path = generate_workload(
            out,
            WorkloadSpec(
                generation_mode=generation_mode,
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
                supervision_arms=_enum_values(
                    supervision_arms,
                    SupervisionArm,
                    "supervision arms",
                ),
                boolean_shapes=_enum_values(
                    boolean_shapes,
                    BooleanShape,
                    "Boolean shapes",
                ),
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
                dev_split=dev_split,
                context_budget=ContextBudget(
                    tokenizer_name, train_max_tokens, eval_max_tokens, max_new_tokens
                )
                if tokenizer_name
                else None,
                split_strategy=split_strategy,
                seed=seed,
                include_audit=include_audit,
                replace=replace,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"train: {train_path}")
    typer.echo(f"dev: {dev_path}")
    typer.echo(f"test: {test_path}")
    typer.echo(f"manifest: {workload_output_paths(out)[3]}")


if __name__ == "__main__":
    app()
