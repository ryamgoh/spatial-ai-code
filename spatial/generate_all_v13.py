"""CLI and compatibility adapters for v13 synthetic generation.

New code should use ``GenerationSpec`` + ``SpatialGenerator.generate`` and
``GenerationCell`` + ``generate_dataset`` from ``spatial_generation_v13``.
This module keeps the command line and the early v13 function interfaces
stable while translating them into that deeper interface.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import typer

from spatial_generation_v13 import (
    DEFAULT_SYSTEM_PROMPT,
    SEMANTIC_SUBTYPES,
    GenerationCell,
    GenerationError,
    GenerationSpec,
    RelationMode,
    SemanticSubtype,
    SpatialGenerator,
    StructuralConstraints,
    generate_dataset,
)


def _semantic_subtype_for_compatibility(
    subtype: str | None, question_type: int, target_num_answers: int | None
) -> SemanticSubtype:
    if subtype is not None:
        try:
            return SemanticSubtype(subtype)
        except ValueError as exc:
            raise ValueError(f"unknown v13 semantic subtype: {subtype}") from exc
    reverse = {
        (0, 1): SemanticSubtype.DIR_1,
        (0, 2): SemanticSubtype.DIR_2,
        (0, 0): SemanticSubtype.DIR_UNDETERMINED,
        (1, 0): SemanticSubtype.WHICH_0,
        (1, 1): SemanticSubtype.WHICH_1,
        (1, 2): SemanticSubtype.WHICH_2,
        (1, 3): SemanticSubtype.WHICH_3,
        (1, 4): SemanticSubtype.WHICH_4,
        (2, None): SemanticSubtype.COUNT_1,
        (2, 0): SemanticSubtype.COUNT_OMIT,
    }
    try:
        return reverse[(question_type, target_num_answers)]
    except KeyError as exc:
        raise ValueError(
            "question_type/target_num_answers does not identify a supported "
            "semantic subtype; pass subtype explicitly"
        ) from exc


def generate_sample(
    *,
    seed: int | None = None,
    relation_mode: str = "mixed",
    question_type: int = 0,
    target_num_answers: int | None = 1,
    num_entities: int = 8,
    num_sentences: int = 10,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_attempts: int = 500,
    subtype: str | None = None,
    require_independent_axes: bool = False,
) -> dict[str, Any] | None:
    """Compatibility adapter for the v13.0/v13.1 function interface."""
    try:
        spec = GenerationSpec(
            semantic_subtype=_semantic_subtype_for_compatibility(
                subtype, question_type, target_num_answers
            ),
            relation_mode=RelationMode(relation_mode),
            constraints=StructuralConstraints(
                require_independent_axes=require_independent_axes
            ),
            num_entities=num_entities,
            num_relations=num_sentences,
            max_attempts=max_attempts,
            system_prompt=system_prompt,
        )
    except ValueError as exc:
        if "is not a valid RelationMode" in str(exc):
            raise ValueError(f"unknown relation mode: {relation_mode}") from exc
        raise
    try:
        return SpatialGenerator().generate(spec, random.Random(seed)).to_row()
    except GenerationError:
        return None


def _build_cells(
    relation_modes: tuple[str, ...],
    subtype_counts: dict[str, int],
    independent_mixed_dir1: int,
) -> list[GenerationCell]:
    cells = [
        GenerationCell(
            name=f"{mode}-{subtype}",
            spec=GenerationSpec(
                semantic_subtype=SemanticSubtype(subtype),
                relation_mode=RelationMode(mode),
            ),
            count=count,
        )
        for mode in relation_modes
        for subtype in SEMANTIC_SUBTYPES
        if (count := subtype_counts.get(subtype, 0)) > 0
    ]
    if independent_mixed_dir1:
        cells.append(
            GenerationCell(
                name="mixed-dir-1-independent",
                spec=GenerationSpec(
                    semantic_subtype=SemanticSubtype.DIR_1,
                    relation_mode=RelationMode.MIXED,
                    constraints=StructuralConstraints(
                        require_independent_axes=True
                    ),
                ),
                count=independent_mixed_dir1,
            )
        )
    return cells


def batch_generate(
    output_file: str,
    *,
    relation_modes: tuple[str, ...] | None = None,
    subtype_counts: dict[str, int] | None = None,
    independent_mixed_dir1: int = 0,
    test_split: float,
    seed: int,
    relation_mode: str | None = None,
    num_type0_1_answer: int = 0,
    num_type0_2_answer: int = 0,
    num_type0_undetermined: int = 0,
    num_type1_1_answer: int = 0,
    num_type2: int = 0,
) -> tuple[Path, Path]:
    """Compatibility adapter that translates counters into explicit cells."""
    if relation_modes is None:
        relation_modes = (relation_mode or "mixed",)
    invalid_modes = set(relation_modes) - {mode.value for mode in RelationMode}
    if invalid_modes:
        raise ValueError(f"unknown relation modes: {sorted(invalid_modes)}")
    if subtype_counts is None:
        subtype_counts = {
            SemanticSubtype.DIR_1.value: num_type0_1_answer,
            SemanticSubtype.DIR_2.value: num_type0_2_answer,
            SemanticSubtype.DIR_UNDETERMINED.value: num_type0_undetermined,
            SemanticSubtype.WHICH_1.value: num_type1_1_answer,
            SemanticSubtype.COUNT_1.value: num_type2,
        }
    invalid_subtypes = set(subtype_counts) - set(SEMANTIC_SUBTYPES)
    if invalid_subtypes:
        raise ValueError(f"unknown semantic subtypes: {sorted(invalid_subtypes)}")
    if any(count < 0 for count in subtype_counts.values()):
        raise ValueError("subtype counts must be non-negative")
    if independent_mixed_dir1 < 0:
        raise ValueError("independent_mixed_dir1 must be non-negative")
    return generate_dataset(
        _build_cells(relation_modes, subtype_counts, independent_mixed_dir1),
        output_file=output_file,
        test_fraction=test_split,
        seed=seed,
    )


app = typer.Typer(help="v13 cardinal/diagonal synthetic SFT generator")


@app.command()
def main(
    out: str = typer.Option("../data/spatial_sft_v13_foundation.jsonl", "--out"),
    relation_modes: str = typer.Option(
        "diagonal,cardinal,mixed",
        "--relation-modes",
        help="Comma-separated relation modes to cross with all 13 subtypes.",
    ),
    subtypes: str = typer.Option(
        ",".join(SEMANTIC_SUBTYPES),
        "--subtypes",
        help="Comma-separated semantic subtypes to generate.",
    ),
    samples_per_cell: int = typer.Option(
        100, min=0, help="Rows for each relation-mode × semantic-subtype cell."
    ),
    independent_mixed_dir1: int = typer.Option(
        100,
        min=0,
        help="Extra mixed dir-1 rows whose shortest X/Y proofs are independent.",
    ),
    test_split: float = typer.Option(0.2, min=0.0, max=1.0),
    seed: int = typer.Option(13),
) -> None:
    modes = tuple(dict.fromkeys(
        mode.strip() for mode in relation_modes.split(",") if mode.strip()
    ))
    selected_subtypes = tuple(dict.fromkeys(
        subtype.strip() for subtype in subtypes.split(",") if subtype.strip()
    ))
    invalid_modes = set(modes) - {mode.value for mode in RelationMode}
    if invalid_modes:
        raise typer.BadParameter(
            f"unknown relation modes: {sorted(invalid_modes)}",
            param_hint="--relation-modes",
        )
    invalid_subtypes = set(selected_subtypes) - set(SEMANTIC_SUBTYPES)
    if invalid_subtypes:
        raise typer.BadParameter(
            f"unknown semantic subtypes: {sorted(invalid_subtypes)}",
            param_hint="--subtypes",
        )
    if not modes:
        raise typer.BadParameter(
            "select at least one relation mode", param_hint="--relation-modes"
        )
    if not selected_subtypes and independent_mixed_dir1 == 0:
        raise typer.BadParameter(
            "select at least one semantic subtype or an independent dir-1 count",
            param_hint="--subtypes",
        )
    train_path, test_path = batch_generate(
        out,
        relation_modes=modes,
        subtype_counts={subtype: samples_per_cell for subtype in selected_subtypes},
        independent_mixed_dir1=independent_mixed_dir1,
        test_split=test_split,
        seed=seed,
    )
    typer.echo(f"wrote {train_path} and {test_path}")


if __name__ == "__main__":
    app()
