"""First v13 synthetic SFT generator: cardinal + diagonal relations.

This is intentionally a narrow foundation.  It preserves the v6 question and
option laws, adds cardinal relations, and records structure measured by
``SpatialSolverV13`` after reparsing the final prompt.  Later v13 iterations
will add controlled proof-depth targets, distractors, and richer conflicts.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

import typer

from generate_all_v6 import AxisGraph, ENTITIES
from spatial_solver_v13 import SpatialSolverV13


SOLVER = SpatialSolverV13()
COMPOUNDS = ["Northeast", "Northwest", "Southeast", "Southwest"]
DIRECTIONS = ["North", "South", "East", "West", *COMPOUNDS]
SPECIAL = "Cannot be determined"
NONE = "None of the Options"
SEMANTIC_SUBTYPES = (
    "dir-1",
    "dir-2",
    "dir-undetermined",
    "dir-cycle",
    "dir-incomplete",
    "dir-omit",
    "which-1",
    "which-2",
    "which-3",
    "which-4",
    "which-0",
    "count-1",
    "count-omit",
)

DEFAULT_SYSTEM_PROMPT = (
    "You are an advanced spatial reasoning agent. Process the spatial "
    "relations step-by-step. A cardinal relation updates only its named axis; "
    "a diagonal relation updates both axes. Build an X-axis order from West "
    "to East and a Y-axis order from South to North. Use A < B to mean A is "
    "West/South of B on that axis. A two-axis direction is proven only when "
    "both components are derived.\n\n"
    "Three Type-0 letter rules:\n"
    "1. If exactly one axis is derived and both matching compounds are listed, "
    "return both letters.\n"
    "2. If exactly one axis is derived and only one matching compound is "
    "listed, select Cannot be determined.\n"
    "3. If both orders hold for a pair on an axis, treat that axis as "
    "contradictory and unknown.\n"
    "If neither axis is derived, select Cannot be determined. If a proven "
    "answer is absent, select None of the Options. End with Answer: followed "
    "by the exact letter set."
)


def _components(direction: str) -> tuple[str | None, str | None]:
    x = "East" if "east" in direction.lower() else "West" if "west" in direction.lower() else None
    y = "North" if "north" in direction.lower() else "South" if "south" in direction.lower() else None
    return x, y


def _relation_for_pair(
    a: str, b: str, coords: dict[str, tuple[int, int]], kind: str, rng: random.Random
) -> dict[str, Any]:
    ax, ay = coords[a]
    bx, by = coords[b]
    east_west = "East" if ax > bx else "West"
    north_south = "North" if ay > by else "South"
    if kind == "diagonal":
        direction = f"{north_south}{east_west.lower()}"
    elif rng.choice(("x", "y")) == "x":
        direction = east_west
    else:
        direction = north_south
    x_comp, y_comp = _components(direction)
    return {
        "a": a,
        "b": b,
        "direction": direction,
        "x": x_comp,
        "y": y_comp,
        "text": f"The {a} is to the {direction} of the {b}.",
    }


def _explicit_relation(a: str, direction: str, b: str) -> dict[str, Any]:
    x_comp, y_comp = _components(direction)
    return {
        "a": a,
        "b": b,
        "direction": direction,
        "x": x_comp,
        "y": y_comp,
        "text": f"The {a} is to the {direction} of the {b}.",
    }


def _make_scene(
    rng: random.Random, num_entities: int, num_sentences: int, relation_mode: str
) -> tuple[list[str], list[dict[str, Any]]]:
    if relation_mode not in {"diagonal", "cardinal", "mixed"}:
        raise ValueError("relation_mode must be diagonal, cardinal, or mixed")
    if not 2 <= num_entities <= len(ENTITIES):
        raise ValueError(f"num_entities must be in [2, {len(ENTITIES)}]")
    entities = rng.sample(ENTITIES, num_entities)
    coords: dict[str, tuple[int, int]] = {}
    for entity in entities:
        while True:
            candidate = (rng.randint(0, 1000), rng.randint(0, 1000))
            if all(candidate[0] != x and candidate[1] != y for x, y in coords.values()):
                coords[entity] = candidate
                break
    pairs = list(itertools.combinations(entities, 2))
    rng.shuffle(pairs)
    pairs = pairs[: min(num_sentences, len(pairs))]
    relations: list[dict[str, Any]] = []
    for index, (a, b) in enumerate(pairs):
        if relation_mode == "mixed":
            # Guarantee both families whenever at least two statements exist.
            kind = "cardinal" if index == 0 else "diagonal" if index == 1 else rng.choice(("cardinal", "diagonal"))
        else:
            kind = relation_mode
        relations.append(_relation_for_pair(a, b, coords, kind, rng))
    return entities, relations


def _graphs(entities: list[str], relations: list[dict[str, Any]]) -> tuple[AxisGraph, AxisGraph]:
    x_graph, y_graph = AxisGraph(), AxisGraph()
    x_graph.nodes.update(entities)
    y_graph.nodes.update(entities)
    for relation in relations:
        a, b = relation["a"], relation["b"]
        if relation["x"] == "East":
            x_graph.add_relation(b, a)
        elif relation["x"] == "West":
            x_graph.add_relation(a, b)
        if relation["y"] == "North":
            y_graph.add_relation(b, a)
        elif relation["y"] == "South":
            y_graph.add_relation(a, b)
    return x_graph, y_graph


def _status(closure: set[tuple[str, str]], target: str, reference: str) -> str | None:
    target_low = (target, reference) in closure
    target_high = (reference, target) in closure
    if target_low == target_high:
        return None
    return "low" if target_low else "high"


def _options_text(options: list[str]) -> str:
    return ", ".join(f"{letter}. {value}" for letter, value in zip("ABCDE", options, strict=True))


def _direction_prompt(
    entities: list[str],
    relations: list[dict[str, Any]],
    x_graph: AxisGraph,
    y_graph: AxisGraph,
    target_num_answers: int | None,
    rng: random.Random,
    subtype: str | None = None,
) -> str | None:
    x_closure, y_closure = x_graph.get_transitive_closure(), y_graph.get_transitive_closure()
    candidates: list[tuple[str, str]] = []
    for target, reference in itertools.permutations(entities, 2):
        x_known = _status(x_closure, target, reference) is not None
        y_known = _status(y_closure, target, reference) is not None
        answer_count = 1 if x_known and y_known else 2 if x_known or y_known else 0
        if target_num_answers is None or answer_count == target_num_answers:
            candidates.append((target, reference))
    if not candidates:
        return None
    target, reference = rng.choice(candidates)
    x_state = _status(x_closure, target, reference)
    y_state = _status(y_closure, target, reference)
    x_comp = "East" if x_state == "high" else "West" if x_state == "low" else None
    y_comp = "North" if y_state == "high" else "South" if y_state == "low" else None
    xs = [x_comp] if x_comp else ["East", "West"]
    ys = [y_comp] if y_comp else ["North", "South"]
    possible = [f"{y}{x.lower()}" for y in ys for x in xs]
    if subtype == "dir-incomplete":
        if len(possible) != 2:
            return None
        kept = rng.choice(possible)
        pool = [direction for direction in DIRECTIONS if direction not in possible]
        options = [kept, *rng.sample(pool, 3), SPECIAL]
    elif subtype == "dir-omit":
        pool = [direction for direction in DIRECTIONS if direction not in possible]
        if len(pool) < 4:
            return None
        options = [*rng.sample(pool, 4), NONE]
    else:
        options = [*COMPOUNDS, SPECIAL]
    rng.shuffle(options)
    return (
        "Consider a map with multiple locations:\n\n"
        + " ".join(relation["text"] for relation in relations)
        + f"\n\nQuestion: In which direction is the {target} relative to the {reference}? "
        + f"Available options: {_options_text(options)}"
    )


def _entities_in_direction(
    entities: list[str],
    reference: str,
    direction: str,
    x_closure: set[tuple[str, str]],
    y_closure: set[tuple[str, str]],
) -> list[str]:
    need_x, need_y = _components(direction)
    found: list[str] = []
    for entity in entities:
        if entity == reference:
            continue
        x_status = _status(x_closure, entity, reference)
        y_status = _status(y_closure, entity, reference)
        x_ok = need_x is None or x_status == ("high" if need_x == "East" else "low")
        y_ok = need_y is None or y_status == ("high" if need_y == "North" else "low")
        if x_ok and y_ok:
            found.append(entity)
    return sorted(found)


def _which_prompt(
    entities: list[str],
    relations: list[dict[str, Any]],
    x_graph: AxisGraph,
    y_graph: AxisGraph,
    target_num_answers: int | None,
    rng: random.Random,
) -> str | None:
    x_closure, y_closure = x_graph.get_transitive_closure(), y_graph.get_transitive_closure()
    desired = 1 if target_num_answers is None else target_num_answers
    candidates = []
    for reference in entities:
        for direction in DIRECTIONS:
            correct = _entities_in_direction(entities, reference, direction, x_closure, y_closure)
            incorrect = [entity for entity in entities if entity != reference and entity not in correct]
            if desired == 0 and len(incorrect) >= 4:
                candidates.append((reference, direction, correct, incorrect))
            elif desired > 0 and len(correct) >= desired and len(incorrect) >= 4 - desired:
                candidates.append((reference, direction, correct, incorrect))
    if not candidates:
        return None
    reference, direction, correct, incorrect = rng.choice(candidates)
    if desired == 0:
        choices = rng.sample(incorrect, 4)
    else:
        choices = rng.sample(correct, desired) + rng.sample(incorrect, 4 - desired)
        rng.shuffle(choices)
    options = choices + [NONE]
    return (
        "Consider a map with multiple locations:\n\n"
        + " ".join(relation["text"] for relation in relations)
        + f"\n\nQuestion: Which object is in the {direction} of the {reference}? "
        + f"Available options: {_options_text(options)}"
    )


def _count_prompt(
    entities: list[str],
    relations: list[dict[str, Any]],
    x_graph: AxisGraph,
    y_graph: AxisGraph,
    target_num_answers: int | None,
    rng: random.Random,
) -> str:
    x_closure, y_closure = x_graph.get_transitive_closure(), y_graph.get_transitive_closure()
    reference = rng.choice(entities)
    direction = rng.choice(DIRECTIONS)
    count = len(_entities_in_direction(entities, reference, direction, x_closure, y_closure))
    omit = target_num_answers == 0
    values: list[int] = [] if omit else [count]
    wrong = [value for value in range(len(entities) + 3) if value != count]
    rng.shuffle(wrong)
    values.extend(wrong[: 4 - len(values)])
    rng.shuffle(values)
    options = [str(value) for value in values] + [NONE]
    return (
        "Consider a map with multiple locations:\n\n"
        + " ".join(relation["text"] for relation in relations)
        + f"\n\nQuestion: How many objects are in the {direction} of the {reference}? "
        + f"Available options: {_options_text(options)}"
    )


def _trace(entities: list[str], relations: list[dict[str, Any]], user_prompt: str) -> str:
    grade = SOLVER.grade(user_prompt)
    x_graph, y_graph = AxisGraph(), AxisGraph()
    x_graph.nodes.update(entities)
    y_graph.nodes.update(entities)
    active: set[str] = set()
    chunks = [
        "### Initialization\n"
        f"**Entities Detected**: {', '.join(sorted(entities))}\n"
        "**Initial X-State**: Empty\n"
        "**Initial Y-State**: Empty"
    ]
    for index, relation in enumerate(relations, 1):
        a, b = relation["a"], relation["b"]
        active.update((a, b))
        extraction: list[str] = []
        if relation["x"] == "East":
            x_graph.add_relation(b, a)
            extraction.append(f"**X-Extraction**: {b} < {a}")
        elif relation["x"] == "West":
            x_graph.add_relation(a, b)
            extraction.append(f"**X-Extraction**: {a} < {b}")
        else:
            extraction.append("**X-Extraction**: none (cardinal Y-only relation)")
        if relation["y"] == "North":
            y_graph.add_relation(b, a)
            extraction.append(f"**Y-Extraction**: {b} < {a}")
        elif relation["y"] == "South":
            y_graph.add_relation(a, b)
            extraction.append(f"**Y-Extraction**: {a} < {b}")
        else:
            extraction.append("**Y-Extraction**: none (cardinal X-only relation)")
        chunks.append(
            f"### Step {index}\n**Sentence**: \"{relation['text']}\"\n"
            + "\n".join(extraction)
            + f"\n**X-State**: {x_graph.format_state(active)}"
            + f"\n**Y-State**: {y_graph.format_state(active)}"
        )
    verdicts = ["### Final Deduction"]
    for letter, value, accepted, reason in grade.verdicts:
        verdicts.append(f"- {letter}. {value} — {reason}. {'In' if accepted else 'Out'}.")
    return "<think>\n" + "\n\n".join(chunks + ["\n".join(verdicts)]) + f"\n</think>\nAnswer: {grade.pretty}"


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
    """Generate one solver-verified sample at the public v13 seam."""
    if subtype is not None and subtype not in SEMANTIC_SUBTYPES:
        raise ValueError(f"unknown v13 semantic subtype: {subtype}")
    if subtype is not None:
        if subtype == "dir-1":
            question_type, target_num_answers = 0, 1
        elif subtype == "dir-2":
            question_type, target_num_answers = 0, 2
        elif subtype in {"dir-undetermined", "dir-cycle"}:
            question_type, target_num_answers = 0, 0
        elif subtype in {"dir-incomplete", "dir-omit"}:
            question_type, target_num_answers = 0, 2 if subtype == "dir-incomplete" else 1
        elif subtype.startswith("which-"):
            question_type = 1
            target_num_answers = int(subtype.split("-", 1)[1])
        elif subtype == "count-1":
            question_type, target_num_answers = 2, None
        else:
            question_type, target_num_answers = 2, 0
    if require_independent_axes and not (
        question_type == 0 and target_num_answers == 1
    ):
        raise ValueError(
            "require_independent_axes currently requires a dir-1 question"
        )
    rng = random.Random(seed)
    for _ in range(max_attempts):
        entities, relations = _make_scene(rng, num_entities, num_sentences, relation_mode)
        if subtype == "dir-cycle":
            target, reference = entities[0], entities[1]
            if relation_mode == "cardinal":
                relations.extend(
                    [
                        _explicit_relation(target, "East", reference),
                        _explicit_relation(reference, "East", target),
                        _explicit_relation(target, "North", reference),
                        _explicit_relation(reference, "North", target),
                    ]
                )
            else:
                relations.extend(
                    [
                        _explicit_relation(target, "Northeast", reference),
                        _explicit_relation(reference, "Northeast", target),
                    ]
                )
        x_graph, y_graph = _graphs(entities, relations)
        if question_type == 0:
            if subtype == "dir-cycle":
                options = [*COMPOUNDS, SPECIAL]
                rng.shuffle(options)
                user_prompt = (
                    "Consider a map with multiple locations:\n\n"
                    + " ".join(relation["text"] for relation in relations)
                    + f"\n\nQuestion: In which direction is the {target} relative to the {reference}? "
                    + f"Available options: {_options_text(options)}"
                )
            else:
                user_prompt = _direction_prompt(
                    entities, relations, x_graph, y_graph, target_num_answers, rng, subtype
                )
        elif question_type == 1:
            user_prompt = _which_prompt(
                entities, relations, x_graph, y_graph, target_num_answers, rng
            )
        elif question_type == 2:
            user_prompt = _count_prompt(
                entities, relations, x_graph, y_graph, target_num_answers, rng
            )
        else:
            raise ValueError("question_type must be 0, 1, or 2")
        if not user_prompt:
            continue
        grade = SOLVER.grade(user_prompt)
        if not grade.accept or len(grade.options) != 5:
            continue
        difficulty = SOLVER.analyze(user_prompt)
        if subtype is not None and difficulty["semantic_subtype"] != subtype:
            continue
        if require_independent_axes and difficulty["axes_independent"] is not True:
            continue
        if question_type == 0 and subtype not in {"dir-incomplete", "dir-omit", "dir-cycle"}:
            expected = target_num_answers
            is_special = len(grade.letters) == 1 and any(
                accepted and SOLVER.is_undetermined_option(value)
                for _, value, accepted, _ in grade.verdicts
            )
            if expected == 0 and not is_special:
                continue
            if expected in {1, 2} and (len(grade.letters) != expected or is_special):
                continue
        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": _trace(entities, relations, user_prompt)},
            ],
            "oracle_option": grade.raw,
            "difficulty": difficulty,
            "generator_version": "v13.1-semantic-grid",
        }
        if sample["difficulty"]["relation_mix"] == (
            {"diagonal": "diagonal-only", "cardinal": "cardinal-only", "mixed": "mixed"}[relation_mode]
        ):
            return sample
    return None


def batch_generate(
    output_file: str,
    *,
    relation_modes: tuple[str, ...] | None = None,
    subtype_counts: dict[str, int] | None = None,
    independent_mixed_dir1: int = 0,
    test_split: float,
    seed: int,
    # Backward-compatible v13.0 foundation interface.
    relation_mode: str | None = None,
    num_type0_1_answer: int = 0,
    num_type0_2_answer: int = 0,
    num_type0_undetermined: int = 0,
    num_type1_1_answer: int = 0,
    num_type2: int = 0,
) -> tuple[Path, Path]:
    if not 0.0 <= test_split <= 1.0:
        raise ValueError("test_split must be between 0 and 1")
    if relation_modes is None:
        relation_modes = (relation_mode or "mixed",)
    invalid_modes = set(relation_modes) - {"diagonal", "cardinal", "mixed"}
    if invalid_modes:
        raise ValueError(f"unknown relation modes: {sorted(invalid_modes)}")
    if subtype_counts is None:
        subtype_counts = {
            "dir-1": num_type0_1_answer,
            "dir-2": num_type0_2_answer,
            "dir-undetermined": num_type0_undetermined,
            "which-1": num_type1_1_answer,
            "count-1": num_type2,
        }
    invalid_subtypes = set(subtype_counts) - set(SEMANTIC_SUBTYPES)
    if invalid_subtypes:
        raise ValueError(f"unknown semantic subtypes: {sorted(invalid_subtypes)}")
    if any(count < 0 for count in subtype_counts.values()):
        raise ValueError("subtype counts must be non-negative")
    if independent_mixed_dir1 < 0:
        raise ValueError("independent_mixed_dir1 must be non-negative")

    seed_rng = random.Random(seed)
    cells: list[list[dict[str, Any]]] = []
    for mode in relation_modes:
        for subtype in SEMANTIC_SUBTYPES:
            cell: list[dict[str, Any]] = []
            for _ in range(subtype_counts.get(subtype, 0)):
                sample = generate_sample(
                    seed=seed_rng.randrange(2**63),
                    relation_mode=mode,
                    subtype=subtype,
                )
                if sample is None:
                    raise RuntimeError(
                        f"could not generate mode={mode} subtype={subtype}"
                    )
                sample["generation_cell"] = f"{mode}-{subtype}"
                cell.append(sample)
            if cell:
                cells.append(cell)

    if independent_mixed_dir1:
        cell = []
        for _ in range(independent_mixed_dir1):
            sample = generate_sample(
                seed=seed_rng.randrange(2**63),
                relation_mode="mixed",
                subtype="dir-1",
                require_independent_axes=True,
            )
            if sample is None:
                raise RuntimeError("could not generate mixed independent dir-1")
            sample["generation_cell"] = "mixed-dir-1-independent"
            cell.append(sample)
        cells.append(cell)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for cell in cells:
        seed_rng.shuffle(cell)
        test_count = int(len(cell) * test_split)
        split_at = len(cell) - test_count
        train_rows.extend(cell[:split_at])
        test_rows.extend(cell[split_at:])
    seed_rng.shuffle(train_rows)
    seed_rng.shuffle(test_rows)
    base = Path(output_file)
    train_path = base.with_name(base.stem + "_train.jsonl")
    test_path = base.with_name(base.stem + "_test.jsonl")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, subset in ((train_path, train_rows), (test_path, test_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_path, test_path


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
    invalid_modes = set(modes) - {"diagonal", "cardinal", "mixed"}
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
        raise typer.BadParameter("select at least one relation mode", param_hint="--relation-modes")
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
