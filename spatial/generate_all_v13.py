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
) -> dict[str, Any] | None:
    """Generate one solver-verified sample at the public v13 seam."""
    rng = random.Random(seed)
    for _ in range(max_attempts):
        entities, relations = _make_scene(rng, num_entities, num_sentences, relation_mode)
        x_graph, y_graph = _graphs(entities, relations)
        if question_type == 0:
            user_prompt = _direction_prompt(
                entities, relations, x_graph, y_graph, target_num_answers, rng
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
        if question_type == 0:
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
            "difficulty": SOLVER.analyze(user_prompt),
            "generator_version": "v13.0-cardinal-foundation",
        }
        if sample["difficulty"]["relation_mix"] == (
            {"diagonal": "diagonal-only", "cardinal": "cardinal-only", "mixed": "mixed"}[relation_mode]
        ):
            return sample
    return None


def batch_generate(
    output_file: str,
    *,
    relation_mode: str,
    num_type0_1_answer: int,
    num_type0_2_answer: int,
    num_type0_undetermined: int,
    num_type1_1_answer: int,
    num_type2: int,
    test_split: float,
    seed: int,
) -> tuple[Path, Path]:
    plan = [
        (0, 1, num_type0_1_answer),
        (0, 2, num_type0_2_answer),
        (0, 0, num_type0_undetermined),
        (1, 1, num_type1_1_answer),
        (2, None, num_type2),
    ]
    seed_rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for question_type, target_num_answers, count in plan:
        for _ in range(count):
            sample = generate_sample(
                seed=seed_rng.randrange(2**63),
                relation_mode=relation_mode,
                question_type=question_type,
                target_num_answers=target_num_answers,
            )
            if sample is None:
                raise RuntimeError(
                    f"could not generate qtype={question_type} target={target_num_answers}"
                )
            rows.append(sample)
    seed_rng.shuffle(rows)
    split_at = len(rows) - int(len(rows) * test_split)
    base = Path(output_file)
    train_path = base.with_name(base.stem + "_train.jsonl")
    test_path = base.with_name(base.stem + "_test.jsonl")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, subset in ((train_path, rows[:split_at]), (test_path, rows[split_at:])):
        with path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_path, test_path


app = typer.Typer(help="v13 cardinal/diagonal synthetic SFT generator")


@app.command()
def main(
    out: str = typer.Option("../data/spatial_sft_v13_foundation.jsonl", "--out"),
    relation_mode: str = typer.Option("mixed", "--relation-mode"),
    num_type0_1_answer: int = typer.Option(100, min=0),
    num_type0_2_answer: int = typer.Option(100, min=0),
    num_type0_undetermined: int = typer.Option(100, min=0),
    num_type1_1_answer: int = typer.Option(100, min=0),
    num_type2: int = typer.Option(100, min=0),
    test_split: float = typer.Option(0.2, min=0.0, max=1.0),
    seed: int = typer.Option(13),
) -> None:
    train_path, test_path = batch_generate(
        out,
        relation_mode=relation_mode,
        num_type0_1_answer=num_type0_1_answer,
        num_type0_2_answer=num_type0_2_answer,
        num_type0_undetermined=num_type0_undetermined,
        num_type1_1_answer=num_type1_1_answer,
        num_type2=num_type2,
        test_split=test_split,
        seed=seed,
    )
    typer.echo(f"wrote {train_path} and {test_path}")


if __name__ == "__main__":
    app()
