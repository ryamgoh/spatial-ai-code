from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

MODULE = Path(__file__).with_name("make_rl_pool.py")
SPEC = importlib.util.spec_from_file_location("make_v14_rl_pool", MODULE)
assert SPEC and SPEC.loader
POOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POOL)


def test_rl_pool_recipe_has_expected_scale_and_question_breadth() -> None:
    train = POOL.build_cells(evaluation=False)
    evaluation = POOL.build_cells(evaluation=True)

    assert sum(cell.count for cell in train) == 680
    assert sum(cell.count for cell in evaluation) == 300
    assert {
        cell.spec.semantic_subtype.value
        for cell in train
        if cell.name.startswith("rl-train-large-")
    } >= {"which-2", "which-4", "count-1", "count-omit"}
    scales = Counter(
        cell.spec.cycle.length
        if cell.spec.cycle
        else max(
            cell.spec.constraints.x_depth.maximum
            if cell.spec.constraints.x_depth
            else 0,
            cell.spec.constraints.y_depth.maximum
            if cell.spec.constraints.y_depth
            else 0,
        )
        for cell in evaluation
        if not cell.name.startswith("rl-eval-large-")
    )
    assert set(scales) == {10, 12, 14}


def test_rl_row_contains_no_gold_trace() -> None:
    engine = POOL.V1.SpatialGenerator()
    cell = POOL.build_cells(evaluation=False)[0]
    example = engine.generate(cell.spec, POOL.random.Random(14001))
    source = example.to_row(generation_cell=cell.name)

    row = POOL.to_rl_row(source)

    assert {message["role"] for message in row["prompt"]} == {"system", "user"}
    assert all(message["role"] != "assistant" for message in row["prompt"])
    assert row["oracle_option"] == source["oracle_option"]
    assert POOL.ANSWER_LINE in row["prompt"][-1]["content"]
