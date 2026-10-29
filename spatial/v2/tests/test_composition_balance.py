"""Bounded quota diagnostics preserve paired contracts and feasibility limits."""

from collections import Counter
from dataclasses import replace

import pytest

from spatial.v2.composition_balance import (
    KINDS,
    cell_key,
    diagnostic_selection,
    interface_matrix,
    menu_inventory,
    query_profile,
    row_metadata,
    schedule,
    topology_key,
)
from spatial.v2.composition_contracts import analyze
from spatial.v2.composition_pilot import renamed_problem
from spatial.v2.grading import AnswerMode
from spatial.v2.solver import And, Direction, direction_constraint
from spatial.v2.tests.test_pivot_evidence import BY_ID


def test_schedule_is_fixed_and_not_retry_to_quota():
    a = schedule()
    assert a == schedule()
    assert len(a) == 80
    assert len({s.story_id for s, _ in a}) == 80
    assert Counter(d for _, d in a if d is not None) == {d: 3 for d in range(1, 11)}
    counts = Counter(s.recipe for s, _ in a)
    assert counts["random-world"] == 24
    assert counts["random-atoms"] == 16
    assert counts["controlled-depth"] == 30


@pytest.mark.parametrize("name", tuple(BY_ID))
def test_menu_inventory_covers_every_feasible_intersection(name):
    semantics = analyze(BY_ID[name].problem())
    menus = menu_inventory(semantics, 42)
    from spatial.v2.composition_contracts import answer_domain

    n, p = len(answer_domain(semantics.problem)), len(semantics.possible)
    assert [len(set(m) & semantics.possible) for m in menus] == list(
        range(max(0, 4 - (n - p)), min(4, p) + 1)
    )
    assert menus == menu_inventory(semantics, 42)
    assert all(len(m) == len(set(m)) == 4 for m in menus)
    matrix = interface_matrix()
    kind = type(semantics.problem.query).__name__.removesuffix("Query").lower()
    for menu in menus:
        for mode in AnswerMode:
            assert (
                cell_key({"query_kind": kind, **row_metadata(semantics, menu, mode)})
                in matrix
            )


def test_matrix_does_not_invent_multiletter_unique_or_ambiguous_unique_substance():
    cells = list(interface_matrix().values())
    for c in cells:
        if c["contract"] == "single":
            assert c["returned_count"] in (None, 1)
            if c["semantic_case"] != "unique":
                assert c["literal_gold_potential"] == ["E"]
        if c["semantic_case"] == "inconsistent":
            assert c["literal_gold_potential"] == ["E"]
        if c["query_kind"] == "direction" and c["semantic_case"] == "unique":
            assert c["returned_count"] in (None, 1)
    assert any(c["returned_count"] == 4 for c in cells)


def test_structure_ignores_renaming_premise_order_and_inverse_orientation():
    problem = BY_ID["D1"].problem()
    renamed, _ = renamed_problem(problem)
    assert query_profile(problem) == query_profile(renamed)
    inverse = replace(
        problem,
        premise=And(
            (
                direction_constraint("Y", "X", Direction.SOUTHWEST),
                direction_constraint("Y", "Z", Direction.NORTH),
            )
        ),
    )
    assert query_profile(problem) == query_profile(inverse)


def test_topology_does_not_reward_disconnected_distractors():
    problem = BY_ID["D1"].problem()
    extra = replace(
        problem,
        objects=(*problem.objects, "ExtraA", "ExtraB"),
        premise=And(
            (
                *problem.premise.operands,
                direction_constraint("ExtraA", "ExtraB", Direction.NORTH),
            )
        ),
    )
    assert topology_key(extra) == topology_key(problem)


def test_diagnostic_balances_bundles_not_independent_contract_rows():
    records = []
    for kind in KINDS:
        for i in range(3):
            base = f"{kind}-{i}"
            rows = [
                {
                    "bundle_id": f"{base}-m{m}",
                    "query_kind": kind,
                    "contract": mode.value,
                    "answer": "F" if m else "A",
                    "coverage": "partial" if m else "full",
                    "substantive_depth_support": True,
                    "depth": 1,
                }
                for m in range(2)
                for mode in AnswerMode
            ]
            records.append(
                {
                    "base_id": base,
                    "query_kind": kind,
                    "status": "accepted",
                    "structure": {"core_topology": str(i % 2)},
                    "rows": rows,
                }
            )
    selected = diagnostic_selection(records)
    assert selected["bundles_per_question_type"] == 2
    assert selected["views"] == 18
    assert set(selected["groups"].values()) == {2}
    assert len(selected["base_ids"]) == len(set(selected["base_ids"])) == 6
    assert all(id_.endswith("-m0") for id_ in selected["bundle_ids"])
