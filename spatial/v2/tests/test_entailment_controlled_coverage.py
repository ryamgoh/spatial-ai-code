"""Coverage checks for the separately versioned controlled construction extension."""

from collections import defaultdict
from dataclasses import replace

import pytest

from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_controlled_coverage import (
    CAP,
    Slot,
    construct,
    size_target,
    slots,
)
from spatial.v2.entailment_controlled_prototype import check_semantics
from spatial.v2.entailment_dataset import (
    measure_depth,
    menus_for,
    selected_explanation_depth,
)
from spatial.v2.entailment_evidence import build_evidence
from spatial.v2.entailment_names import NAMES
from spatial.v2.solver import Direction


def test_manifest_is_bounded_and_crosses_all_count_depth_direction_cells():
    inventory = slots()
    assert len(inventory) == len({s.identifier for s in inventory}) == CAP
    assert {
        (s.span, s.direction, s.count_value)
        for s in inventory
        if s.kind == "count" and s.case in ("listed", "unlisted") and not s.compact
    } == {(d, r, c) for d in range(1, 11) for r in Direction for c in range(5)}


def test_all_constructed_sizes_scope_and_names():
    groups = defaultdict(set)
    for slot in slots():
        story, problem = construct(slot)
        sizes = len(problem.objects), len(problem.premise.operands)
        assert sizes == size_target(slot)
        assert set(problem.objects) <= set(NAMES)
        assert len(set(problem.premise.operands)) == sizes[1]
        assert story.source_family == slot.family
        groups[slot.size_group].add(sizes)
        if slot.compact:
            assert sizes in ((6, 6), (9, 9))
        if slot.kind != "direction":
            assert set(problem.query.candidates) == set(problem.objects) - {
                problem.query.reference
            }
    assert all(len(sizes) == 1 for sizes in groups.values())


@pytest.mark.parametrize(
    "slot",
    [
        s
        for s in slots()
        if s.compact
        and s.extra_size == 0
        or not s.compact
        and s.span in (2, 10)
        and (
            s.case in ("no-entailed", "inconsistent")
            or s.direction in (Direction.NORTH, Direction.NORTHEAST)
        )
    ],
    ids=lambda s: s.identifier,
)
def test_real_semantics_and_depth(slot):
    _, problem = construct(slot)
    solver = QuestionSolver()
    evidence = build_evidence(problem, solver=solver)
    check_semantics(slot, evidence.semantics)
    menus = menus_for(evidence.semantics, slot.seed, omission_percent=100)
    for menu in menus:
        assert (
            selected_explanation_depth(evidence, menu, Contract.UNIQUE)
            == slot.requested_depth
        )
    if slot.requested_depth is not None:
        assert measure_depth(problem, solver=solver)["depth"] == slot.requested_depth


def test_invalid_parameters_and_determinism():
    slot = Slot(1, Direction.NORTH, "direction", "listed", compact=True)
    assert construct(slot) == construct(slot)
    for invalid in (
        replace(slot, span=2),
        replace(slot, kind="count", case="multiple"),
        replace(slot, count_value=5),
        replace(slot, variant=9),
    ):
        with pytest.raises(ValueError):
            construct(invalid)
