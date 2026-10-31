"""The new prototype must not silently change the existing study policy."""

from collections import defaultdict

import pytest

from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_controlled_prototype import (
    Slot,
    check_semantics,
    construct,
    size_target,
    slots,
)
from spatial.v2.entailment_dataset import (
    measure_depth,
    menus_for,
    selected_explanation_depth,
)
from spatial.v2.entailment_evidence import build_evidence
from spatial.v2.entailment_names import NAMES


def test_fixed_cap_size_matching_and_ordinary_names():
    inventory = slots()
    assert len(inventory) == len({s.identifier for s in inventory}) == 260
    groups = defaultdict(set)
    for slot in inventory:
        story, problem = construct(slot)
        actual = (len(problem.objects), len(problem.premise.operands))
        assert actual == size_target(slot)
        assert set(problem.objects) <= set(NAMES)
        assert len(set(problem.premise.operands)) == len(problem.premise.operands)
        assert story.source_family == slot.family
        groups[(slot.span, slot.extra_size)].add(actual)
        if slot.kind != "direction":
            assert set(problem.query.candidates) == set(problem.objects) - {
                problem.query.reference
            }
    assert len(groups) == 20
    assert all(len(sizes) == 1 for sizes in groups.values())


@pytest.mark.parametrize(
    "slot",
    [s for s in slots() if s.span in (1, 2, 8, 10) and s.extra_size == 0],
    ids=lambda s: s.identifier,
)
def test_intended_cases_and_real_positive_depth(slot):
    _, problem = construct(slot)
    solver = QuestionSolver()
    evidence = build_evidence(problem, solver=solver)
    check_semantics(slot, evidence.semantics)
    menu = menus_for(evidence.semantics, slot.seed, omission_percent=100)[
        -1 if slot.case == "unlisted" else 0
    ]
    actual = selected_explanation_depth(evidence, menu, Contract.UNIQUE)
    assert actual == slot.requested_depth
    if slot.requested_depth is not None:
        assert measure_depth(problem, solver=solver)["depth"] == slot.requested_depth


def test_deterministic_and_outside_manifest_rejected():
    slot = slots()[0]
    assert construct(slot) == construct(slot)
    with pytest.raises(ValueError, match="manifest"):
        construct(Slot(11, 0, "direction", "listed"))
    with pytest.raises(ValueError, match="manifest"):
        construct(Slot(2, 0, "count", "multiple"))
