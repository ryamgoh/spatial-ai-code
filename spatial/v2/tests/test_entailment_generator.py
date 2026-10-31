"""Public-interface regression checks against the preserved coverage recipes."""

from dataclasses import replace

import pytest

from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_controlled_coverage import construct, slots
from spatial.v2.entailment_controlled_prototype import check_semantics
from spatial.v2.entailment_dataset import (
    measure_depth,
    menus_for,
    selected_explanation_depth,
)
from spatial.v2.entailment_evidence import build_evidence
from spatial.v2.entailment_generator import SceneRequest, build_scene
from spatial.v2.solver import Direction


def request_for(slot):
    return SceneRequest(
        span=slot.span,
        direction=slot.direction,
        kind=slot.kind,
        case="sole" if slot.case in ("listed", "unlisted") else slot.case,
        count_value=slot.count_value,
        variant=slot.variant,
        size=("compact-extra" if slot.extra_size else "compact")
        if slot.compact
        else "contextual",
    )


@pytest.mark.parametrize("slot", slots(), ids=lambda s: s.identifier)
def test_exact_equivalence_to_every_fixed_coverage_scene(slot):
    old_story, old_problem = construct(slot)
    story, problem = build_scene(
        request_for(slot), identity=slot.identifier, family=slot.family
    )
    assert problem == old_problem
    # Only provenance changes, not names, premise order, edges or query scope.
    assert replace(story, recipe=old_story.recipe) == old_story


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
def test_public_interface_semantics_and_actual_depth(slot):
    request = request_for(slot)
    _, problem = build_scene(request, identity=slot.identifier, family=slot.family)
    solver = QuestionSolver()
    evidence = build_evidence(problem, solver=solver)
    check_semantics(slot, evidence.semantics)
    for menu in menus_for(evidence.semantics, slot.seed, omission_percent=100):
        assert (
            selected_explanation_depth(evidence, menu, Contract.UNIQUE)
            == request.requested_depth
        )
    if request.requested_depth is not None:
        assert measure_depth(problem, solver=solver)["depth"] == request.requested_depth


@pytest.mark.parametrize(
    "changes",
    [
        {"span": True},
        {"span": 0},
        {"span": 11},
        {"direction": "North"},
        {"count_value": True},
        {"count_value": 5},
        {"variant": 3},
        {"case": "listed"},
        {"case": "unlisted"},
        {"case": "multiple"},
        {"kind": "other"},
        {"size": "other"},
        {"span": 2, "size": "compact"},
    ],
)
def test_invalid_requests_fail_before_construction(changes):
    with pytest.raises(ValueError):
        SceneRequest(
            **{
                "span": 1,
                "direction": Direction.NORTH,
                "kind": "direction",
                "case": "sole",
                **changes,
            }
        )


def test_no_calls_to_historical_construction_helpers(monkeypatch):
    import spatial.v2.composition_scene_sampler as sampler
    import spatial.v2.entailment_controlled_coverage as coverage
    import spatial.v2.entailment_controlled_prototype as prototype

    request = SceneRequest(2, Direction.NORTH, "direction", "sole")
    expected = build_scene(request, identity="interface-fixture", family="fixture")

    def fail(*_args, **_kwargs):
        pytest.fail("standalone generator called a historical construction helper")

    monkeypatch.setattr(sampler, "_support", fail)
    monkeypatch.setattr(prototype, "_core", fail)
    monkeypatch.setattr(coverage, "original_core", fail)
    monkeypatch.setattr(coverage, "construct", fail)
    assert (
        build_scene(request, identity="interface-fixture", family="fixture") == expected
    )


def test_identity_validation_and_no_implicit_menu_or_label():
    request = SceneRequest(1, Direction.NORTH, "direction", "sole")
    with pytest.raises(ValueError):
        build_scene(request, identity="", family="fixture")
    with pytest.raises(ValueError):
        build_scene(request, identity="fixture", family=" ")
    with pytest.raises(TypeError):
        build_scene(None, identity="fixture", family="fixture")
    assert request.requested_depth == 1
    assert replace(request, case="no-entailed").requested_depth is None
    assert replace(request, case="inconsistent").requested_depth is None
