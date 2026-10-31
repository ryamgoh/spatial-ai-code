"""Spatial wording must preserve all evidence obligations in both styles."""

from copy import deepcopy

import pytest

from spatial.v2.composition_contracts import VerificationError
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_dataset import menus_for, selected_explanation_depth
from spatial.v2.entailment_evidence import (
    SPATIAL_PROSE_VERSION,
    build_evidence,
    check_pair,
    reference_options,
    render_pair,
)
from spatial.v2.entailment_generator import SceneRequest, build_scene
from spatial.v2.entailment_readable_v3 import audit_plan, contains_path, make_plan
from spatial.v2.solver import Direction
from spatial.v2.tests.test_pivot_evidence import BY_ID


@pytest.mark.parametrize("name", sorted(BY_ID))
def test_matched_spatial_rendering_preserves_evidence_and_answers(name):
    solver = QuestionSolver()
    evidence = build_evidence(BY_ID[name].problem(), solver=solver)
    before = deepcopy(evidence)
    for menu in menus_for(evidence.semantics, 0, omission_percent=100):
        old = render_pair(
            evidence, menu, Contract.UNIQUE, solver=solver, restate_premises=True
        )
        pair = render_pair(
            evidence,
            menu,
            Contract.UNIQUE,
            solver=solver,
            restate_premises=True,
            spatial_prose=True,
        )
        check_pair(
            evidence,
            menu,
            Contract.UNIQUE,
            pair,
            solver=solver,
            restate_premises=True,
            spatial_prose=True,
        )
        assert evidence == before
        for style, text in pair.items():
            assert text.rsplit("Answer: ", 1)[1] == old[style].rsplit("Answer: ", 1)[1]
            assert "The claim that" not in text
            assert "mutually exclusive" not in text
            assert "sole entailed" not in text
        assert "x-coordinate" not in pair["natural"]
        assert "y-coordinate" not in pair["natural"]


@pytest.mark.parametrize("name", ("D2", "C1"))
def test_exact_ambiguity_uses_two_distinct_existing_outcomes(name):
    evidence = build_evidence(BY_ID[name].problem())
    plan = make_plan(evidence)
    assert len(plan["witness_keys"]) == len(set(plan["outcomes"])) == 2
    audit_plan(evidence, plan)
    bad = deepcopy(plan)
    bad["outcomes"][1] = bad["outcomes"][0]
    with pytest.raises(VerificationError):
        audit_plan(evidence, bad)
    bad = deepcopy(plan)
    bad["exact_ambiguity"] = False
    with pytest.raises(VerificationError):
        audit_plan(evidence, bad)


def test_which_groups_cover_all_other_objects_including_unlisted():
    evidence = build_evidence(BY_ID["S6"].problem())
    plan = make_plan(evidence)
    assert {v for group in plan["negative_groups"].values() for v in group} == (
        set(evidence.semantics.problem.query.candidates) - evidence.semantics.entailed
    )
    pair = render_pair(
        evidence,
        ("U", "V", "W", "T"),
        Contract.UNIQUE,
        restate_premises=True,
        spatial_prose=True,
    )
    assert all(text.endswith("Answer: F") for text in pair.values())
    for candidate in ("U", "V", "W", "T"):
        assert candidate in pair["natural"]
    bad = deepcopy(plan)
    next(iter(bad["negative_groups"].values())).pop()
    with pytest.raises(VerificationError, match="alternative"):
        audit_plan(evidence, bad)


def test_deep_count_reuses_paths_without_losing_depth_or_memberships():
    _, problem = build_scene(
        SceneRequest(8, Direction.NORTH, "count", "sole", count_value=0),
        identity="readable-v3-fixed-test",
        family="fixture",
    )
    evidence = build_evidence(problem)
    plan = make_plan(evidence)
    long_required = [p for p in plan["paths"] if len(p["indices"]) > 1]
    assert len(plan["chains"]) < len(long_required)
    assert all(any(contains_path(c, p) for c in plan["chains"]) for p in long_required)
    assert max(len(c["indices"]) for c in plan["chains"]) == 8
    menu = menus_for(evidence.semantics, 0)[0]
    assert selected_explanation_depth(evidence, menu, Contract.UNIQUE) == 8
    pair = render_pair(
        evidence, menu, Contract.UNIQUE, restate_premises=True, spatial_prose=True
    )
    assert pair["natural"].count("Chain 1:") == 1
    bad = deepcopy(plan)
    bad["chains"].pop(0)
    with pytest.raises(VerificationError):
        audit_plan(evidence, bad)
    bad = deepcopy(plan)
    bad["members"].pop()
    with pytest.raises(VerificationError):
        audit_plan(evidence, bad)


def test_grouping_never_merges_different_citations():
    path = {"axis": "x", "nodes": ["A", "B", "C"], "indices": [0, 1], "signs": [-1, -1]}
    assert contains_path(
        path, {"axis": "x", "nodes": ["A", "B"], "indices": [0], "signs": [-1]}
    )
    assert not contains_path(
        path, {"axis": "x", "nodes": ["A", "B"], "indices": [2], "signs": [-1]}
    )


def test_version_dispatch_and_reference_tampering():
    assert reference_options({"reference_version": SPATIAL_PROSE_VERSION}) == {
        "restate_premises": True,
        "spatial_prose": True,
    }
    evidence = build_evidence(BY_ID["D1"].problem())
    menu = menus_for(evidence.semantics, 0)[0]
    with pytest.raises(ValueError):
        render_pair(
            evidence, menu, Contract.LISTED, restate_premises=True, spatial_prose=True
        )
    pair = render_pair(
        evidence, menu, Contract.UNIQUE, restate_premises=True, spatial_prose=True
    )
    pair["natural"] = "Unsupported claim.\n\n" + pair["natural"]
    with pytest.raises(VerificationError):
        check_pair(
            evidence,
            menu,
            Contract.UNIQUE,
            pair,
            restate_premises=True,
            spatial_prose=True,
        )
