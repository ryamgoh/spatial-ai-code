"""Replacement preparation preserves semantics while varying real relations."""

from copy import deepcopy

import pytest

from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_dataset import (
    measure_depth,
    menus_for,
    selected_explanation_depth,
)
from spatial.v2.entailment_evidence import (
    CONCISE_VERSION,
    build_evidence,
    check_pair,
    reference_options,
    render_pair,
)
from spatial.v2.entailment_generator import SceneRequest, build_scene
from spatial.v2.solver import Direction
from spatial.v2.tests.test_pivot_evidence import BY_ID


@pytest.mark.parametrize("span", (1, 8))
@pytest.mark.parametrize("direction", tuple(Direction))
@pytest.mark.parametrize("count", range(5))
def test_structural_variation_preserves_exact_count_and_depth(span, direction, count):
    request = SceneRequest(span, direction, "count", "sole", count_value=count)
    _, problem = build_scene(
        request,
        identity="variation-test",
        family="fixture",
        structure_seed=2034103100 + count,
    )
    solver = QuestionSolver()
    evidence = build_evidence(problem, solver=solver)
    assert evidence.semantics.entailed == {count}
    assert measure_depth(problem, solver=solver)["depth"] == span


@pytest.mark.parametrize("direction", tuple(Direction))
def test_sole_which_with_contingent_context(direction):
    request = SceneRequest(1, direction, "which", "sole")
    _, problem = build_scene(
        request,
        identity="contingent-test",
        family="fixture",
        structure_seed=2034103101,
        contingent=True,
    )
    solver = QuestionSolver()
    evidence = build_evidence(problem, solver=solver)
    assert len(evidence.semantics.entailed) == 1
    assert "contingent" in solver.negative_categories(problem).values()
    for menu in menus_for(evidence.semantics, 0, omission_percent=100):
        assert selected_explanation_depth(evidence, menu, Contract.UNIQUE) == 1


@pytest.mark.parametrize("kind", ("direction", "which", "count"))
@pytest.mark.parametrize("case", ("no-entailed", "inconsistent"))
@pytest.mark.parametrize("variant", (0, 1))
def test_e_variants_survive_real_context_variation(kind, case, variant):
    request = SceneRequest(4, Direction.NORTHEAST, kind, case, variant=variant)
    _, problem = build_scene(
        request, identity="e-test", family="fixture", structure_seed=2034103102
    )
    evidence = build_evidence(problem)
    assert not evidence.semantics.entailed
    assert evidence.semantics.consistent == (case == "no-entailed")


def test_variation_is_not_only_renaming():
    request = SceneRequest(1, Direction.NORTH, "direction", "sole")
    keys = {
        query_profile(
            build_scene(
                request,
                identity=f"fixture-{seed}",
                family="fixture",
                structure_seed=2034103100 + seed,
            )[1]
        )["key"]
        for seed in range(32)
    }
    # This checks that variation is structural, not that a sample proves supply.
    # The production preflight separately enforces every distinct-question quota.
    assert 1 < len(keys) <= 32


@pytest.mark.parametrize(
    "name", ("D1", "D2", "S1", "S4", "S5", "S6", "C1", "C2", "C3", "I1")
)
def test_concise_rendering_is_checked_and_does_not_mutate_evidence(name):
    evidence = build_evidence(BY_ID[name].problem())
    before = deepcopy(evidence)
    for menu in menus_for(evidence.semantics, 0, omission_percent=100):
        old = render_pair(evidence, menu, Contract.UNIQUE, restate_premises=True)
        pair = render_pair(
            evidence, menu, Contract.UNIQUE, restate_premises=True, concise=True
        )
        check_pair(
            evidence, menu, Contract.UNIQUE, pair, restate_premises=True, concise=True
        )
        assert evidence == before
        for style, text in pair.items():
            assert text.rsplit("Answer: ", 1)[1] == old[style].rsplit("Answer: ", 1)[1]
            assert "every contract" not in text
            assert "No minimality claim" not in text
            assert "Unique requires" not in text
        if name in ("D1", "C1", "C2", "C3") and len(evidence.semantics.entailed) == 1:
            assert all("so it is not entailed." not in text for text in pair.values())
    assert reference_options({"reference_version": CONCISE_VERSION}) == {
        "restate_premises": True,
        "concise": True,
    }


def test_unique_which_keeps_other_object_rejections():
    evidence = build_evidence(BY_ID["S6"].problem())
    pair = render_pair(
        evidence,
        ("U", "V", "W", "T"),
        Contract.UNIQUE,
        restate_premises=True,
        concise=True,
    )
    for candidate in ("U", "V", "W", "T"):
        assert f"The claim that {candidate} is north of O is false" in pair["natural"]
    assert pair["natural"].endswith("Answer: F")


def test_concise_rendering_rejects_wrong_contract_and_mutation():
    evidence = build_evidence(BY_ID["D1"].problem())
    menu = menus_for(evidence.semantics, 0)[0]
    with pytest.raises(ValueError):
        render_pair(
            evidence, menu, Contract.LISTED, restate_premises=True, concise=True
        )
    pair = render_pair(
        evidence, menu, Contract.UNIQUE, restate_premises=True, concise=True
    )
    pair["natural"] = "Unsupported assertion.\n\n" + pair["natural"]
    with pytest.raises(ValueError):
        check_pair(
            evidence, menu, Contract.UNIQUE, pair, restate_premises=True, concise=True
        )
