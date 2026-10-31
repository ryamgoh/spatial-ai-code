"""Bounded all-scene constructions, not a production allocation approval."""

import pytest

from spatial.v2.composition_depth import membership_support
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_balance_probe import probe_scene
from spatial.v2.entailment_dataset import (
    measure_depth,
    uniqueness_failure_depth,
    verify_transformations,
)
from spatial.v2.entailment_evidence import build_evidence
from spatial.v2.solver import Direction


@pytest.mark.parametrize("depth", (1, 2, 8, 10))
@pytest.mark.parametrize("direction", tuple(Direction))
@pytest.mark.parametrize("matches", (1, 2, 3, 4))
def test_all_scene_matches_and_decision_depth(depth, direction, matches):
    story, problem = probe_scene(3, depth, direction, matches)
    solver = QuestionSolver()
    evidence = build_evidence(problem, solver=solver)
    assert len(evidence.semantics.entailed) == matches
    assert set(problem.query.candidates) == set(problem.objects) - {
        problem.query.reference
    }
    assert measure_depth(problem, solver=solver)["depth"] == depth
    assert uniqueness_failure_depth(evidence) == (depth if matches > 1 else None)
    for support in evidence.positive_support.values():
        assert max(len(p["indices"]) for p in support["paths"]) == depth
    for candidate in set(problem.query.candidates) - evidence.semantics.entailed:
        assert membership_support(problem, candidate, False)["depth"] <= depth
    verify_transformations(story, problem, evidence.semantics, solver, scene_scope=True)


def test_variants_are_not_independent_families():
    family = probe_scene(3, 1, Direction.NORTH, 1)[0].source_family
    assert probe_scene(3, 10, Direction.NORTHEAST, 4)[0].source_family == family
    assert probe_scene(4, 10, Direction.NORTHEAST, 4)[0].source_family != family


@pytest.mark.parametrize(
    "arguments",
    (
        (-1, 2, Direction.NORTH, 2),
        (64, 2, Direction.NORTH, 2),
        (3, 0, Direction.NORTH, 2),
        (3, 11, Direction.NORTH, 2),
        (3, 2, Direction.NORTH, 0),
        (3, 2, Direction.NORTH, 5),
    ),
)
def test_probe_bounds_are_explicit(arguments):
    with pytest.raises(ValueError):
        probe_scene(*arguments)
