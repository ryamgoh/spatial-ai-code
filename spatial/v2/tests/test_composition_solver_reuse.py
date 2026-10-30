"""Solver reuse must retain exact witnesses and never reuse unknown results."""

from copy import deepcopy

from spatial.v2.composition_scene_sampler import sample_scene
from spatial.v2.composition_targeted_sampler import sample_targeted
from spatial.v2.solver import SpatialSolverV2, membership_constraint


def test_repeated_analysis_and_membership_do_not_rebuild_positive_maps(monkeypatch):
    from spatial.v2.composition_solver_reuse import QuestionSolver

    problem = sample_scene(3, 2, matches=2).problems()[1]
    original = SpatialSolverV2(backend="z3")
    expected = original.analyze(problem)
    expected_claims = [
        original.assess(problem, membership_constraint(c, problem.query))
        for c in problem.query.candidates
    ]
    solver = QuestionSolver()
    calls = []
    canonical = solver._engine._canonical_coordinates

    def counted(*args):
        calls.append(1)
        return canonical(*args)

    monkeypatch.setattr(solver._engine, "_canonical_coordinates", counted)
    actual = solver.analyze(problem)
    first_calls = len(calls)
    assert actual == expected
    assert solver.analyze(problem) == expected
    assert len(calls) == first_calls
    for candidate, result in zip(problem.query.candidates, expected_claims):
        claim = membership_constraint(candidate, problem.query)
        assert solver.assess(problem, claim) == result
    # Only counterexample maps for impossible candidates are missing from analysis.
    assert len(calls) == first_calls + len(problem.query.candidates) - 2
    actual.witnesses.clear()
    assert solver.analyze(problem) == expected


def test_cache_is_bounded_and_clearable():
    from spatial.v2.composition_solver_reuse import QuestionSolver

    solver = QuestionSolver(cache_size=2)
    for problem in sample_scene(3, 1).problems():
        solver.analyze(problem, include_witnesses=False)
    assert len(solver._results) == 2
    solver.clear()
    assert not solver._results


def test_unknown_is_not_cached(monkeypatch):
    from spatial.v2.composition_solver_reuse import QuestionSolver

    problem = sample_scene(3, 1).problems()[0]
    solver = QuestionSolver()
    expected = deepcopy(solver.analyze(problem, include_witnesses=False))
    solver.clear()
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("diagnostic unknown")
        return expected

    monkeypatch.setattr(solver, "_analyze_direction", fail_once)
    assert solver.analyze(problem, include_witnesses=False).error
    assert solver.analyze(problem, include_witnesses=False) == expected
    assert len(calls) == 2


def test_contingent_membership_reuses_exact_positive_and_negative_maps():
    from spatial.v2.composition_solver_reuse import QuestionSolver

    problem = sample_targeted(3, "contingent").problems()[1]
    original = SpatialSolverV2(backend="z3")
    solver = QuestionSolver()
    assert solver.analyze(problem) == original.analyze(problem)
    for candidate in problem.query.candidates:
        claim = membership_constraint(candidate, problem.query)
        assert solver.assess(problem, claim) == original.assess(problem, claim)


def test_claim_timeout_is_not_cached(monkeypatch):
    from spatial.v2.composition_solver_reuse import QuestionSolver

    problem = sample_scene(3, 1, matches=0).problems()[1]
    solver = QuestionSolver()
    solver.analyze(problem)
    actual = solver._engine.satisfy
    calls = []

    def once(*args):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("diagnostic unknown")
        return actual(*args)

    monkeypatch.setattr(solver._engine, "satisfy", once)
    claim = membership_constraint(problem.query.candidates[0], problem.query)
    assert solver.assess(problem, claim).error
    assert not solver.assess(problem, claim).error
    assert len(calls) == 2
