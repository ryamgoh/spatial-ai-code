"""Solver-only Count diagnostic; never admitted to the atomic training curriculum."""

from dataclasses import replace

import pytest

from spatial.v2.entailment_contracts import analyze
from spatial.v2.solver import SpatialSolverV2, WhichQuery
from spatial.v2.tests.test_count_certificates import correlated_problem


def test_entailment_count_is_joint_when_member_identities_change():
    problem, _ = correlated_problem()
    solver = SpatialSolverV2("z3")
    count = solver.analyze_entailment(problem)
    memberships = solver.analyze_entailment(
        replace(
            problem,
            query=WhichQuery(
                problem.query.directions, problem.query.reference,
                problem.query.candidates,
            ),
        )
    )
    assert count.error is None and count.consistent
    assert set(count.entailed_values) == {1}
    assert memberships.error is None and memberships.consistent
    assert not memberships.entailed_values
    for coordinates in memberships.counterexamples.values():
        qualifies = [
            coordinates[name][0] > coordinates["R"][0]
            and coordinates[name][1] > coordinates["R"][1]
            for name in ("A", "B")
        ]
        assert sum(qualifies) == 1
    # This diagnostic is not permission to emit Boolean-premise examples.
    with pytest.raises(ValueError, match="flat conjunction"):
        analyze(problem, solver=solver)
