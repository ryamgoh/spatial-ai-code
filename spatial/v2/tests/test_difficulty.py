"""Semantic support includes excluded answers and jointly removable redundancy."""

from spatial.v2.difficulty import measure_difficulty
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    Not,
    Or,
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
    direction_constraint,
)


def test_ambiguous_direction_keeps_premise_restricting_possibilities():
    fact = Or(
        (
            direction_constraint("A", "R", Direction.NORTH),
            direction_constraint("A", "R", Direction.SOUTH),
        )
    )
    problem = SpatialProblem(("A", "R"), And((fact,)), DirectionQuery("A", "R"))
    solver = SpatialSolverV2()
    difficulty = measure_difficulty(problem, solver.analyze(problem), solver)
    assert difficulty["supporting_premise_indices"] == [0]
    assert difficulty["num_distractor_premises"] == 0


def test_negative_membership_evidence_is_support_for_which_and_count():
    for query_type in (WhichQuery, CountQuery):
        query = query_type(frozenset({Direction.NORTH}), "R", ("A", "B"))
        problem = SpatialProblem(
            ("A", "B", "R"),
            And(
                (
                    direction_constraint("A", "R", Direction.NORTH),
                    Not(direction_constraint("B", "R", Direction.NORTH)),
                )
            ),
            query,
        )
        solver = SpatialSolverV2()
        difficulty = measure_difficulty(problem, solver.analyze(problem), solver)
        assert difficulty["supporting_premise_indices"] == [0, 1]
        assert difficulty["num_distractor_premises"] == 0


def test_redundant_exclusions_are_not_all_marked_distractors():
    exclusion = Not(direction_constraint("A", "R", Direction.NORTH))
    problem = SpatialProblem(
        ("A", "R"),
        And((exclusion, exclusion)),
        WhichQuery(frozenset({Direction.NORTH}), "R", ("A",)),
    )
    solver = SpatialSolverV2()
    difficulty = measure_difficulty(problem, solver.analyze(problem), solver)
    assert len(difficulty["supporting_premise_indices"]) == 1
    assert difficulty["num_distractor_premises"] == 1


def test_semantic_signature_matches_full_analysis_on_both_backends():
    for backend in ("z3", "reference"):
        solver = SpatialSolverV2(backend=backend)
        a = direction_constraint("A", "R", Direction.NORTH)
        b = direction_constraint("B", "R", Direction.SOUTH)
        premise = And((a, Or((b, Not(b)))))
        queries = (
            DirectionQuery("B", "R"),
            WhichQuery(frozenset({Direction.NORTH}), "R", ("A", "B")),
            CountQuery(frozenset({Direction.NORTH}), "R", ("A", "B")),
        )
        for query in queries:
            problem = SpatialProblem(("A", "B", "R"), premise, query)
            analysis = solver.analyze(problem)
            signature = solver.semantic_signature(problem)
            if isinstance(query, DirectionQuery):
                expected = (True, analysis.possible_directions)
            elif isinstance(query, WhichQuery):
                expected = (
                    True,
                    analysis.possible_entities,
                    analysis.entailed_entities,
                )
            else:
                membership = solver.analyze(
                    SpatialProblem(problem.objects, premise, queries[1])
                )
                expected = (
                    True,
                    analysis.possible_counts,
                    membership.possible_entities,
                    membership.entailed_entities,
                )
            assert signature == expected


def test_semantic_signature_skips_canonical_models_and_propagates_unknown(monkeypatch):
    import pytest

    solver = SpatialSolverV2(backend="z3")
    fact = direction_constraint("A", "R", Direction.NORTH)
    problem = SpatialProblem(("A", "R"), fact, DirectionQuery("A", "R"))

    def reject_model(*args):
        raise AssertionError("semantic support requested a canonical witness")

    monkeypatch.setattr(solver._engine, "_canonical_coordinates", reject_model)
    assert solver.semantic_signature(problem) == (True, (Direction.NORTH,))

    def unknown(*args):
        raise RuntimeError("z3 returned unknown: timeout")

    monkeypatch.setattr(solver._engine, "_check", unknown)
    with pytest.raises(RuntimeError, match="unknown"):
        solver.semantic_signature(problem)


def test_unknown_support_check_keeps_premises(monkeypatch):
    solver = SpatialSolverV2()
    excluded = Not(direction_constraint("A", "R", Direction.NORTH))
    problem = SpatialProblem(
        ("A", "R"),
        And((excluded, excluded)),
        WhichQuery(frozenset({Direction.NORTH}), "R", ("A",)),
    )
    analysis = solver.analyze(problem)

    def unknown(*args):
        raise RuntimeError("z3 returned unknown: timeout")

    monkeypatch.setattr(solver, "semantic_signature", unknown)
    difficulty = measure_difficulty(problem, analysis, solver)
    assert difficulty["supporting_premise_indices"] == [0, 1]
    assert difficulty["num_distractor_premises"] == 0
    assert difficulty["support_checks_complete"] is False
