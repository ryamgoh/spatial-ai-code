"""Entailment-only semantics, independent expected labels, and version boundaries."""

from dataclasses import replace
from itertools import permutations

import pytest

from spatial.v2.composition_contracts import (
    VerificationError,
    answer_domain,
)
from spatial.v2.composition_contracts import (
    parse_prompt as parse_old_prompt,
)
from spatial.v2.composition_contracts import (
    render_prompt as render_old_prompt,
)
from spatial.v2.entailment_contracts import (
    Contract,
    analyze,
    check_counterexample,
    encode,
    parse_answer,
    parse_prompt,
    render_prompt,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.solver import (
    And,
    Direction,
    EntailmentAnalysis,
    SpatialSolverV2,
    WhichQuery,
    direction_constraint,
)
from spatial.v2.tests.test_pivot_evidence import BY_ID, CASES, ELIGIBLE

EXPECTED = {
    "D1": {"Northeast"},
    "D2": set(),
    "S1": {"U"},
    "S2": set(),
    "S3": {"U", "V"},
    "S4": set(),
    "S5": {"U", "V", "W", "T", "Y"},
    "S6": {"Y"},
    "C1": set(),
    "C2": {2},
    "C3": {0},
    "I1": set(),
    "I2": {"North"},
    "I3": set(),
}
VIEWS = (
    ("D1", ("West", "Northeast", "South", "Northwest"), ("B", "B", "B")),
    ("D1", ("North", "East", "South", "West"), ("F", "F", "F")),
    ("D2", ("West", "East", "South", "Northeast"), ("E", "F", "F")),
    ("S1", ELIGIBLE, ("A", "A", "A")),
    ("S2", ELIGIBLE, ("E", "F", "F")),
    ("S3", ELIGIBLE, ("E", "AB", "AB")),
    ("S4", ELIGIBLE, ("E", "F", "F")),
    ("S5", ELIGIBLE, ("E", "ABCD", "F")),
    ("S6", ELIGIBLE, ("F", "F", "F")),
    ("C1", (0, 1, 2, 3), ("E", "F", "F")),
    ("C2", (0, 1, 2, 3), ("C", "C", "C")),
    ("C3", (0, 1, 2, 3), ("A", "A", "A")),
    ("I1", ("North", "East", "South", "West"), ("E", "E", "E")),
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_full_domain_entailment_and_negative_models(case):
    result = analyze(case.problem())
    assert result.consistent == case.consistent
    assert result.entailed == EXPECTED[case.name]
    expected_negative = (
        set(answer_domain(case.problem())) - result.entailed
        if case.consistent
        else set()
    )
    assert set(result.counterexamples) == expected_negative
    for value, coordinates in result.counterexamples.items():
        check_counterexample(case.problem(), coordinates, value)


@pytest.mark.parametrize("name,menu,gold", VIEWS)
def test_truth_tables_and_all_menu_permutations(name, menu, gold):
    semantics = analyze(BY_ID[name].problem())
    assert tuple(encode(semantics, menu, c).letters for c in Contract) == gold
    for permuted in permutations(menu):
        for contract, original in zip(Contract, gold):
            answer = encode(semantics, permuted, contract).letters
            if original in ("E", "F"):
                assert answer == original
            else:
                assert {permuted["ABCD".index(c)] for c in answer} == {
                    menu["ABCD".index(c)] for c in original
                }
            assert parse_answer(answer, contract) == answer


def test_unique_selection_allows_additional_contingent_candidate():
    template = BY_ID["S4"].problem()
    problem = replace(
        template,
        premise=And(
            (
                direction_constraint("U", "O", Direction.NORTH),
                direction_constraint("W", "O", Direction.EAST),
                direction_constraint("T", "O", Direction.SOUTH),
            )
        ),
    )
    result = analyze(problem)
    assert result.entailed == {"U"}
    assert encode(result, ELIGIBLE, Contract.UNIQUE).letters == "A"
    old = SpatialSolverV2(backend="z3").analyze(problem, include_witnesses=False)
    assert "V" in old.contingent_entities


@pytest.mark.parametrize("name,menu,gold", VIEWS)
@pytest.mark.parametrize("contract", list(Contract))
def test_full_prompt_roundtrip_and_version_rejection(name, menu, gold, contract):
    problem = BY_ID[name].problem()
    prompt = render_prompt(problem, menu, contract)
    assert parse_prompt(prompt) == (problem, menu, contract)
    with pytest.raises(ValueError):
        parse_old_prompt(prompt)
    with pytest.raises(ValueError):
        parse_prompt(render_old_prompt(problem, menu, AnswerMode.SINGLE))
    with pytest.raises(ValueError):
        parse_prompt(
            prompt.replace("in every valid arrangement", "in some arrangement")
        )


@pytest.mark.parametrize("name", ("D1", "D2", "C1", "C2", "C3"))
def test_listed_exhaustive_identity_including_omissions(name):
    from itertools import combinations

    problem = BY_ID[name].problem()
    result = analyze(problem)
    for menu in combinations(answer_domain(problem), 4):
        assert (
            encode(result, menu, Contract.LISTED).letters
            == encode(result, menu, Contract.EXHAUSTIVE).letters
        )


def test_no_all_pairs_canonicalization_or_positive_enumeration(monkeypatch):
    solver = SpatialSolverV2(backend="z3")

    def fail(*args, **kwargs):
        pytest.fail("entailment analysis used legacy positive witness generation")

    monkeypatch.setattr(solver._engine, "_canonical_coordinates", fail)
    monkeypatch.setattr(solver._engine, "analyze", fail)
    monkeypatch.setattr(solver._engine, "analyze_which", fail)
    monkeypatch.setattr(solver._engine, "analyze_count", fail)
    for case in CASES:
        assert analyze(case.problem(), solver=solver).entailed == EXPECTED[case.name]


@pytest.mark.parametrize("error", ("unknown", "timeout", "solver failure"))
def test_unknown_is_not_inconsistency(monkeypatch, error):
    monkeypatch.setattr(
        SpatialSolverV2,
        "analyze_entailment",
        lambda *args: EntailmentAnalysis(False, error=error),
    )
    with pytest.raises(VerificationError, match="inconclusive"):
        analyze(BY_ID["D1"].problem())


def test_missing_counterexample_is_rejected(monkeypatch):
    problem = BY_ID["D2"].problem()
    solver = SpatialSolverV2(backend="z3")
    result = solver.analyze_entailment(problem)
    result.counterexamples.pop("East")
    monkeypatch.setattr(solver, "analyze_entailment", lambda *args: result)
    with pytest.raises(VerificationError, match="cover every"):
        analyze(problem, solver=solver)


def test_boolean_is_not_an_exact_count_value(monkeypatch):
    problem = BY_ID["C2"].problem()
    solver = SpatialSolverV2(backend="z3")
    result = replace(solver.analyze_entailment(problem), entailed_values=(True,))
    monkeypatch.setattr(solver, "analyze_entailment", lambda *args: result)
    with pytest.raises(VerificationError, match="invalid entailed"):
        analyze(problem, solver=solver)


def test_counterexample_that_satisfies_claim_is_rejected():
    result = analyze(BY_ID["D2"].problem())
    wrong = {"O": (0, 0), "X": (2, 0), "Y": (1, 0)}
    with pytest.raises(VerificationError, match="does not falsify"):
        check_counterexample(result.problem, wrong, "East")


def test_legacy_semantics_cannot_be_mislabelled():
    from spatial.v2.composition_contracts import analyze as old_analyze

    with pytest.raises(TypeError, match="entailment-only"):
        encode(old_analyze(BY_ID["S4"].problem()), ELIGIBLE, Contract.LISTED)


@pytest.mark.parametrize("name", ("D1", "D2", "S1", "S4", "C1", "C2"))
def test_reference_backend_crosscheck(name):
    problem = BY_ID[name].problem()
    reference = SpatialSolverV2(backend="reference").analyze(
        problem, include_witnesses=False
    )
    if isinstance(problem.query, WhichQuery):
        expected = set(reference.entailed_entities)
    else:
        values = (
            [d.value for d in reference.possible_directions]
            if hasattr(reference, "possible_directions")
            else list(reference.possible_counts)
        )
        expected = set(values) if len(values) == 1 else set()
    assert analyze(problem).entailed == expected
