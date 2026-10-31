"""Independent evidence replay and matched entailment-only reference explanations."""

import re
from copy import deepcopy
from dataclasses import replace

import pytest

from spatial.v2.composition_contracts import UnsupportedEvidence, VerificationError
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract, encode, parse_answer
from spatial.v2.entailment_evidence import (
    build_evidence,
    check_pair,
    render_pair,
    verify_evidence,
)
from spatial.v2.tests.test_entailment_contracts import VIEWS
from spatial.v2.tests.test_pivot_evidence import BY_ID


@pytest.mark.parametrize("name,menu,expected", VIEWS)
@pytest.mark.parametrize("contract", list(Contract))
def test_all_fixture_views_have_checked_matched_explanations(
    name, menu, expected, contract
):
    solver = QuestionSolver()
    evidence = build_evidence(BY_ID[name].problem(), solver=solver)
    pair = render_pair(evidence, menu, contract, solver=solver)
    check_pair(evidence, menu, contract, pair, solver=solver)
    assert set(pair) == {"natural", "symbolic"}
    for text in pair.values():
        assert (
            parse_answer(text, contract, stage=1)
            == encode(evidence.semantics, menu, contract).letters
        )
        assert "possible set" not in text
        assert ":(0," not in text
        assert "points" not in text
    assert pair["natural"] != pair["symbolic"]


def test_contingent_claim_is_rejected_not_called_impossible():
    evidence = build_evidence(BY_ID["S4"].problem())
    text = render_pair(evidence, ("U", "V", "W", "T"), Contract.LISTED)["natural"]
    assert "The claim that U is north of O is false in Case" in text
    assert "so it is not entailed" in text
    assert "U cannot qualify" not in text
    assert "No displayed answer is entailed." in text
    assert "No domain answer is entailed." not in text
    assert text.endswith("Answer: F")


def test_exhaustive_omission_uses_positive_proof_not_possible_witness():
    evidence = build_evidence(BY_ID["S5"].problem())
    pair = render_pair(evidence, ("U", "V", "W", "T"), Contract.EXHAUSTIVE)
    for text in pair.values():
        assert "Y" in text
        assert "in every valid arrangement" in text
        assert "Case" not in text
        assert text.endswith("Answer: F")


def test_unique_selection_unlisted_proves_uniqueness_over_full_domain():
    evidence = build_evidence(BY_ID["S6"].problem())
    pair = render_pair(evidence, ("U", "V", "W", "T"), Contract.UNIQUE)
    assert "Y" in pair["natural"]
    assert "in every valid arrangement" in pair["natural"]
    for candidate in ("U", "V", "W", "T"):
        assert (
            f"The claim that {candidate} is north of O is false in Case"
            in pair["natural"]
        )
        assert f"The claim that N({candidate},O) is false in Case" in pair["symbolic"]
    assert all(text.endswith("Answer: F") for text in pair.values())


def test_count_zero_is_proved_not_an_empty_entailment_set():
    evidence = build_evidence(BY_ID["C3"].problem())
    assert evidence.semantics.entailed == {0}
    pair = render_pair(evidence, (0, 1, 2, 3), Contract.UNIQUE)
    assert "Exactly 0 qualify in every valid arrangement" in pair["natural"]
    assert pair["natural"].endswith("Answer: A")


@pytest.mark.parametrize("kind", ("missing-axis", "wrong-endpoint", "wrong-citation"))
def test_corrupted_positive_proofs_are_rejected(kind):
    evidence = build_evidence(BY_ID["D1"].problem())
    positive = deepcopy(evidence.positive_support)
    paths = positive["Northeast"]["paths"]
    if kind == "missing-axis":
        paths.pop()
    elif kind == "wrong-endpoint":
        paths[0]["nodes"][0] = "Y"
    else:
        paths[0]["indices"][0] = 99
    with pytest.raises((VerificationError, UnsupportedEvidence)):
        verify_evidence(replace(evidence, positive_support=positive))


def test_count_proof_cannot_skip_candidate_or_change_count():
    evidence = build_evidence(BY_ID["C2"].problem())
    support = deepcopy(evidence.positive_support)
    support[2]["memberships"].pop()
    with pytest.raises(VerificationError, match="every eligible"):
        verify_evidence(replace(evidence, positive_support=support))


def test_negative_map_and_text_corruption_are_rejected():
    evidence = build_evidence(BY_ID["D2"].problem())
    maps = deepcopy(evidence.semantics.counterexamples)
    maps["East"] = {"O": (0, 0), "X": (2, 0), "Y": (1, 0)}
    with pytest.raises(VerificationError, match="falsify"):
        verify_evidence(
            replace(
                evidence, semantics=replace(evidence.semantics, counterexamples=maps)
            )
        )
    menu = ("North", "East", "South", "West")
    pair = render_pair(evidence, menu, Contract.LISTED)
    pair["natural"] = pair["natural"].replace("not entailed", "entailed")
    with pytest.raises(VerificationError, match="matched rendering"):
        check_pair(evidence, menu, Contract.LISTED, pair)


def test_new_cache_preserves_negative_maps_and_clears_per_question(monkeypatch):
    solver = QuestionSolver()
    calls = []
    original = solver._engine.analyze_entailment

    def counted(problem):
        calls.append(problem)
        return original(problem)

    monkeypatch.setattr(solver._engine, "analyze_entailment", counted)
    problem = BY_ID["D2"].problem()
    first = solver.analyze_entailment(problem)
    first.counterexamples.clear()
    second = solver.analyze_entailment(problem)
    assert second.counterexamples
    assert len(calls) == 1
    solver.clear()
    solver.analyze_entailment(problem)
    assert len(calls) == 2


def test_evidence_generation_does_not_canonicalize_maps(monkeypatch):
    solver = QuestionSolver()

    def fail(*args, **kwargs):
        pytest.fail("entailment evidence performed all-pairs canonicalization")

    monkeypatch.setattr(solver._engine, "_canonical_coordinates", fail)
    for name in ("D1", "D2", "S1", "S4", "C1", "C2", "C3", "I1"):
        build_evidence(BY_ID[name].problem(), solver=solver)


@pytest.mark.parametrize("name,menu,expected", VIEWS)
@pytest.mark.parametrize("contract", list(Contract))
def test_self_contained_keys_define_every_cited_premise(name, menu, expected, contract):
    problem = BY_ID[name].problem()
    evidence = build_evidence(problem)
    old = render_pair(evidence, menu, contract)
    pair = render_pair(evidence, menu, contract, restate_premises=True)
    for style, text in pair.items():
        assert text.rsplit("Answer: ", 1)[1] == old[style].rsplit("Answer: ", 1)[1]
        assert not re.search(r"\bp\d+\b", text)
        citations = set(re.findall(r"\bP(\d+)\b", text))
        if not citations:
            assert not text.startswith("Premises used")
            continue
        key, body = text.split("\n\n", 1)
        assert key.startswith(
            "Premises used (labels follow their order in the question):\n"
        )
        definitions = re.findall(r"^P(\d+): (.+)$", key, re.MULTILINE)
        assert [int(i) for i, _ in definitions] == sorted(int(i) for i in citations)
        assert set(re.findall(r"\bP(\d+)\b", body)) == citations
        for index, statement in definitions:
            atom = problem.premise.operands[int(index) - 1]
            assert statement == (
                f"{atom.subject} is to the {next(iter(atom.allowed)).value} of {atom.reference}."
            )
    assert pair["natural"].split("\n\n", 1)[0] == pair["symbolic"].split("\n\n", 1)[
        0
    ] or not pair["natural"].startswith("Premises used")
    check_pair(evidence, menu, contract, pair, restate_premises=True)


def test_changed_premise_restatement_is_rejected():
    evidence = build_evidence(BY_ID["D1"].problem())
    menu = ("North", "Northeast", "South", "West")
    pair = render_pair(evidence, menu, Contract.UNIQUE, restate_premises=True)
    pair["symbolic"] = pair["symbolic"].replace(" is to the ", " could be ", 1)
    with pytest.raises(VerificationError, match="matched rendering"):
        check_pair(evidence, menu, Contract.UNIQUE, pair, restate_premises=True)
