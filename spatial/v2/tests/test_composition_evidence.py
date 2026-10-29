"""Expanded composition evidence coverage, mutation rejection, and invariance."""

from dataclasses import replace

import pytest

from spatial.v2.composition_conflicts import (
    build_conflict,
    check_conflict,
    verify_conflict,
)
from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    encode,
    parse_answer,
    parse_prompt,
)
from spatial.v2.composition_evidence import (
    build_evidence,
    build_example,
    render_pair,
    verify_evidence,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.model_certificates import ModelCheckError
from spatial.v2.proofs import ProofCheckError
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    SpatialSolverV2,
    WhichQuery,
)
from spatial.v2.tests.test_pivot_evidence import BY_ID, CASES, VIEWS
from spatial.v2.which_certificates import WhichCertificateCheckError

MODES = (AnswerMode.SINGLE, AnswerMode.VISIBLE_POSSIBLE, AnswerMode.ALL_POSSIBLE)


@pytest.mark.parametrize("name,menu,expected", VIEWS)
@pytest.mark.parametrize("mode_index", range(3))
def test_all_fixture_views_have_checked_pairs(name, menu, expected, mode_index):
    problem, mode = BY_ID[name].problem(), MODES[mode_index]
    row = build_example(problem, menu, mode, story_id=name, family_id=name)
    assert row["answer"] == expected[mode_index]
    assert parse_prompt(row["prompt"]) == (problem, menu, mode)
    for text in row["references"].values():
        assert parse_answer(text, mode, stage=1) == row["answer"]
    assert row["references"]["natural"] != row["references"]["symbolic"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_rename_and_order_variants(case):
    problem = case.problem()
    names = {obj: f"Object{i}" for i, obj in enumerate(problem.objects)}
    query = problem.query
    if isinstance(query, DirectionQuery):
        query = replace(
            query, target=names[query.target], reference=names[query.reference]
        )
    else:
        query = replace(
            query,
            candidates=tuple(names[c] for c in query.candidates),
            reference=names[query.reference],
        )
    variant = replace(
        problem,
        query=query,
        objects=tuple(names[obj] for obj in problem.objects),
        premise=And(
            tuple(
                replace(
                    atom, subject=names[atom.subject], reference=names[atom.reference]
                )
                for atom in reversed(problem.premise.operands)
            )
        ),
    )
    evidence = build_evidence(variant)
    expected = (
        frozenset(names[c] for c in case.possible)
        if isinstance(query, WhichQuery)
        else case.possible
    )
    assert evidence.semantics.possible == expected
    assert evidence.semantics.consistent == case.consistent


@pytest.mark.parametrize("name,menu,expected", VIEWS)
def test_menu_permutation_preserves_evidence_and_changes_letters(name, menu, expected):
    evidence = build_evidence(BY_ID[name].problem())
    permuted = tuple(reversed(menu))
    for mode in MODES:
        pair = render_pair(evidence, permuted, mode)
        decision = encode(evidence.semantics, permuted, mode)
        for text in pair.values():
            assert parse_answer(text, mode, stage=1) == decision.letters


@pytest.mark.parametrize("name", ("D2", "S4", "C1"))
def test_missing_witness_prevents_rendering(name):
    evidence = build_evidence(BY_ID[name].problem())
    witnesses = dict(evidence.semantics.witnesses)
    witnesses.pop(next(iter(witnesses)))
    corrupted = replace(
        evidence, semantics=replace(evidence.semantics, witnesses=witnesses)
    )
    with pytest.raises(VerificationError, match="every possible value"):
        verify_evidence(corrupted)


def test_missing_direction_exclusion_rejected():
    evidence = build_evidence(BY_ID["D2"].problem())
    cert = evidence.certificate
    with pytest.raises(ValueError, match="every declared direction"):
        verify_evidence(
            replace(
                evidence, certificate=replace(cert, candidates=cert.candidates[:-1])
            )
        )


def test_missing_membership_exclusion_rejected():
    evidence = build_evidence(BY_ID["S1"].problem())
    cert = evidence.certificate
    with pytest.raises(WhichCertificateCheckError):
        verify_evidence(
            replace(
                evidence, certificate=replace(cert, candidates=cert.candidates[:-1])
            )
        )


def test_contingent_selection_needs_a_real_counterexample():
    evidence = build_evidence(BY_ID["S4"].problem())
    cert = evidence.certificate
    item = cert.candidates[0]
    contingent = item.evidence
    false_counterexample = replace(
        contingent.counterexample, assignments=contingent.witness.assignments
    )
    item = replace(
        item, evidence=replace(contingent, counterexample=false_counterexample)
    )
    with pytest.raises((WhichCertificateCheckError, ModelCheckError)):
        verify_evidence(
            replace(
                evidence,
                certificate=replace(cert, candidates=(item, *cert.candidates[1:])),
            )
        )


def test_joint_count_cannot_drop_pair_exclusion():
    evidence = build_evidence(BY_ID["C1"].problem())
    assert evidence.semantics.possible == frozenset({0, 1})
    assert evidence.certificate.incompatible_pairs
    with pytest.raises(UnsupportedEvidence, match="joint assignment"):
        verify_evidence(
            replace(
                evidence,
                certificate=replace(evidence.certificate, incompatible_pairs=()),
            )
        )


def test_count_pair_cannot_use_an_unrelated_conflict():
    evidence = build_evidence(BY_ID["C1"].problem())
    pair = evidence.certificate.incompatible_pairs[0]
    corrupt = replace(pair, conflict=build_conflict(BY_ID["I1"].problem()))
    with pytest.raises(VerificationError, match="pair conflict"):
        verify_evidence(
            replace(
                evidence,
                certificate=replace(
                    evidence.certificate, incompatible_pairs=(corrupt,)
                ),
            )
        )


@pytest.mark.parametrize("name", ("I1", "I3"))
def test_conflict_subset_is_checked_and_tampering_fails(name):
    evidence = build_evidence(BY_ID[name].problem())
    conflict = evidence.certificate
    verify_conflict(conflict, SpatialSolverV2(backend="z3"))
    first = conflict.paths[0][0]
    invalid = replace(first, premise_index=100)
    corrupt = replace(
        conflict, paths=((invalid, *conflict.paths[0][1:]), *conflict.paths[1:])
    )
    with pytest.raises(VerificationError, match="premise edge"):
        check_conflict(corrupt)
    if name == "I1":
        assert conflict.premise_indices == (0, 1, 2)
    else:
        assert conflict.kind == "forced-coincidence"
        assert len(conflict.paths) == 2
        with pytest.raises(VerificationError, match="both axes"):
            check_conflict(replace(conflict, paths=conflict.paths[:1]))


def test_consistent_cycle_is_not_a_conflict():
    with pytest.raises(UnsupportedEvidence):
        build_conflict(BY_ID["I2"].problem())


def test_equality_cycle_alone_is_not_a_strict_conflict():
    from spatial.v2.composition_conflicts import Conflict, Edge

    problem = BY_ID["I2"].problem()
    # The East premise yields equal y coordinates, not a strict-order cycle.
    path = (
        Edge(0, 1, "X", "Y", False),
        Edge(0, 1, "Y", "X", False),
    )
    with pytest.raises(VerificationError, match="strict edge"):
        check_conflict(Conflict(problem, "strict-cycle", (path,)))


def test_count_domain_above_feasibility_limit_is_explicitly_unsupported():
    template = BY_ID["C3"].problem()
    candidates = tuple(f"Point{i}" for i in range(9))
    problem = replace(
        template,
        objects=("O", *candidates),
        query=replace(template.query, candidates=candidates),
        premise=And(
            tuple(
                replace(template.premise.operands[0], subject=obj) for obj in candidates
            )
        ),
    )
    with pytest.raises(UnsupportedEvidence, match="eight eligible objects"):
        build_evidence(problem)


def test_subset_timeout_never_counts_as_inconsistency(monkeypatch):
    solver = SpatialSolverV2(backend="z3")
    conflict = build_conflict(BY_ID["I1"].problem())
    original = solver.analyze
    monkeypatch.setattr(
        solver, "analyze", lambda *a, **kw: replace(original(*a, **kw), error="timeout")
    )
    with pytest.raises(VerificationError, match="inconclusive"):
        verify_conflict(conflict, solver)


@pytest.mark.parametrize("name", ("S1", "C1"))
def test_membership_solver_timeout_is_rejected(monkeypatch, name):
    solver = SpatialSolverV2(backend="z3")
    original = solver.assess
    monkeypatch.setattr(
        solver, "assess", lambda *a, **kw: replace(original(*a, **kw), error="timeout")
    )
    with pytest.raises(UnsupportedEvidence, match="timeout"):
        build_evidence(BY_ID[name].problem(), solver=solver)


def test_partial_exhaustive_uses_one_unlisted_witness():
    evidence = build_evidence(BY_ID["D2"].problem())
    pair = render_pair(
        evidence, ("North", "East", "South", "Northeast"), AnswerMode.ALL_POSSIBLE
    )
    for text in pair.values():
        assert "West" in text
        assert "unlisted" in text
        assert text.count("valid map") + text.count("Valid map") == 1
        assert "Answer: F" in text


def test_selection_no_match_and_multiple_matches_do_not_invent_ambiguity():
    for name, phrase in (("S2", "No eligible object"), ("S3", "Multiple objects")):
        evidence = build_evidence(BY_ID[name].problem())
        pair = render_pair(evidence, ("U", "V", "W", "T"), AnswerMode.SINGLE)
        for text in pair.values():
            assert phrase in text
            assert "These valid maps give different answers" not in text


def test_empty_selection_and_zero_count_have_distinct_conclusions():
    selection = render_pair(
        build_evidence(BY_ID["S2"].problem()),
        ("U", "V", "W", "T"),
        AnswerMode.ALL_POSSIBLE,
    )
    count = render_pair(
        build_evidence(BY_ID["C3"].problem()), (0, 1, 2, 3), AnswerMode.ALL_POSSIBLE
    )
    assert selection["natural"].endswith("Answer: F")
    assert count["natural"].endswith("Answer: A")
    assert "possible set is {0}" in count["natural"]


def test_broken_proof_step_is_rejected():
    evidence = build_evidence(BY_ID["S1"].problem())
    cert = evidence.certificate
    item = cert.candidates[0]
    proof = item.evidence.proof
    corrupted = replace(
        item,
        evidence=replace(item.evidence, proof=replace(proof, steps=proof.steps[:-1])),
    )
    with pytest.raises((WhichCertificateCheckError, ProofCheckError)):
        verify_evidence(
            replace(
                evidence,
                certificate=replace(cert, candidates=(corrupted, *cert.candidates[1:])),
            )
        )


def test_fixture_pack_keeps_related_stories_and_views_together():
    from spatial.v2.tests.build_composition_pack import build_pack

    pack = build_pack()
    assert pack["views"] == 57
    assert pack["reference_renderings"] == 114
    assert pack["unique_stories"] == 12
    assert pack["unique_families"] == 11
    assert pack["rejected"] == []
    ids = {
        row["fixture_id"]: (row["story_id"], row["family_id"])
        for row in pack["examples"]
    }
    assert ids["S3"] == ids["C2"]
    assert ids["S2"] == ids["C3"]
    assert ids["I1"][1] == ids["I2"][1]
    assert ids["I1"][0] != ids["I2"][0]


def test_small_ambiguity_reference_backend_agreement():
    problem = BY_ID["D2"].problem()
    reference = SpatialSolverV2(backend="reference").analyze(problem)
    evidence = build_evidence(problem)
    assert reference.error is None
    assert (
        frozenset(d.value for d in reference.possible_directions)
        == evidence.semantics.possible
    )


@pytest.mark.parametrize("name", ("C1", "I3"))
@pytest.mark.parametrize("quarter_turns", (1, 2, 3))
def test_count_and_conflicts_are_not_north_or_x_axis_specific(name, quarter_turns):
    problem = BY_ID[name].problem()
    directions = tuple(Direction)

    def rotate(direction):
        return directions[(directions.index(direction) + 2 * quarter_turns) % 8]

    query = problem.query
    if not isinstance(query, DirectionQuery):
        query = replace(
            query, directions=frozenset(rotate(d) for d in query.directions)
        )
    rotated = replace(
        problem,
        query=query,
        premise=And(
            tuple(
                replace(atom, allowed=frozenset(rotate(d) for d in atom.allowed))
                for atom in problem.premise.operands
            )
        ),
    )
    evidence = build_evidence(rotated)
    assert evidence.semantics.consistent == BY_ID[name].consistent
    assert evidence.semantics.possible == BY_ID[name].possible
