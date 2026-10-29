"""Production composition interface checks against the independent pivot oracle."""

from dataclasses import replace
from itertools import permutations

import pytest

from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    allowed_answers,
    analyze,
    encode,
    parse_answer,
    parse_prompt,
    render_prompt,
)
from spatial.v2.composition_direction import (
    build_evidence,
    build_example,
    render_pair,
    verify_evidence,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.proofs import ProofCheckError
from spatial.v2.solver import And, Direction, DirectionAnalysis, SpatialSolverV2
from spatial.v2.tests.test_pivot_evidence import BY_ID, CASES, VIEWS

MODES = (
    AnswerMode.SINGLE,
    AnswerMode.VISIBLE_POSSIBLE,
    AnswerMode.ALL_POSSIBLE,
)


@pytest.mark.parametrize("name,menu,expected", VIEWS)
def test_production_mapping_all_fixture_views(name, menu, expected):
    semantics = analyze(BY_ID[name].problem())
    assert tuple(encode(semantics, menu, mode).letters for mode in MODES) == expected
    for permuted in permutations(menu):
        for mode, original in zip(MODES, expected):
            decision = encode(semantics, permuted, mode)
            assert parse_answer(decision.letters, mode) == decision.letters
            if original in {"E", "F"}:
                assert decision.letters == original
            else:
                assert {permuted["ABCD".index(c)] for c in decision.letters} == {
                    menu["ABCD".index(c)] for c in original
                }


@pytest.mark.parametrize("name,menu,expected", VIEWS)
@pytest.mark.parametrize("mode", MODES)
def test_full_prompt_round_trip(name, menu, expected, mode):
    problem = BY_ID[name].problem()
    prompt = render_prompt(problem, menu, mode)
    assert parse_prompt(prompt) == (problem, menu, mode)
    assert (
        sum(
            line.startswith(tuple(f"{c}. " for c in "ABCDEF"))
            for line in prompt.splitlines()
        )
        == 6
    )


@pytest.mark.parametrize("mode", MODES)
def test_answer_grammar(mode):
    answers = allowed_answers(mode)
    assert len(answers) == (6 if mode is AnswerMode.SINGLE else 17)
    for answer in answers:
        assert parse_answer(answer, mode) == answer
        assert parse_answer(f"Reasoning.\nAnswer: {answer}\n", mode, stage=1) == answer
    for invalid in (
        "",
        "BA",
        "AA",
        "AE",
        "EF",
        "AF",
        "A,B",
        "A B",
        "a",
        "G",
        "Answer: A",
    ):
        with pytest.raises(ValueError):
            parse_answer(invalid, mode)
    if mode is AnswerMode.SINGLE:
        with pytest.raises(ValueError):
            parse_answer("AB", mode)
    for invalid in ("A", "Answer: A\nMore text", "Answer: A\nAnswer: B", "Answer: A."):
        with pytest.raises(ValueError):
            parse_answer(invalid, mode, stage=1)


@pytest.mark.parametrize(
    "old,new",
    (
        ("Contract: single", "Contract: all-possible"),
        ("None of the above", "Cannot be determined"),
        ("no valid map exists", "a valid map exists"),
        ("in every valid map?", "in some valid map?"),
        ("Objects: X, Y, Z", "Objects: X, Y"),
    ),
)
def test_template_corruption_rejected(old, new):
    prompt = render_prompt(BY_ID["D1"].problem(), VIEWS[0][1], AnswerMode.SINGLE)
    assert old in prompt
    with pytest.raises(ValueError):
        parse_prompt(prompt.replace(old, new))


@pytest.mark.parametrize(
    "menu",
    (
        ("North", "East", "South"),
        ("North", "East", "South", "South"),
        ("North", "East", "South", "Nowhere"),
    ),
)
def test_invalid_menus_rejected(menu):
    with pytest.raises(ValueError, match="four distinct"):
        encode(analyze(BY_ID["D1"].problem()), menu, AnswerMode.SINGLE)


def test_count_bool_is_not_integer_menu_value():
    with pytest.raises(ValueError):
        encode(analyze(BY_ID["C3"].problem()), (False, 1, 2, 3), AnswerMode.SINGLE)


def test_menu_does_not_shrink_candidate_domain():
    problem = BY_ID["S6"].problem()
    menu = ("U", "V", "W", "T")
    restored, _, _ = parse_prompt(render_prompt(problem, menu, AnswerMode.SINGLE))
    assert restored.query.candidates == ("U", "V", "W", "T", "Y")
    assert (
        encode(analyze(restored), menu, AnswerMode.SINGLE).escape_reason
        == "unique-unlisted"
    )


def test_restricted_direction_domain_rejected():
    problem = BY_ID["D1"].problem()
    problem = replace(
        problem,
        query=replace(problem.query, candidate_directions=frozenset({Direction.NORTH})),
    )
    with pytest.raises(ValueError, match="eight directions"):
        analyze(problem)


@pytest.mark.parametrize("error", ("timeout", "unknown", "solver failure"))
def test_solver_errors_never_become_escape_labels(monkeypatch, error):
    monkeypatch.setattr(
        SpatialSolverV2,
        "analyze",
        lambda *args, **kwargs: DirectionAnalysis(False, error=error),
    )
    with pytest.raises(VerificationError, match="inconclusive"):
        build_evidence(BY_ID["D1"].problem())


def test_raised_solver_failure_rejected(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("timeout")

    monkeypatch.setattr(SpatialSolverV2, "analyze", fail)
    with pytest.raises(VerificationError, match="solver failure"):
        analyze(BY_ID["D1"].problem())


@pytest.mark.parametrize("corruption", ("missing", "coincident", "wrong-answer"))
def test_witness_corruption_rejected(monkeypatch, corruption):
    problem = BY_ID["D1"].problem()
    result = SpatialSolverV2(backend="z3").analyze(problem)
    if corruption == "missing":
        result = replace(result, witnesses={})
    else:
        coordinates = (
            {"X": (0, 0), "Y": (0, 0), "Z": (0, 0)}
            if corruption == "coincident"
            else {"X": (-1, 1), "Y": (0, 0), "Z": (0, -1)}
        )
        result = replace(result, witnesses={Direction.NORTHEAST: coordinates})
    monkeypatch.setattr(SpatialSolverV2, "analyze", lambda *args, **kwargs: result)
    with pytest.raises(VerificationError):
        analyze(problem)


@pytest.mark.parametrize("name", ("D1", "I2"))
@pytest.mark.parametrize("mode", MODES)
def test_checked_direction_pair(name, mode):
    problem = BY_ID[name].problem()
    evidence = build_evidence(problem)
    menu = ("West", "Northeast", "North", "South")
    pair = render_pair(evidence, menu, mode)
    decision = encode(evidence.semantics, menu, mode)
    for text in pair.values():
        assert parse_answer(text, mode, stage=1) == decision.letters
        for step in evidence.proof.steps:
            assert f"{step.id}:" in text
    assert pair["natural"] != pair["symbolic"]
    example = build_example(problem, menu, mode, story_id=name, family_id=name)
    assert example["status"] == "feasibility-only-not-a-dataset-release"
    assert example["audit"]["witnesses"]
    assert parse_prompt(example["prompt"])[0] == problem


@pytest.mark.parametrize("mode", MODES)
def test_missing_menu_direction_pair(mode):
    evidence = build_evidence(BY_ID["D1"].problem())
    pair = render_pair(evidence, ("North", "East", "South", "West"), mode)
    for text in pair.values():
        assert parse_answer(text, mode, stage=1) == "F"


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.name not in {"D1", "I2"}], ids=lambda c: c.name
)
def test_unsupported_evidence_is_not_released(case):
    with pytest.raises(UnsupportedEvidence):
        build_evidence(case.problem())


def test_proof_and_semantic_tampering_rejected():
    evidence = build_evidence(BY_ID["D1"].problem())
    proof = evidence.proof
    with pytest.raises(ProofCheckError):
        verify_evidence(replace(evidence, proof=replace(proof, steps=proof.steps[:-1])))
    with pytest.raises(VerificationError):
        verify_evidence(
            replace(
                evidence,
                semantics=replace(evidence.semantics, possible=frozenset({"North"})),
            )
        )
    with pytest.raises(VerificationError, match="stored witnesses"):
        verify_evidence(
            replace(evidence, semantics=replace(evidence.semantics, witnesses={}))
        )


def test_direction_rename_and_order_variant():
    problem = BY_ID["D1"].problem()
    names = {obj: f"Object{i}" for i, obj in enumerate(problem.objects)}
    variant = replace(
        problem,
        objects=tuple(names[obj] for obj in problem.objects),
        premise=And(
            tuple(
                replace(
                    atom, subject=names[atom.subject], reference=names[atom.reference]
                )
                for atom in reversed(problem.premise.operands)
            )
        ),
        query=replace(problem.query, target=names["X"], reference=names["Z"]),
    )
    original, changed = build_evidence(problem), build_evidence(variant)
    assert original.semantics.possible == changed.semantics.possible
    verify_evidence(changed)


def test_direction_reference_backend_agreement():
    problem = BY_ID["D1"].problem()
    reference = analyze(problem, solver=SpatialSolverV2(backend="reference"))
    assert reference.possible == build_evidence(problem).semantics.possible


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("links", (1, 2, 4))
def test_direction_path_is_not_hardcoded_to_d1(direction, links):
    template = BY_ID["D1"].problem()
    objects = tuple(f"Point{i}" for i in range(links + 1))
    problem = replace(
        template,
        objects=objects,
        premise=And(
            tuple(
                replace(
                    template.premise.operands[0],
                    subject=objects[i],
                    reference=objects[i + 1],
                    allowed=frozenset({direction}),
                )
                for i in range(links)
            )
        ),
        query=replace(template.query, target=objects[0], reference=objects[-1]),
    )
    evidence = build_evidence(problem)
    assert evidence.semantics.possible == frozenset({direction.value})
    menu = (direction.value, *(d.value for d in Direction if d != direction))[:4]
    for text in render_pair(evidence, menu, AnswerMode.SINGLE).values():
        assert parse_answer(text, AnswerMode.SINGLE, stage=1) == "A"
