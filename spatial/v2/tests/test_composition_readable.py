"""Readable preview tests inspect actual wording as well as shared evidence."""

import re
from copy import deepcopy
from dataclasses import replace

import pytest

from spatial.v2.composition_contracts import VerificationError
from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_readable import (
    VERSION,
    _assumed_direction_conflict,
    _opposing_path,
    _path,
    build_outline,
    check_pair,
    render_outline,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    SpatialProblem,
    SpatialSolverV2,
    direction_constraint,
)
from spatial.v2.tests.test_pivot_evidence import BY_ID, VIEWS


def test_mixed_direction_actual_relationships_in_both_styles():
    evidence = build_evidence(BY_ID["D1"].problem())
    menu = ("West", "Northeast", "South", "Northwest")
    outline = build_outline(evidence, menu, AnswerMode.SINGLE)
    pair = render_outline(outline)
    symbolic, natural = pair["symbolic"], pair["natural"]
    assert "x(X) >" not in symbolic  # Paths are oriented in increasing order.
    assert "=> NE(X,Z)" in symbolic
    assert "place X exactly northeast of Z" in natural
    assert "horizontal position" in natural and "bottom to top" in natural
    assert "x(Z) = x(Y) < x(X) => x(Z) < x(X)" in symbolic
    assert "y(Z) < y(Y) < y(X) => y(Z) < y(X)" in symbolic
    assert "The only possible value is Northeast." in natural
    assert "Answer: B" in natural and "Answer: B" in symbolic
    assert "No displayed" not in natural
    check_pair(evidence, menu, AnswerMode.SINGLE, pair)


@pytest.mark.parametrize("direction", tuple(Direction))
def test_eight_directions_ten_links_no_bookkeeping_or_missing_links(direction):
    names = tuple(f"Object{i}" for i in range(11))
    problem = SpatialProblem(
        names,
        And(
            tuple(
                direction_constraint(names[i + 1], names[i], direction)
                for i in range(10)
            )
        ),
        DirectionQuery(names[-1], names[0]),
    )
    evidence = build_evidence(problem)
    snapshot = tagged_dataclass_to_dict(evidence)
    menu = (direction.value, *(d.value for d in Direction if d != direction))[:4]
    outline = build_outline(evidence, menu, AnswerMode.ALL_POSSIBLE)
    paths = [u for u in outline["units"] if u["kind"] == "path"]
    assert len(paths) == 2
    for path in paths:
        assert len(path["indices"]) == 10
        assert set(path["nodes"]) == set(names)
    pair = render_outline(outline)
    for text in pair.values():
        assert all(name in text for name in names)
        assert all(
            label not in text
            for label in ("PATH-", "D-X-", "Q-DIR", "P1-X", "AXIS_INVERSION")
        )
        assert text.endswith("Answer: A")
    assert tagged_dataclass_to_dict(evidence) == snapshot


@pytest.mark.parametrize("name,menu,expected", VIEWS)
@pytest.mark.parametrize("mode_index", range(3))
def test_fixture_gold_for_every_contract(name, menu, expected, mode_index):
    evidence = build_evidence(BY_ID[name].problem())
    mode = (AnswerMode.SINGLE, AnswerMode.VISIBLE_POSSIBLE, AnswerMode.ALL_POSSIBLE)[
        mode_index
    ]
    pair = render_outline(build_outline(evidence, menu, mode))
    assert all(t.endswith(f"Answer: {expected[mode_index]}") for t in pair.values())
    check_pair(evidence, menu, mode, pair)


@pytest.mark.parametrize(
    "mode,menu,excluded,gold",
    (
        (
            AnswerMode.VISIBLE_POSSIBLE,
            ("West", "East", "South", "Northeast"),
            {"South", "Northeast"},
            "AB",
        ),
        (
            AnswerMode.ALL_POSSIBLE,
            ("West", "East", "South", "Northeast"),
            {"North", "Northeast", "Southeast", "South", "Southwest", "Northwest"},
            "AB",
        ),
        (
            AnswerMode.VISIBLE_POSSIBLE,
            ("North", "South", "Northeast", "Southwest"),
            {"North", "South", "Northeast", "Southwest"},
            "F",
        ),
    ),
)
def test_ambiguous_direction_explains_exact_contract_domain(mode, menu, excluded, gold):
    evidence = build_evidence(BY_ID["D2"].problem())
    outline = build_outline(evidence, menu, mode)
    units = outline["units"]
    shown = {
        value
        for unit in units
        if unit["kind"] == "excluded-directions"
        for value in unit["values"]
    }
    assert shown == excluded
    assert len([unit for unit in units if unit["kind"] == "path"]) == 1
    assert {unit["outcome"] for unit in units if unit["kind"] == "scenario"} == (
        {"East", "West"} & set(menu)
    )
    pair = render_outline(outline)
    assert "y(X) = y(O) = y(Y)" in pair["symbolic"]
    assert all(text.endswith(f"Answer: {gold}") for text in pair.values())
    check_pair(evidence, menu, mode, pair)


def test_unconstrained_pair_all_four_listed_values_have_witnesses():
    problem = SpatialProblem(
        ("X", "Y", "O"),
        And((direction_constraint("X", "O", Direction.EAST),)),
        DirectionQuery("X", "Y"),
    )
    evidence = build_evidence(problem)
    menu = ("North", "East", "South", "West")
    outline = build_outline(evidence, menu, AnswerMode.VISIBLE_POSSIBLE)
    assert {u["outcome"] for u in outline["units"] if u["kind"] == "scenario"} == set(
        menu
    )
    assert all(
        text.endswith("Answer: ABCD") for text in render_outline(outline).values()
    )
    incomplete = build_outline(evidence, menu, AnswerMode.ALL_POSSIBLE)
    assert len([u for u in incomplete["units"] if u["kind"] == "scenario"]) == 1
    assert all(
        text.endswith("Answer: F") for text in render_outline(incomplete).values()
    )


def test_assumed_direction_exposes_indirect_forced_coincidence():
    problem = SpatialProblem(
        ("X", "Y", "O", "T"),
        And(
            (
                direction_constraint("X", "O", Direction.NORTH),
                direction_constraint("Y", "O", Direction.NORTH),
                direction_constraint("X", "T", Direction.EAST),
            )
        ),
        DirectionQuery("Y", "T"),
    )
    assert _opposing_path(problem, "Y", "T", Direction.EAST) is None
    units = _assumed_direction_conflict(
        problem, "Y", "T", Direction.EAST, SpatialSolverV2(backend="z3")
    )
    pair = render_outline({"version": VERSION, "units": units})
    assert "temporary assumption 1" in pair["natural"]
    assert "no-coincidence rule" in pair["natural"]
    assert "Assume E(Y,T)" in pair["symbolic"]
    assert "x(X)=x(Y) and y(X)=y(Y)" in pair["symbolic"]


@pytest.mark.parametrize("index", (-1, 2))
def test_out_of_range_path_citations_are_rejected(index):
    with pytest.raises(VerificationError, match="premise index"):
        _path(BY_ID["D1"].problem(), "x", ("X", "Y"), (index,))


def test_joint_count_exposes_conflict_not_marginal_counting():
    evidence = build_evidence(BY_ID["C1"].problem())
    outline = build_outline(evidence, (0, 1, 2, 3), AnswerMode.ALL_POSSIBLE)
    pair = render_outline(outline)
    assert "U and V cannot qualify together" in pair["natural"]
    assert "temporary assumptions" in pair["natural"]
    assert (
        "both coordinates equal" in pair["natural"]
        or "smaller than itself" in pair["natural"]
    )
    assert "Counts 2, 3 and 4 are therefore impossible" in pair["natural"]
    assert "No object is required to qualify." in pair["natural"]
    assert "must include none" not in pair["natural"]
    assert "sizes {0, 1}" in pair["symbolic"]
    assert pair["natural"].endswith("Answer: AB")
    models = [u for u in outline["units"] if u["kind"] == "scenario"]
    assert sorted(u["outcome"] for u in models) == [0, 1]
    assert all(len(u["points"]) == 5 for u in outline["audit_witnesses"])
    assert all(u["outcome"] == len(u["members"]) for u in models)


@pytest.mark.parametrize("name", ("I1", "I3"))
def test_conflicts_show_actual_axiom_and_return_e(name):
    evidence = build_evidence(BY_ID[name].problem())
    pair = render_outline(
        build_outline(evidence, ("North", "East", "South", "West"), AnswerMode.SINGLE)
    )
    assert (
        "smaller than itself" in pair["natural"]
        if name == "I1"
        else "no-coincidence rule" in pair["natural"]
    )
    assert all(text.endswith("Answer: E") for text in pair.values())


@pytest.mark.parametrize("style", ("natural", "symbolic"))
@pytest.mark.parametrize("mutation", ("relation", "object", "citation", "answer"))
def test_mutated_rendered_outputs_are_rejected(style, mutation):
    evidence = build_evidence(BY_ID["D1"].problem())
    menu = ("West", "Northeast", "South", "Northwest")
    pair = render_outline(build_outline(evidence, menu, AnswerMode.SINGLE))
    before = pair[style]
    if mutation == "relation":
        pair[style] = (
            before.replace("to the right of", "to the left of")
            if style == "natural"
            else before.replace("<", ">")
        )
    elif mutation == "object":
        pair[style] = before.replace("X", "WrongObject")
    elif mutation == "citation":
        pair[style] = (
            before.replace("sources: 2", "sources: 99")
            if style == "natural"
            else before.replace("p1", "p99")
        )
    else:
        pair[style] = before.replace("Answer: B", "Answer: A")
    assert pair[style] != before
    with pytest.raises(VerificationError):
        check_pair(evidence, menu, AnswerMode.SINGLE, pair)


def test_invalid_citation_cannot_create_a_displayed_path():
    problem = BY_ID["D1"].problem()
    with pytest.raises(VerificationError):
        _path(problem, "x", ("X", "Z"), (0,))


def test_changed_semantics_cannot_be_hidden_by_readable_output():
    evidence = build_evidence(BY_ID["D1"].problem())
    corrupted = replace(
        evidence, semantics=replace(evidence.semantics, possible=frozenset({"South"}))
    )
    with pytest.raises(VerificationError):
        build_outline(corrupted, ("North", "East", "South", "West"), AnswerMode.SINGLE)


def test_outline_rendering_does_not_mutate_selected_evidence():
    outline = build_outline(
        build_evidence(BY_ID["C1"].problem()), (0, 1, 2, 3), AnswerMode.ALL_POSSIBLE
    )
    original = deepcopy(outline)
    render_outline(outline)
    assert outline == original


def test_single_links_are_not_repeated_and_count_grammar_is_singular():
    pair = render_outline(
        build_outline(
            build_evidence(BY_ID["C2"].problem()),
            (0, 1, 2, 3),
            AnswerMode.SINGLE,
        )
    )
    assert "[p1] y(O) < y(U)." in pair["symbolic"]
    assert "y(O) < y(U) => y(O) < y(U)" not in pair["symbolic"]
    assert "U is above O (premise sources: 1)." in pair["natural"]
    assert "leaves only size 2." in pair["natural"]


@pytest.mark.parametrize(
    "name,menu,mode",
    (
        ("D2", ("West", "East", "South", "Northeast"), AnswerMode.SINGLE),
        ("S4", ("U", "V", "W", "T"), AnswerMode.SINGLE),
        ("C1", (0, 1, 2, 3), AnswerMode.ALL_POSSIBLE),
    ),
)
def test_witness_coordinates_are_audit_only_and_scenarios_are_qualitative(
    name, menu, mode
):
    evidence = build_evidence(BY_ID[name].problem())
    outline = build_outline(evidence, menu, mode)
    pair = render_outline(outline)
    models = [unit for unit in outline["units"] if unit["kind"] == "scenario"]
    assert models
    assert pair["natural"].count("One possibility consistent with all premises") == len(
        models
    )
    assert pair["symbolic"].count("(possible together under all premises)") == len(
        models
    )
    assert outline["audit_witnesses"]
    assert all("points" not in unit for unit in outline["units"])
    for text in pair.values():
        assert re.search(r"\(-?\d+, ?-?\d+\)", text) is None
        assert "map 1" not in text and "map 2" not in text
    check_pair(evidence, menu, mode, pair)


def test_ambiguous_direction_uses_shared_anchor_and_qualitative_orders():
    outline = build_outline(
        build_evidence(BY_ID["D2"].problem()),
        ("West", "East", "South", "Northeast"),
        AnswerMode.SINGLE,
    )
    pair = render_outline(outline)
    assert "From left to right, the order is O, Y and X." in pair["natural"]
    assert "From left to right, the order is O, X and Y." in pair["natural"]
    assert "O, X and Y share the same height." in pair["natural"]
    assert "x(O) < x(Y) < x(X)" in pair["symbolic"]
    assert "x(O) < x(X) < x(Y)" in pair["symbolic"]


def test_single_rejected_count_grammar():
    outline = build_outline(
        build_evidence(BY_ID["C1"].problem()), (0, 1, 2, 3), AnswerMode.ALL_POSSIBLE
    )
    unit = next(u for u in outline["units"] if u["kind"] == "count-constraints")
    unit["rejected"] = [7]
    pair = render_outline(outline)
    assert "Count 7 is therefore impossible." in pair["natural"]


def test_qualitatively_identical_completions_share_a_case():
    from spatial.v2.composition_readable import _model
    from spatial.v2.composition_scenarios import qualitative_outline

    problem = BY_ID["S4"].problem()
    evidence = build_evidence(problem)
    original = build_outline(evidence, ("U", "V", "W", "T"), AnswerMode.SINGLE)
    points = original["audit_witnesses"][0]["points"]
    coordinates = {obj: (x, y) for obj, x, y in points}
    # Translation changes every coordinate without changing spatial alternatives.
    translated = {obj: (x + 10, y - 4) for obj, (x, y) in coordinates.items()}
    outline = qualitative_outline(
        [_model(problem, coordinates), _model(problem, translated)], problem, VERSION
    )
    assert len(outline["units"]) == 1
    assert len(outline["audit_witnesses"]) == 2
    pair = render_outline(outline)
    assert pair["natural"].count("Case 1:") == 1
    assert "Case 2:" not in pair["natural"]
