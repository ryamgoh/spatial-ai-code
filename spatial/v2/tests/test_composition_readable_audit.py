"""Printed-text audit checks do not trust renderer template regeneration."""

import pytest

from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_readable import build_outline, render_outline
from spatial.v2.composition_readable_audit import check_row
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.tests.test_pivot_evidence import BY_ID, VIEWS


def make_row(name, menu, mode):
    problem = BY_ID[name].problem()
    outline = build_outline(build_evidence(problem), menu, mode)
    return {
        "references": render_outline(outline),
        "explanation_outline": outline,
    }, tagged_dataclass_to_dict(problem)


@pytest.mark.parametrize("name,menu,_expected", VIEWS)
@pytest.mark.parametrize("mode", tuple(AnswerMode))
def test_audit_all_fixture_text(name, menu, _expected, mode):
    row, problem = make_row(name, menu, mode)
    check_row(row, problem)


@pytest.mark.parametrize("style", ("natural", "symbolic"))
@pytest.mark.parametrize("mutation", ("link", "citation", "endpoint", "omission"))
def test_corrupt_printed_chain_rejected(style, mutation):
    row, problem = make_row(
        "D1", ("West", "Northeast", "South", "Northwest"), AnswerMode.SINGLE
    )
    before = row["references"][style]
    if mutation == "link":
        after = before.replace(
            "is to the left of", "has the same horizontal position as"
        ).replace("< x(X)", "= x(X)")
    elif mutation == "citation":
        after = before.replace("sources: 2", "sources: 99").replace("p2", "p99")
    elif mutation == "endpoint":
        after = before.replace("Therefore, X", "Therefore, Y").replace(
            "=> x(Z) < x(X)", "=> x(Z) < x(Y)"
        )
    else:
        paragraphs = before.split("\n\n")
        index = next(
            i for i, p in enumerate(paragraphs) if "sources:" in p or p.startswith("[")
        )
        paragraphs.pop(index)
        after = "\n\n".join(paragraphs)
    assert after != before
    row["references"][style] = after
    with pytest.raises(ValueError):
        check_row(row, problem)


@pytest.mark.parametrize("style", ("natural", "symbolic"))
def test_wrong_printed_witness_outcome_rejected(style):
    row, problem = make_row("C1", (0, 1, 2, 3), AnswerMode.ALL_POSSIBLE)
    before = row["references"][style]
    after = before.replace("count=0.", "count=4.").replace(
        "a count of 0.", "a count of 4."
    )
    assert after != before
    row["references"][style] = after
    with pytest.raises(ValueError, match="qualitative evidence|joint count"):
        check_row(row, problem)


def test_matching_but_wrong_outcomes_in_both_styles_are_rejected():
    row, problem = make_row("C1", (0, 1, 2, 3), AnswerMode.ALL_POSSIBLE)
    row["references"]["natural"] = row["references"]["natural"].replace(
        "a count of 0.", "a count of 4."
    )
    row["references"]["symbolic"] = row["references"]["symbolic"].replace(
        "count=0.", "count=4."
    )
    with pytest.raises(ValueError, match="joint count"):
        check_row(row, problem)


def test_matching_but_invalid_qualitative_orders_in_both_styles_are_rejected():
    row, problem = make_row(
        "D2", ("West", "East", "South", "Northeast"), AnswerMode.SINGLE
    )
    row["references"]["natural"] = row["references"]["natural"].replace(
        "the order is O, Y and X", "the order is X, Y and O"
    )
    row["references"]["symbolic"] = row["references"]["symbolic"].replace(
        "x(O) < x(Y) < x(X)", "x(X) < x(Y) < x(O)"
    )
    with pytest.raises(ValueError, match="audit completion"):
        check_row(row, problem)


def test_audit_only_map_must_satisfy_unshown_premises():
    row, problem = make_row("S4", ("U", "V", "W", "T"), AnswerMode.SINGLE)
    witness = row["explanation_outline"]["audit_witnesses"][0]
    witness["points"][0][1:] = witness["points"][1][1:]
    with pytest.raises(ValueError, match="coincident"):
        check_row(row, problem)


def test_wrong_case_reference_is_rejected():
    row, problem = make_row("S4", ("U", "V", "W", "T"), AnswerMode.SINGLE)
    for style in row["references"]:
        row["references"][style] = row["references"][style].replace(
            "qualifies in Case 1", "qualifies in Case 2"
        )
    with pytest.raises(ValueError, match="contingency"):
        check_row(row, problem)


def test_numerical_coordinates_cannot_leak_back_into_supervision():
    row, problem = make_row("C1", (0, 1, 2, 3), AnswerMode.ALL_POSSIBLE)
    row["references"]["natural"] += "\nU is at (1, 2)."
    with pytest.raises(ValueError, match="numerical"):
        check_row(row, problem)
