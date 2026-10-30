import random
from copy import deepcopy

import pytest

from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_organization import (
    evidence_inventory,
    evidence_units,
    organize_outline,
)
from spatial.v2.composition_readable import build_outline, render_outline
from spatial.v2.composition_readable_audit import check_row
from spatial.v2.composition_selection_expansion import make_story
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict


@pytest.fixture(scope="module")
def example():
    story = make_story("shared-exclusion", 3, 0, random.Random(5))
    problem = story.problems()[1]
    outline = build_outline(
        build_evidence(problem), story.candidates[:4], AnswerMode.SINGLE
    )
    return problem, outline


@pytest.mark.parametrize("pattern", ("match-chain", "rival-chain", "shared-exclusion"))
@pytest.mark.parametrize("depth", (1, 3, 10))
@pytest.mark.parametrize("sample", (0, 1))
def test_organization_preserves_every_deduction_and_gold(pattern, depth, sample):
    story = make_story(pattern, depth, sample, random.Random(30 + sample))
    problem = story.problems()[1]
    outline = build_outline(
        build_evidence(problem), story.candidates[:4], AnswerMode.SINGLE
    )
    old = deepcopy(outline)
    old["version"] = "readable-shared-evidence-v4"
    old["units"] = evidence_units(outline)
    snapshot = deepcopy(old)
    revised = organize_outline(old, problem, AnswerMode.SINGLE)
    assert old == snapshot
    assert evidence_inventory(old) == evidence_inventory(revised)
    assert organize_outline(revised, problem, AnswerMode.SINGLE) == revised
    assert revised["units"][0]["kind"] == "purpose"
    memberships = [u for u in evidence_units(revised) if u["kind"] == "membership"]
    assert memberships[0]["status"] == "entailed"
    before, after = render_outline(old), render_outline(revised)
    for style in after:
        assert after[style].splitlines()[-1] == before[style].splitlines()[-1]
    check_row(
        {"explanation_outline": revised, "references": after},
        tagged_dataclass_to_dict(problem),
    )


@pytest.mark.parametrize("style", ("natural", "symbolic", "both"))
@pytest.mark.parametrize("mutation", ("candidate", "sign", "direction", "omit"))
def test_group_mutations_fail_even_when_both_styles_agree(example, style, mutation):
    problem, outline = example
    row = {
        "explanation_outline": deepcopy(outline),
        "references": render_outline(outline),
    }
    group = next(u for u in outline["units"] if u["kind"] == "membership-group")
    for name in ("natural", "symbolic") if style == "both" else (style,):
        paragraphs = row["references"][name].split("\n\n")
        index = next(
            i
            for i, p in enumerate(paragraphs)
            if p.startswith("For each c in ") or ", so none can be " in p
        )
        before = paragraphs[index]
        if mutation == "candidate":
            paragraphs[index] = before.replace(
                group["memberships"][0]["candidate"], "WrongObject"
            )
        elif mutation == "sign":
            if name == "symbolic":
                op = {-1: "<", 0: "=", 1: ">"}[group["sign"]]
                paragraphs[index] = before.replace(
                    f" {op} ", " = " if op != "=" else " < ", 1
                )
            else:
                relation = {
                    ("x", -1): "to the left of",
                    ("x", 1): "to the right of",
                    ("y", -1): "below",
                    ("y", 1): "above",
                    ("x", 0): "same horizontal position as",
                    ("y", 0): "same height as",
                }[group["axis"], group["sign"]]
                paragraphs[index] = before.replace(relation, "INVALID", 1)
        elif mutation == "direction":
            paragraphs[index] = before.replace("=> not ", "=> ").replace(
                "so none can be", "so all must be"
            )
        else:
            paragraphs.pop(index)
        row["references"][name] = "\n\n".join(paragraphs)
        assert row["references"][name] != render_outline(outline)[name]
    with pytest.raises(ValueError):
        check_row(row, tagged_dataclass_to_dict(problem))


def test_group_cannot_claim_uncited_candidate_support(example):
    problem, original = example
    outline = deepcopy(original)
    group = next(u for u in outline["units"] if u["kind"] == "membership-group")
    candidate = group["memberships"][0]["candidate"]
    outline["units"] = [
        u
        for u in outline["units"]
        if not (u["kind"] == "path" and candidate in (u["nodes"][0], u["nodes"][-1]))
    ]
    with pytest.raises(ValueError, match="unconditional candidate support"):
        check_row(
            {"explanation_outline": outline, "references": render_outline(outline)},
            tagged_dataclass_to_dict(problem),
        )


def test_organizing_current_version_is_idempotent(example):
    problem, outline = example
    assert organize_outline(outline, problem, AnswerMode.SINGLE) == outline


@pytest.mark.parametrize("style", ("natural", "symbolic", "both"))
@pytest.mark.parametrize("mutation", ("citation", "candidate", "relation", "omission"))
def test_grouped_branch_corruption_is_rejected(example, style, mutation):
    problem, outline = example
    row = {"explanation_outline": outline, "references": render_outline(outline)}
    for name in ("natural", "symbolic") if style == "both" else (style,):
        paragraphs = row["references"][name].split("\n\n")
        index = next(
            i
            for i, p in enumerate(paragraphs)
            if p.startswith("Branches ") or "(respective premise sources:" in p
        )
        before = paragraphs[index]
        if mutation == "citation":
            after = before.replace("comparison 1", "comparison 99")
        elif mutation == "candidate":
            after = before.replace("Object", "MissingObject", 1)
        elif mutation == "relation":
            after = (
                before.replace(" < ", " > ").replace(" = ", " > ")
                if name == "symbolic"
                else before.replace("are each", "are never")
            )
        else:
            after = ""
        assert after != before
        paragraphs[index] = after
        row["references"][name] = "\n\n".join(paragraphs)
    with pytest.raises(ValueError):
        check_row(row, tagged_dataclass_to_dict(problem))
