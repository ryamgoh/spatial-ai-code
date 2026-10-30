import random
from copy import deepcopy

import pytest

from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_readable import build_outline, render_outline
from spatial.v2.composition_readable_audit import check_row
from spatial.v2.composition_selection_expansion import make_story
from spatial.v2.composition_shared_paths import factor_shared_paths
from spatial.v2.grading import AnswerMode
from spatial.v2.serialization import tagged_dataclass_to_dict


@pytest.fixture(scope="module")
def shared():
    story = make_story("shared-exclusion", 10, 0, random.Random(5))
    problem = story.problems()[1]
    outline = build_outline(
        build_evidence(problem), story.candidates[:4], AnswerMode.SINGLE
    )
    return {
        "explanation_outline": outline,
        "references": render_outline(outline),
    }, tagged_dataclass_to_dict(problem)


def test_shared_paths_keep_full_evidence_and_shorten_both_styles(shared):
    row, problem = shared
    outline = row["explanation_outline"]
    assert any(u.get("comparison") for u in outline["units"])
    assert any(u.get("shared_segments") for u in outline["units"])
    original = deepcopy(outline)
    original["units"] = [u for u in original["units"] if "comparison" not in u]
    for unit in original["units"]:
        unit.pop("shared_segments", None)
    old = render_outline(original)
    for style, text in row["references"].items():
        assert len(text) < len(old[style])
        assert text.count("Comparison 1:") == 1
        assert text.endswith(old[style].splitlines()[-1])
    check_row(row, problem)
    assert max(len(u["indices"]) for u in outline["units"] if u["kind"] == "path") == 10


@pytest.mark.parametrize("style", ("natural", "symbolic", "both"))
@pytest.mark.parametrize("mutation", ("reference", "definition", "forward", "sign"))
def test_shared_reference_mutations_fail_independent_audit(shared, style, mutation):
    row, problem = deepcopy(shared)
    for name in ("natural", "symbolic") if style == "both" else (style,):
        before = row["references"][name]
        if mutation == "reference":
            after = before.replace("comparison 1", "comparison 99")
        elif mutation == "definition":
            after = before.replace("Comparison 1:", "Comparison 99:")
        elif mutation == "forward":
            paragraphs = before.split("\n\n")
            index = next(
                i for i, p in enumerate(paragraphs) if p.startswith("Comparison 1:")
            )
            definition = paragraphs.pop(index)
            paragraphs.append(definition)
            after = "\n\n".join(paragraphs)
        elif name == "symbolic":
            after = before.replace(" < ", " = ")
        else:
            after = before.replace("above", "below").replace(
                "to the right of", "to the left of"
            )
        assert after != before
        row["references"][name] = after
    with pytest.raises(ValueError):
        check_row(row, problem)


def test_assumption_scopes_are_never_factored(shared):
    row, _ = deepcopy(shared)
    outline = row["explanation_outline"]
    outline["units"] = [u for u in outline["units"] if "comparison" not in u]
    for unit in outline["units"]:
        unit.pop("shared_segments", None)
    outline["units"].insert(0, {"kind": "assume-direction"})
    original = deepcopy(outline)
    assert factor_shared_paths(outline) == original
    assert outline == original


def test_reverse_equal_trunks_share_one_checked_comparison():
    from spatial.v2.composition_readable import VERSION

    atoms = [
        {"subject": a, "reference": b, "allowed": ["North"]}
        for a, b in (("A", "B"), ("B", "C"), ("C", "D"))
    ]
    paths = [
        {
            "kind": "path",
            "axis": "x",
            "nodes": nodes,
            "signs": [0, 0, 0],
            "indices": indices,
            "original_count": 3,
        }
        for nodes, indices in (
            (["A", "B", "C", "D"], [0, 1, 2]),
            (["D", "C", "B", "A"], [2, 1, 0]),
        )
    ]
    outline = {"version": VERSION, "units": paths}
    before = deepcopy(outline)
    factored = factor_shared_paths(outline)
    assert outline == before
    assert sum("comparison" in u for u in factored["units"]) == 1
    check_row(
        {"explanation_outline": factored, "references": render_outline(factored)},
        {"premise": {"operands": atoms}},
    )
