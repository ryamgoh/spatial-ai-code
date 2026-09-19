"""v6 spatial laws: what SpatialSolver must (and must not) return.

The gold is necessity on two linear orders, not "everything consistent."

Run:
    uv run --python 3.12 --no-project --with pytest --with typer \
      pytest spatial/test_spatial_laws.py -q
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

from spatial_solver import SpatialSolver

SOLVER = SpatialSolver()

COMPOUNDS = ("Northeast", "Northwest", "Southeast", "Southwest")
CARDINALS = ("North", "South", "East", "West")
E = "Cannot be determined"

DIR_OPTIONS = {
    "A": "Northeast",
    "B": "Northwest",
    "C": "Southeast",
    "D": "Southwest",
    "E": E,
}


def _map_prompt(sentences: list[str], question: str, options: dict[str, str]) -> str:
    opts = ", ".join(f"{k}. {v}" for k, v in options.items())
    return (
        "Consider a map with multiple locations:\n\n"
        + " ".join(sentences)
        + f"\n\nQuestion: {question} Available options: {opts}"
    )


def gold(text: str) -> str:
    return SOLVER.solve(text)


def letters(text: str) -> set[str]:
    result = gold(text)
    assert not result.startswith("Error"), result
    if result == "No valid options found":
        return set()
    return set(result.split(","))


# ---------------------------------------------------------------------------
# Type 0 — both axes proven (unique compound)
# ---------------------------------------------------------------------------

def test_type0_both_axes_unique_northeast():
    """A is NE of B ⇒ only Northeast. Not North, not E, not the other compounds."""
    sents = ["The Church is to the Northeast of the Pharmacy."]
    text = _map_prompt(
        sents,
        "In which direction is the Church relative to the Pharmacy?",
        {"A": "North", "B": "Northeast", "C": "Southwest", "D": "Northwest", "E": E},
    )
    assert gold(text) == "B"


def test_type0_converse_is_southwest():
    """NE(A,B) is the converse of SW(B,A)."""
    sents = ["The Church is to the Northeast of the Pharmacy."]
    text = _map_prompt(
        sents,
        "In which direction is the Pharmacy relative to the Church?",
        DIR_OPTIONS,
    )
    # DIR_OPTIONS D = Southwest
    assert gold(text) == "D"


def test_type0_transitivity_ne_then_ne():
    """C NE of A, A NE of B ⇒ C NE of B (both axes compose)."""
    sents = [
        "The Church is to the Northeast of the Pharmacy.",
        "The Zoo is to the Northeast of the Church.",
    ]
    text = _map_prompt(
        sents,
        "In which direction is the Zoo relative to the Pharmacy?",
        DIR_OPTIONS,
    )
    assert gold(text) == "A"  # Northeast


def test_counterexample_type0_cardinal_is_not_gold_when_compound_proven():
    """Y is North and X is East. 'North' is true as a half, but Type 0 gold is the compound."""
    sents = ["The Church is to the Northeast of the Pharmacy."]
    text = _map_prompt(
        sents,
        "In which direction is the Church relative to the Pharmacy?",
        {"A": "North", "B": "East", "C": "Northeast", "D": "South", "E": E},
    )
    assert gold(text) == "C"
    assert "A" not in letters(text)
    assert "B" not in letters(text)


def test_counterexample_type0_e_out_when_both_axes_known():
    sents = ["The Church is to the Northeast of the Pharmacy."]
    text = _map_prompt(
        sents,
        "In which direction is the Church relative to the Pharmacy?",
        DIR_OPTIONS,
    )
    assert "E" not in letters(text)


# ---------------------------------------------------------------------------
# Type 0 — one axis proven (disjunction, not E)
# ---------------------------------------------------------------------------

def _one_axis_south_of_museum():
    """Park vs Gas Station: North of GS is proven (via Museum), East/West is not.

    Museum < Park on X, Museum < Gas Station on X (Park vs GS unordered on X).
    Y: Gas Station < Museum < Park (Park is North of GS).
    """
    return [
        "The Park is to the Northeast of the Museum.",
        "The Gas Station is to the Southeast of the Museum.",
    ]


def test_type0_one_axis_both_remaining_compounds():
    text = _map_prompt(
        _one_axis_south_of_museum(),
        "In which direction is the Park relative to the Gas Station?",
        DIR_OPTIONS,
    )
    # North proven, East/West unknown → NE and NW
    assert letters(text) == {"A", "B"}


def test_type0_truncated_disjunction_is_cannot_determine():
    """South proven; SE listed, SW omitted → E, not unique Southeast."""
    text = _map_prompt(
        _one_axis_south_of_museum(),
        "In which direction is the Gas Station relative to the Park?",
        {
            "A": "Southeast",
            "B": "Northeast",
            "C": "Northwest",
            "D": "North",
            "E": E,
        },
    )
    assert gold(text) == "E"
    assert "A" not in letters(text)


def test_counterexample_type0_one_axis_is_not_e():
    """South/North was derived, so 'Cannot be determined' is wrong."""
    text = _map_prompt(
        _one_axis_south_of_museum(),
        "In which direction is the Park relative to the Gas Station?",
        DIR_OPTIONS,
    )
    assert "E" not in letters(text)


def test_counterexample_type0_one_axis_does_not_pick_the_south_compounds():
    text = _map_prompt(
        _one_axis_south_of_museum(),
        "In which direction is the Park relative to the Gas Station?",
        DIR_OPTIONS,
    )
    assert "C" not in letters(text)  # Southeast needs South
    assert "D" not in letters(text)  # Southwest needs South


# ---------------------------------------------------------------------------
# Type 0 — neither axis (E, never A,B,C,D)
# ---------------------------------------------------------------------------

def _disconnected_under_church():
    """Hospital and Zoo both relate to Church, not to each other."""
    return [
        "The Church is to the Northwest of the Zoo.",
        "The Church is to the Northwest of the Hospital.",
        "The Museum is to the Southeast of the Church.",
    ]


def test_type0_both_unknown_is_e():
    text = _map_prompt(
        _disconnected_under_church(),
        "In which direction is the Hospital relative to the Zoo?",
        DIR_OPTIONS,
    )
    assert gold(text) == "E"


def test_counterexample_type0_both_unknown_is_not_all_four_compounds():
    """The original 4-ans bug: treating 'all four are possible' as gold."""
    text = _map_prompt(
        _disconnected_under_church(),
        "In which direction is the Hospital relative to the Zoo?",
        DIR_OPTIONS,
    )
    assert letters(text) == {"E"}
    assert letters(text).isdisjoint({"A", "B", "C", "D"})


def test_counterexample_type0_same_side_of_anchor_does_not_order_the_pair():
    """Both west of Zoo does not prove Pharmacy vs Fire Department."""
    sents = [
        "The Zoo is to the Northeast of the Pharmacy.",
        "The Zoo is to the Northeast of the Fire Department.",
    ]
    text = _map_prompt(
        sents,
        "In which direction is the Pharmacy relative to the Fire Department?",
        DIR_OPTIONS,
    )
    assert gold(text) == "E"


def test_type0_no_e_option_and_unknown_is_empty():
    """Original SpatialMap has no E: unknown dir → no valid option, not ABCD."""
    text = _map_prompt(
        _disconnected_under_church(),
        "In which direction is the Hospital relative to the Zoo?",
        {"A": "Northeast", "B": "Northwest", "C": "Southeast", "D": "Southwest"},
    )
    assert gold(text) == "No valid options found"


# ---------------------------------------------------------------------------
# Cycles — contradiction, not a unique order
# ---------------------------------------------------------------------------

def test_type0_two_cycle_is_e():
    """A NE of B and B NE of A: both axes cycle. Cannot prove a direction."""
    sents = [
        "The Church is to the Northeast of the Pharmacy.",
        "The Pharmacy is to the Northeast of the Church.",
    ]
    text = _map_prompt(
        sents,
        "In which direction is the Church relative to the Pharmacy?",
        DIR_OPTIONS,
    )
    assert gold(text) == "E"


def test_type0_three_cycle_is_e():
    sents = [
        "The Park is to the Northeast of the Museum.",
        "The Museum is to the Northeast of the Zoo.",
        "The Zoo is to the Northeast of the Park.",
    ]
    text = _map_prompt(
        sents,
        "In which direction is the Park relative to the Zoo?",
        DIR_OPTIONS,
    )
    assert gold(text) == "E"


def test_counterexample_cycle_is_not_the_first_edge():
    """Naive closure would see Church>Pharmacy and return Northeast. Forbidden."""
    sents = [
        "The Church is to the Northeast of the Pharmacy.",
        "The Pharmacy is to the Northeast of the Church.",
    ]
    text = _map_prompt(
        sents,
        "In which direction is the Church relative to the Pharmacy?",
        DIR_OPTIONS,
    )
    assert gold(text) != "A"
    assert letters(text).isdisjoint({"A", "B", "C", "D"})


def test_cycle_on_x_only_leaves_the_consistent_axis():
    """A NE of B plus B SE of A: X cycles, Y agrees (A is North of B).

    Then North is proven and East/West is not → NE or NW, not E, not a unique SE/NE.
    """
    sents = [
        "The Church is to the Northeast of the Pharmacy.",
        "The Pharmacy is to the Southeast of the Church.",
    ]
    text = _map_prompt(
        sents,
        "In which direction is the Church relative to the Pharmacy?",
        DIR_OPTIONS,
    )
    # Y: both sentences put Church north of Pharmacy.
    # X: Church east of Pharmacy AND Pharmacy east of Church.
    assert letters(text) == {"A", "B"}
    assert "E" not in letters(text)


def test_type1_cycle_does_not_count_as_definite():
    """Cyclic pair is not 'definitely Northeast' of each other."""
    sents = [
        "The Church is to the Northeast of the Pharmacy.",
        "The Pharmacy is to the Northeast of the Church.",
    ]
    text = _map_prompt(
        sents,
        "Which object is in the Northeast of the Pharmacy?",
        {"A": "Church", "B": "Pharmacy", "C": "Church", "D": "Pharmacy", "E": E},
    )
    assert "A" not in letters(text)
    assert gold(text) == "E"


def test_type2_cycle_counts_zero_definite():
    sents = [
        "The Church is to the Northeast of the Pharmacy.",
        "The Pharmacy is to the Northeast of the Church.",
    ]
    text = _map_prompt(
        sents,
        "How many objects are in the Northeast of the Pharmacy?",
        {"A": "2", "B": "1", "C": "0", "D": "3"},
    )
    assert gold(text) == "C"


# ---------------------------------------------------------------------------
# Type 1 — proven entities only
# ---------------------------------------------------------------------------

def test_type0_listed_answer_fifth_slot_is_only_a_distractor():
    sents = ["The Church is to the Northeast of the Pharmacy."]
    for fifth in ("Cannot be determined", "None of the Options"):
        text = _map_prompt(
            sents,
            "In which direction is the Church relative to the Pharmacy?",
            {
                "A": "Northeast",
                "B": "Northwest",
                "C": "Southeast",
                "D": "Southwest",
                "E": fifth,
            },
        )
        assert gold(text) == "A"
        assert "E" not in letters(text)


def test_type0_two_ans_both_specials_out_when_pair_listed():
    """South proven, SE and SW both listed → those two. Not cannot-determine, not none-of-options."""
    text = _map_prompt(
        _one_axis_south_of_museum(),
        "In which direction is the Gas Station relative to the Park?",
        {
            "A": "Southeast",
            "B": "Southwest",
            "C": "Northeast",
            "D": "North",
            "E": E,
            "F": "None of the Options",
        },
    )
    assert letters(text) == {"A", "B"}
    assert "E" not in letters(text)
    assert "F" not in letters(text)


def test_type0_unique_not_listed_is_none_of_the_options_not_undetermined():
    sents = ["The Church is to the Northeast of the Pharmacy."]
    text = _map_prompt(
        sents,
        "In which direction is the Church relative to the Pharmacy?",
        {
            "A": "Northwest",
            "B": "Southeast",
            "C": "Southwest",
            "D": "South",
            "E": E,
            "F": "None of the Options",
        },
    )
    assert gold(text) == "F"
    assert "E" not in letters(text)


def test_type1_four_proven_entities():
    sents = [
        "The Post Office is to the Southeast of the City Hall.",
        "The Gas Station is to the Southeast of the City Hall.",
        "The Bank is to the Southeast of the City Hall.",
        "The Library is to the Southeast of the City Hall.",
    ]
    text = _map_prompt(
        sents,
        "Which object is in the Southeast of the City Hall?",
        {
            "A": "Post Office",
            "B": "Gas Station",
            "C": "Bank",
            "D": "Library",
            "E": "None of the Options",
        },
    )
    assert letters(text) == {"A", "B", "C", "D"}
    assert "E" not in letters(text)


def test_type1_three_proven_entities():
    sents = [
        "The Post Office is to the Southeast of the City Hall.",
        "The Gas Station is to the Southeast of the City Hall.",
        "The Bank is to the Southeast of the City Hall.",
        "The Hospital is to the Northeast of the City Hall.",
    ]
    text = _map_prompt(
        sents,
        "Which object is in the Southeast of the City Hall?",
        {
            "A": "Post Office",
            "B": "Gas Station",
            "C": "Bank",
            "D": "Hospital",
            "E": "None of the Options",
        },
    )
    assert letters(text) == {"A", "B", "C"}
    assert "E" not in letters(text)


def test_type1_requires_every_axis():
    """Southeast = East AND South. Post Office has both; Hospital is East only."""
    sents = [
        "The Post Office is to the Southeast of the City Hall.",
        "The Hospital is to the Northeast of the City Hall.",
        "The Park is to the Northwest of the City Hall.",
        "The Museum is to the Southwest of the City Hall.",
    ]
    text = _map_prompt(
        sents,
        "Which object is in the Southeast of the City Hall?",
        {
            "A": "Hospital",
            "B": "Post Office",
            "C": "Park",
            "D": "Museum",
        },
    )
    assert gold(text) == "B"


def test_counterexample_type1_east_is_not_southeast():
    sents = [
        "The Hospital is to the Northeast of the City Hall.",
        "The Post Office is to the Southeast of the City Hall.",
    ]
    text = _map_prompt(
        sents,
        "Which object is in the Southeast of the City Hall?",
        {"A": "Hospital", "B": "Post Office", "C": "City Hall", "D": "Park"},
    )
    assert "A" not in letters(text)
    assert "B" in letters(text)


def test_counterexample_type1_unproven_is_not_in():
    """Church and Zoo share a north-of-Pharmacy chain but are unordered vs each other."""
    sents = [
        "The Church is to the Northeast of the Pharmacy.",
        "The Zoo is to the Northeast of the Pharmacy.",
        "The Coffee Shop is to the Northeast of the Zoo.",
    ]
    text = _map_prompt(
        sents,
        "Which object is in the North of the Zoo?",
        {"A": "Church", "B": "Pharmacy", "C": "Coffee Shop", "D": "Zoo"},
    )
    assert letters(text) == {"C"}
    assert "A" not in letters(text)


def test_counterexample_type1_no_all_four_fallback():
    """Nothing proven in that direction → empty (or E), never A,B,C,D."""
    sents = [
        "The Church is to the Northwest of the Zoo.",
        "The Church is to the Northwest of the Hospital.",
    ]
    text = _map_prompt(
        sents,
        "Which object is in the Northwest of the Zoo?",
        {"A": "Hospital", "B": "Zoo", "C": "Hospital", "D": "Zoo"},
    )
    # Hospital vs Zoo unknown; Zoo is not NW of itself.
    assert gold(text) in {"No valid options found", "E"}
    assert gold(text) != "A,B,C,D"


def test_type1_none_proven_with_e_is_e():
    sents = ["The Church is to the Northeast of the Pharmacy."]
    text = _map_prompt(
        sents,
        "Which object is in the Southwest of the Pharmacy?",
        {"A": "Church", "B": "Pharmacy", "C": "Church", "D": "Pharmacy", "E": E},
    )
    # Church is NE of Pharmacy, so not SW. Pharmacy is not SW of itself.
    assert gold(text) == "E"


# ---------------------------------------------------------------------------
# Type 2 — definite count
# ---------------------------------------------------------------------------

def test_type2_missing_true_count_is_none_of_the_options():
    """Proven count is 2, but 2 is not listed → E, not a wrong integer."""
    sents = [
        "The Post Office is to the Southeast of the City Hall.",
        "The Gas Station is to the Southeast of the City Hall.",
        "The Hospital is to the Northeast of the City Hall.",
    ]
    text = _map_prompt(
        sents,
        "How many objects are in the Southeast of the City Hall?",
        {"A": "0", "B": "1", "C": "3", "D": "4", "E": "None of the Options"},
    )
    assert gold(text) == "E"


def test_counterexample_type2_none_of_the_options_out_when_count_is_listed():
    sents = [
        "The Post Office is to the Southeast of the City Hall.",
        "The Gas Station is to the Southeast of the City Hall.",
    ]
    text = _map_prompt(
        sents,
        "How many objects are in the Southeast of the City Hall?",
        {"A": "2", "B": "0", "C": "1", "D": "3", "E": "None of the Options"},
    )
    assert gold(text) == "A"
    assert "E" not in letters(text)


def test_type1_proven_entity_not_listed_is_none_of_the_options():
    sents = ["The Post Office is to the Southeast of the City Hall."]
    text = _map_prompt(
        sents,
        "Which object is in the Southeast of the City Hall?",
        {"A": "Hospital", "B": "Park", "C": "Museum", "D": "Bank", "E": "None of the Options"},
    )
    assert gold(text) == "E"


def test_counterexample_type0_none_of_the_above_is_not_cannot_determine():
    """Unique NE is proven but not listed. That is None of the Options, not undetermined."""
    sents = ["The Church is to the Northeast of the Pharmacy."]
    text = _map_prompt(
        sents,
        "In which direction is the Church relative to the Pharmacy?",
        {
            "A": "Northwest",
            "B": "Southeast",
            "C": "Southwest",
            "D": "South",
            "E": "None of the Options",
        },
    )
    assert gold(text) == "E"


def test_type2_counts_only_definite_members():
    sents = [
        "The Post Office is to the Southeast of the City Hall.",
        "The Hospital is to the Northeast of the City Hall.",
        "The Gas Station is to the Southeast of the City Hall.",
    ]
    text = _map_prompt(
        sents,
        "How many objects are in the Southeast of the City Hall?",
        {"A": "3", "B": "2", "C": "1", "D": "0"},
    )
    # Post Office + Gas Station; Hospital is NE not SE
    assert gold(text) == "B"


def test_counterexample_type2_does_not_count_unproven_or_wrong_side():
    sents = [
        "The Zoo is to the Northeast of the Pharmacy.",
        "The Church is to the Northeast of the Pharmacy.",
    ]
    text = _map_prompt(
        sents,
        "How many objects are in the North of the Zoo?",
        {"A": "2", "B": "1", "C": "0", "D": "3"},
    )
    # Church is not proven north of Zoo; Pharmacy is south of Zoo.
    assert gold(text) == "C"


# ---------------------------------------------------------------------------
# Generator must obey the same solver (link)
# ---------------------------------------------------------------------------

def test_type2_listed_count_fifth_slot_is_only_a_distractor():
    """When 2 is listed, gold is that letter. Fifth can be either special (both Out)."""
    sents = [
        "The Post Office is to the Southeast of the City Hall.",
        "The Gas Station is to the Southeast of the City Hall.",
    ]
    for fifth in ("None of the Options", "Cannot be determined"):
        text = _map_prompt(
            sents,
            "How many objects are in the Southeast of the City Hall?",
            {"A": "2", "B": "0", "C": "1", "D": "3", "E": fifth},
        )
        assert gold(text) == "A"
        assert "E" not in letters(text)


def test_every_generated_item_has_five_options():
    from generate_all_v6 import generate_sample

    random.seed(5)
    specs = [
        (0, 1, {}),
        (0, 2, {}),
        (0, 0, {}),
        (0, 0, {"incomplete_pair": True}),
        (0, 0, {"omit_live": True}),
        (1, 1, {}),
        (1, 2, {}),
        (2, None, {}),
        (2, 0, {}),
    ]
    for q, tgt, extra in specs:
        got = 0
        for _ in range(40):
            s = generate_sample(
                num_entities=6, num_sentences=6,
                target_num_answers=tgt, question_type=q, **extra,
            )
            if not s:
                continue
            user = next(m["content"] for m in s["messages"] if m["role"] == "user")
            opts = SOLVER.parse_options(user[user.find("Question:"):])
            assert set(opts) == set("ABCDE"), (q, tgt, extra, opts)
            got += 1
            if got >= 2:
                break
        assert got >= 1, (q, tgt, extra)


def test_generator_omit_live_is_none_of_the_options():
    from generate_all_v6 import generate_sample

    random.seed(44)
    got = 0
    for _ in range(80):
        sample = generate_sample(
            num_entities=6, num_sentences=6,
            target_num_answers=0, question_type=0, omit_live=True,
        )
        if not sample:
            continue
        user = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
        grade = SOLVER.grade(user)
        assert grade.accept
        assert _special_is_none_of_options(grade)
        assert "Cannot be determined" not in [
            val for let, val, inn, _ in grade.verdicts if inn
        ]
        got += 1
        if got >= 3:
            break
    assert got >= 3


def _special_is_none_of_options(grade):
    inns = [val for _, val, inn, _ in grade.verdicts if inn]
    return len(inns) == 1 and SOLVER.is_none_of_above(inns[0])


def test_generator_incomplete_pair_is_cannot_determine():
    from generate_all_v6 import generate_sample

    random.seed(41)
    got = 0
    for _ in range(80):
        sample = generate_sample(
            num_entities=6, num_sentences=6,
            target_num_answers=0, question_type=0,
            incomplete_pair=True,
        )
        if not sample:
            continue
        user = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
        assert re.findall(r"Answer:\s*([A-E].*)", asst)[-1].strip() == "E"
        assert SOLVER.grade(user).letters == {"E"}
        got += 1
        if got >= 3:
            break
    assert got >= 3


def test_generator_full_cycle_is_e():
    from generate_all_v6 import generate_sample

    random.seed(21)
    got = 0
    for _ in range(80):
        sample = generate_sample(
            num_entities=6,
            num_sentences=5,
            target_num_answers=0,
            question_type=0,
            inject_conflict="full",
        )
        if not sample:
            continue
        asst = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
        user = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        assert re.findall(r"Answer:\s*([A-E].*)", asst)[-1].strip() == "E"
        assert SOLVER.grade(user).letters == {"E"}
        got += 1
        if got >= 3:
            break
    assert got >= 3


def test_generator_one_axis_conflict_is_two_compounds_not_e():
    from generate_all_v6 import generate_sample

    random.seed(22)
    got = 0
    for _ in range(80):
        sample = generate_sample(
            num_entities=6,
            num_sentences=5,
            target_num_answers=2,
            question_type=0,
            inject_conflict="one_axis",
        )
        if not sample:
            continue
        asst = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
        ans = re.findall(r"Answer:\s*([A-E](?:,\s*[A-E])*)", asst)[-1]
        assert "E" not in ans
        assert len(re.findall(r"[A-E]", ans)) == 2
        got += 1
        if got >= 3:
            break
    assert got >= 3


def test_shuffle_special_moves_none_off_letter_e():
    from generate_all_v6 import generate_sample

    random.seed(99)
    letters = set()
    got = 0
    for _ in range(40):
        sample = generate_sample(
            num_entities=6, num_sentences=6,
            target_num_answers=0, question_type=2,
            shuffle_special=True, shuffle_none_phrase=True,
        )
        if not sample:
            continue
        user = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
        ans = re.findall(r"Answer:\s*([A-E])", asst)[-1]
        letters.add(ans)
        assert SOLVER.agrees(user, ans)
        got += 1
    assert got >= 8
    assert len(letters) >= 2


def test_generator_type2_omit_true_count_is_none_of_the_options():
    from generate_all_v6 import generate_sample

    random.seed(31)
    got = 0
    for _ in range(80):
        sample = generate_sample(
            num_entities=6, num_sentences=6,
            target_num_answers=0, question_type=2,
        )
        if not sample:
            continue
        user = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
        assert re.search(r"None of (the Options|these options|the given options)|Not among the options|No listed option is correct", user)
        assert re.findall(r"Answer:\s*([A-E].*)", asst)[-1].strip() == "E"
        got += 1
        if got >= 3:
            break
    assert got >= 3


@pytest.mark.parametrize(
    "q_type,tgt",
    [(0, 1), (0, 2), (0, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 0), (2, None), (2, 0)],
)
def test_generator_v6_gold_agrees_with_solver(q_type, tgt):
    from generate_all_v6 import generate_sample

    random.seed(7 + q_type + (tgt or 0))
    got = 0
    for _ in range(120):
        n_ent = 8 if q_type == 1 and tgt in (0, 3, 4) else 6
        n_sent = 4 if (q_type == 0 and tgt == 0) else (
            12 if q_type == 1 and tgt in (0, 3, 4) else 7
        )
        sample = generate_sample(
            num_entities=n_ent,
            num_sentences=n_sent,
            target_num_answers=tgt,
            question_type=q_type,
        )
        if not sample:
            continue
        user = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
        ans = re.findall(r"Answer:\s*([A-E](?:,\s*[A-E])*)", asst)[-1]
        assert SOLVER.agrees(user, ans), (q_type, tgt, ans, SOLVER.solve(user))
        n = len(re.findall(r"[A-E]", ans))
        if q_type == 0:
            assert n < 4
        else:
            assert 1 <= n <= 4
        got += 1
        if got >= 5:
            break
    assert got >= 5, f"could not generate enough samples for type={q_type} tgt={tgt}"
