import hashlib
import json
import random
from collections import Counter
from copy import deepcopy

import pytest

from spatial.v2.composition_balance import build_attempt, menu_inventory
from spatial.v2.composition_contracts import analyze
from spatial.v2.composition_depth import measure_depth
from spatial.v2.composition_direction_coverage import (
    check_target,
    depth_visible_menus,
    schedule,
)
from spatial.v2.composition_feasibility import stable_int
from spatial.v2.composition_selection_expansion import PATTERNS, make_story
from spatial.v2.composition_selection_expansion import schedule as historical_schedule
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import Direction, SpatialSolverV2

ENTRIES = schedule()


def test_historical_schedule_is_unchanged():
    data = [
        (s.story_id, s.source_family, d, tagged_dataclass_to_dict(s.problems()[1]))
        for s, d in historical_schedule()
    ]
    assert hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() == (
        "7568a21d0b5631d0aab639c932cfc0d2f2e2675dde9eab8b984a35829a8cdcf4"
    )


def test_exact_direction_depth_pattern_coverage():
    assert ENTRIES == schedule()
    assert len({s.story_id for s, _ in ENTRIES}) == 240
    assert Counter((s.recipe, s.direction, d) for s, d in ENTRIES) == {
        (pattern, direction, depth): 1
        for pattern in PATTERNS
        for direction in Direction
        for depth in range(1, 11)
    }
    assert len({s.source_family for s, _ in ENTRIES}) == 3


@pytest.mark.parametrize("story,depth", ENTRIES, ids=[s.story_id for s, _ in ENTRIES])
def test_every_direction_has_unique_match_and_answer_dependent_depth(story, depth):
    problem = story.problems()[1]
    semantics = analyze(problem)
    measured = measure_depth(problem)
    record = {
        "status": "accepted",
        "semantics": {
            "consistent": semantics.consistent,
            "possible": sorted(semantics.possible),
            "certain": sorted(semantics.certain),
        },
        "measurement": measured,
        "rows": [
            {"contract": c, "substantive_depth_support": True}
            for c in ("single", "visible-possible", "all-possible")
        ],
    }
    check_target(record, story, depth)
    assert record["status"] == "accepted", record
    assert problem.query.directions == {story.direction}


def test_bad_direction_rejected():
    with pytest.raises(ValueError, match="must be a Direction"):
        make_story("match-chain", 3, 0, random.Random(1), direction="Northeast")


def test_target_check_rejects_missing_contract_depth_support():
    story, depth = ENTRIES[0]
    original = {
        "status": "accepted",
        "semantics": {
            "consistent": True,
            "possible": [story.target],
            "certain": [story.target],
        },
        "measurement": measure_depth(story.problems()[1]),
        "rows": [{"contract": "single", "substantive_depth_support": True}],
    }
    record = deepcopy(original)
    check_target(record, story, depth)
    assert record["status"] == "verification-failed"
    assert record["stage"] == "target-coverage"


def test_menu_correction_preserves_semantics_and_printed_depth(monkeypatch):
    from spatial.v2 import composition_balance

    story, depth = next(
        (s, d)
        for s, d in ENTRIES
        if s.recipe == "rival-chain" and d == 3 and s.direction is Direction.SOUTHWEST
    )
    problem = story.problems()[1]
    semantics = analyze(problem)
    seed = stable_int(story.story_id + "-which")
    original = menu_inventory(semantics, seed)
    corrected = depth_visible_menus(semantics, seed)
    deep = next(
        m["candidate"]
        for m in measure_depth(problem)["memberships"]
        if not m["qualifies"] and m["depth"] == depth
    )
    assert deep not in original[1] and deep in corrected[1]
    assert corrected[0] == original[0]
    for before, after in zip(original, corrected):
        assert len(after) == len(set(after)) == 4
        assert semantics.possible.intersection(
            before
        ) == semantics.possible.intersection(after)
        if story.target in after:
            assert before.index(story.target) == after.index(story.target)
    monkeypatch.setattr(
        composition_balance, "token_measurement", lambda *args: {"admitted": True}
    )
    solver = SpatialSolverV2(backend="z3")
    old = build_attempt(story, problem, depth, None, solver)
    assert old["status"] == "accepted"
    check_target(old, story, depth)
    assert old["status"] == "verification-failed"
    new = build_attempt(
        story, problem, depth, None, solver, menu_selector=depth_visible_menus
    )
    check_target(new, story, depth)
    assert new["status"] == "accepted", new
    assert new["semantics"] == old["semantics"]
    assert [r["answer"] for r in new["rows"]] == [r["answer"] for r in old["rows"]]
    assert new["problem"] == old["problem"]
