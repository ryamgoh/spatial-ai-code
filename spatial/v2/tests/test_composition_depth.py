"""Depth, family isolation, strict/equality paths, and bounded pilot checks."""

import json
import random
from dataclasses import replace
from itertools import combinations, pairwise, permutations

import pytest

from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    analyze,
)
from spatial.v2.composition_depth import measure_depth, membership_support
from spatial.v2.composition_feasibility import (
    Attempt,
    build_base,
    depth_story,
    schedule,
    split_conflicts,
    summarize,
)
from spatial.v2.composition_feasibility_report import report
from spatial.v2.composition_pilot import renamed_problem
from spatial.v2.difficulty import Axis, _axis_path
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    SpatialProblem,
    SpatialSolverV2,
    WhichQuery,
    direction_between,
    direction_constraint,
)


@pytest.mark.parametrize("depth", range(1, 11))
def test_requested_depth_for_all_three_queries(depth):
    story = depth_story(depth, 0, random.Random(800 + depth))
    for problem in story.problems():
        measurement = measure_depth(problem)
        assert measurement["depth"] == depth
        if not isinstance(problem.query, DirectionQuery):
            assert measurement["ablations"][-1]["answer_changed"]
            assert len(measurement["memberships"]) == 5


@pytest.mark.parametrize("depth", (5, 10))
def test_deep_negative_membership_and_zero_count(depth):
    # sample 1's zero-match case occurs at depths divisible by five.
    story = depth_story(depth, 1, random.Random(90))
    which, count = story.problems()[1:]
    assert not analyze(which).possible
    assert analyze(count).possible == {0}
    for problem in (which, count):
        measurement = measure_depth(problem)
        assert measurement["depth"] == depth
        assert all(not m["qualifies"] for m in measurement["memberships"])
        assert measurement["ablations"][-1]["possible_after"]


def test_shortcut_rejected_before_rendering():
    story = depth_story(8, 0, random.Random(10))
    problem = story.problems()[0]
    direction = Direction(next(iter(analyze(problem).possible)))
    atom = direction_constraint(story.target, story.reference, direction)
    changed = replace(problem, premise=And((*problem.premise.operands, atom)))
    assert measure_depth(changed)["depth"] == 1
    with pytest.raises(UnsupportedEvidence, match="requested depth 8"):
        build_base(
            Attempt(story, "train-pilot", 8),
            changed,
            None,
            SpatialSolverV2(backend="z3"),
        )


def test_ambiguous_and_inconsistent_have_no_depth():
    objects = ("A", "B", "O")
    atom = direction_constraint("A", "O", Direction.NORTH)
    problem = SpatialProblem(objects, And((atom,)), DirectionQuery("A", "B"))
    with pytest.raises(UnsupportedEvidence, match="ambiguous"):
        measure_depth(problem)
    contradiction = direction_constraint("A", "O", Direction.SOUTH)
    with pytest.raises(UnsupportedEvidence, match="inconsistent"):
        measure_depth(replace(problem, premise=And((atom, contradiction))))


def test_count_with_contingent_memberships_has_no_fixed_depth():
    problem = SpatialProblem(
        ("A", "B", "O"),
        And((direction_constraint("A", "O", Direction.NORTH),)),
        CountQuery(frozenset({Direction.NORTH}), "O", ("A", "B")),
    )
    with pytest.raises(UnsupportedEvidence, match="non-fixed-memberships"):
        measure_depth(problem)


def test_count_dependence_is_required_not_assumed_from_paths():
    story = depth_story(3, 0, random.Random(19))
    problem = story.problems()[2]
    # Every premise has an identical alternative, so no single ablation changes
    # the count. This deliberately fails the conservative admission condition.
    duplicated = replace(problem, premise=And(problem.premise.operands * 2))
    with pytest.raises(UnsupportedEvidence, match="no-demonstrated-answer-dependence"):
        measure_depth(duplicated)


def test_exclusion_can_use_one_axis_without_complete_direction():
    problem = SpatialProblem(
        ("A", "B", "O"),
        And(
            (
                direction_constraint("A", "B", Direction.NORTHEAST),
                direction_constraint("B", "O", Direction.SOUTHEAST),
            )
        ),
        WhichQuery(frozenset({Direction.NORTH}), "O", ("A",)),
    )
    support = membership_support(problem, "A", False)
    assert support["depth"] == 2
    assert len(support["paths"]) == 1
    assert support["paths"][0]["axis"] == "x"
    assert not analyze(problem).possible


def test_unknown_solver_result_is_not_a_depth_label():
    class TimeoutSolver:
        def analyze(self, *args, **kwargs):
            raise RuntimeError("timeout")

    problem = depth_story(2, 0, random.Random(1)).problems()[0]
    with pytest.raises(VerificationError, match="timeout"):
        measure_depth(problem, solver=TimeoutSolver())


def test_schedule_is_fixed_and_roles_precede_admission():
    first = schedule()
    assert first == schedule()
    assert len(first) == 94
    assert sum(len(a.story.problems()) for a in first) == 282
    families = {}
    for item in first:
        families.setdefault(item.story.source_family, set()).add(item.role)
        if item.requested_depth and item.requested_depth > 8:
            assert item.role == "extrapolation-pilot"
    assert all(len(roles) == 1 for roles in families.values())
    assert first != schedule(9)


def test_measured_depth_survives_rename_and_premise_reordering():
    story = depth_story(10, 0, random.Random(42))
    for problem in story.problems():
        variant, _ = renamed_problem(problem)
        assert measure_depth(variant)["depth"] == measure_depth(problem)["depth"] == 10


def test_token_rejected_base_contributes_no_matched_views_to_summary():
    result = summarize(
        [
            {
                "status": "token-rejected",
                "data": {
                    "rows": [{"id": "must-not-be-counted"}],
                    "tokens": [
                        {
                            "style": "natural",
                            "target_tokens": 5000,
                            "training_tokens": 5600,
                            "stage2_reserved_tokens": 4800,
                        }
                    ],
                },
            }
        ]
    )
    assert result["accepted_bases"] == result["views_per_style"] == 0
    assert result["token_maxima_before_admission"]["natural"]["target_tokens"] == 5000


def test_canonical_collision_quarantines_whole_family():
    story = depth_story(3, 0, random.Random(9))
    sibling = replace(depth_story(4, 0, random.Random(3)), source_family="family-a")
    attempts = (
        Attempt(replace(story, source_family="family-a"), "train-pilot", 3),
        Attempt(sibling, "train-pilot", 4),
        Attempt(
            replace(story, story_id="copy", source_family="family-b"), "dev-pilot", 3
        ),
    )
    _, bad = split_conflicts(attempts)
    assert bad == {"family-a", "family-b"}


def test_empty_summary_does_not_invent_results():
    result = summarize([])
    assert result["accepted_bases"] == 0
    assert all(
        v["prediction_from_train_only"] is None for v in result["baseline"].values()
    )


def test_report_refuses_incomplete_pilot(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"attempt_budget": 1}))
    (tmp_path / "summary.json").write_text(json.dumps(summarize([])))
    (tmp_path / "attempts.jsonl").write_text("")
    with pytest.raises(ValueError, match="incomplete pilot"):
        report(tmp_path)


@pytest.mark.parametrize("seed", range(12))
def test_shortest_axis_path_matches_independent_small_path_enumeration(seed):
    rng = random.Random(seed)
    objects = ("A", "B", "C", "D")
    points = dict(
        zip(objects, rng.sample([(x, y) for x in range(4) for y in range(4)], 4))
    )
    pairs = rng.sample(list(combinations(objects, 2)), rng.randint(1, 6))
    atoms = tuple(
        direction_constraint(a, b, direction_between(points[a], points[b]))
        for a, b in pairs
    )
    problem = SpatialProblem(objects, And(atoms), DirectionQuery("A", "D"))
    for axis_index, axis in enumerate((Axis.X, Axis.Y)):
        # Independent edge construction from the sampled coordinate relations.
        # Coordinates determine only included edges, never absent relations.
        edges = {}
        for a, b in pairs:
            delta = points[b][axis_index] - points[a][axis_index]
            if delta >= 0:
                edges[a, b] = delta > 0
            if delta <= 0:
                edges[b, a] = delta < 0
        for subject, reference in permutations(objects, 2):
            for sign in (-1, 0, 1):
                start, end = (reference, subject) if sign > 0 else (subject, reference)
                middle = [x for x in objects if x not in {start, end}]
                expected = None
                for size in range(3):
                    for nodes in permutations(middle, size):
                        path = (start, *nodes, end)
                        flags = [edges.get((a, b)) for a, b in pairwise(path)]
                        if any(flag is None for flag in flags):
                            continue
                        if (sign == 0 and not any(flags)) or (sign != 0 and any(flags)):
                            expected = len(path) - 1
                            break
                    if expected is not None:
                        break
                actual = _axis_path(problem, axis, subject, reference, sign)
                assert (len(actual.premise_indices) if actual else None) == expected
