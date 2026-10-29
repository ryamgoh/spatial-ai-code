"""Pilot construction, leakage prevention, menu feasibility, and rejection tests."""

import json
from dataclasses import replace

import pytest

from spatial.v2.composition_contracts import (
    UnsupportedEvidence,
    VerificationError,
    analyze,
    encode,
)
from spatial.v2.composition_pilot import (
    assign_splits,
    check_splits,
    construct_stories,
    menus_for,
    renamed_problem,
    run_pilot,
    story_key,
    verify_base,
    write_pilot,
)
from spatial.v2.grading import AnswerMode
from spatial.v2.solver import And, Direction, direction_constraint


def test_construction_is_seeded_bounded_and_not_fixture_imports():
    stories = construct_stories(per_recipe=2)
    assert stories == construct_stories(per_recipe=2)
    assert stories != construct_stories(seed=9, per_recipe=2)
    assert len(stories) == 20
    for story in stories:
        assert all(len(atom.allowed) == 1 for atom in story.premise.operands)
        assert len(story.candidates) >= 4
    with pytest.raises(ValueError):
        construct_stories(per_recipe=6)


def test_story_key_ignores_queries_renames_order_and_inverse_orientation():
    story = construct_stories(per_recipe=1)[0]
    problem = story.problems()[0]
    variant, _ = renamed_problem(problem)
    key = story_key(problem.objects, problem.premise)
    assert story_key(variant.objects, variant.premise) == key
    directions = tuple(Direction)
    inverse = And(
        tuple(
            direction_constraint(
                atom.reference,
                atom.subject,
                directions[(directions.index(next(iter(atom.allowed))) + 4) % 8],
            )
            for atom in problem.premise.operands
        )
    )
    assert story_key(problem.objects, inverse) == key
    assert all(story_key(p.objects, p.premise) == key for p in story.problems())


def test_duplicate_and_matched_cycle_families_share_split():
    stories = construct_stories(per_recipe=1)
    original = stories[0]
    duplicate = replace(original, story_id="duplicate", source_family="other-source")
    assignments = assign_splits((*stories, duplicate), 8)
    assert assignments[original.story_id] == assignments["duplicate"]
    cycle = [s for s in stories if s.recipe == "cycle"]
    assert len(cycle) == 2
    assert assignments[cycle[0].story_id] == assignments[cycle[1].story_id]
    assert {split for _, split in assignments.values()} == {"train", "dev", "test"}


@pytest.mark.parametrize("recipe", ("singleton", "empty", "multiple", "joint"))
def test_paired_menus_obey_coverage_without_fake_values(recipe):
    story = next(s for s in construct_stories(per_recipe=1) if s.recipe == recipe)
    semantics = analyze(story.problems()[2])
    menus = menus_for(semantics, 42)
    assert menus == menus_for(semantics, 42)
    for menu in menus:
        assert len(menu) == len(set(menu)) == 4
        assert set(menu) <= set(range(len(story.candidates) + 1))
    if len(menus) == 2:
        assert sum(a != b for a, b in zip(*menus)) == 1
        assert semantics.possible <= set(menus[0])
        assert not semantics.possible <= set(menus[1])
        assert encode(semantics, menus[1], AnswerMode.ALL_POSSIBLE).letters == "F"


def test_proposed_distractor_that_changes_semantics_is_rejected():
    story = next(s for s in construct_stories(per_recipe=1) if s.recipe == "singleton")
    problem = story.problems()[2]
    matching = next(
        i
        for i, atom in enumerate(problem.premise.operands)
        if {atom.subject, atom.reference} == {"C0", "O"}
    )
    story = replace(story, distractors=(matching,))
    with pytest.raises(VerificationError, match="distractors"):
        verify_base(story, problem, analyze(problem))


def test_split_checker_rejects_cross_split_story():
    bases = {
        "a": {"story_key": "same", "family_id": "f1", "split": "train"},
        "b": {"story_key": "same", "family_id": "f2", "split": "test"},
    }
    rows = [
        {
            "base_id": key,
            "family_id": base["family_id"],
            "split": base["split"],
            "story_id": key,
        }
        for key, base in bases.items()
    ]
    with pytest.raises(VerificationError, match="leakage"):
        check_splits(rows, bases)


@pytest.fixture(scope="module")
def pilot():
    return run_pilot(per_recipe=1)


def test_pilot_all_views_have_matched_references_and_coverage(pilot):
    assert pilot["counts"]["attempted_bases"] == 30
    assert pilot["counts"]["attempt_status"].get("verification-failed", 0) == 0
    assert any(
        base["checks"]["reference_crosscheck"].startswith("passed")
        for base in pilot["bases"].values()
    )
    assert {r["query_kind"] for r in pilot["rows"]} == {"direction", "which", "count"}
    for row in pilot["rows"]:
        assert set(row["references"]) == {"natural", "symbolic"}
        assert row["references"]["natural"].endswith("Answer: " + row["answer"])
        assert row["references"]["symbolic"].endswith("Answer: " + row["answer"])
    check_splits(pilot["rows"], pilot["bases"])


def test_export_separates_model_inputs_from_gold_and_evidence(pilot, tmp_path):
    write_pilot(pilot, tmp_path)
    inputs = [
        json.loads(line)
        for line in (tmp_path / "model-inputs.jsonl").read_text().splitlines()
    ]
    assert len(inputs) == len(pilot["rows"])
    for row in inputs:
        assert set(row) == {"id", "split", "messages"}
        assert [message["role"] for message in row["messages"]] == ["user"]
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["manual_audit"] == "pending"
    assert receipt["source_sha256"]


def test_rejections_preserve_original_split_assignments(monkeypatch):
    import spatial.v2.composition_pilot as module

    expected = assign_splits(construct_stories(per_recipe=1), 20261028)

    def reject(*args, **kwargs):
        raise UnsupportedEvidence("unsupported test evidence")

    monkeypatch.setattr(module, "build_evidence", reject)
    pack = run_pilot(per_recipe=1)
    assert pack["split_assignments"] == expected
    assert not pack["rows"]
    assert len(pack["attempts"]) == 30
    assert all(a["status"] == "unsupported" for a in pack["attempts"])
    assert all("problem" in a for a in pack["attempts"])


def test_correctness_failure_is_reported_not_silently_unsupported(monkeypatch):
    import spatial.v2.composition_pilot as module

    def fail(*args, **kwargs):
        raise VerificationError("corrupt evidence")

    monkeypatch.setattr(module, "build_evidence", fail)
    pack = run_pilot(per_recipe=1)
    assert pack["counts"]["attempt_status"] == {"verification-failed": 30}
    assert not pack["rows"]
    assert all("problem" in a for a in pack["attempts"])
