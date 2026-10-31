"""Development-only multi-answer constructions; production allocation is unchanged."""

import json
from pathlib import Path

import pytest

from spatial.v2.composition_production import Entry
from spatial.v2.composition_scene_sampler import sample_scene
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION
from spatial.v2.entailment_contracts import VERSION, Contract
from spatial.v2.entailment_dataset import (
    audit_row,
    measure_depth,
    uniqueness_failure_depth,
)
from spatial.v2.entailment_evidence import build_evidence
from spatial.v2.entailment_production import (
    MULTI_SELECTION_DEPTH,
    SCHEMA,
    admit,
    common_policy,
)
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.solver import Direction

CELLS = [
    (depth, direction, matches, pattern)
    for depth in (1, 4, 8)
    for direction in Direction
    for matches in (2, 3, 4)
    for pattern in ("homogeneous", "mixed-chain", "axis-split")
    if (pattern == "homogeneous" or depth >= 2)
    and (pattern != "mixed-chain" or tuple(Direction).index(direction) % 2)
]


@pytest.mark.parametrize("depth,direction,matches,pattern", CELLS)
def test_all_members_and_uniqueness_certificate_have_requested_depth(
    depth, direction, matches, pattern
):
    seed = (depth + matches + tuple(Direction).index(direction)) % 32
    story = sample_scene(
        seed,
        depth,
        direction=direction,
        matches=matches,
        proof_pattern=pattern,
        shared_selection_support=True,
    )
    assert len(story.objects) <= 31
    assert story.source_family == sample_scene(seed, 2).source_family
    problem = story.problems()[1]
    solver = QuestionSolver()
    evidence = build_evidence(problem, solver=solver)
    # Presentation shuffles candidate order independently of membership.
    assert len(evidence.semantics.entailed) == matches
    assert evidence.semantics.entailed <= set(story.candidates)
    assert story.target in evidence.semantics.entailed
    assert measure_depth(problem, solver=solver)["depth"] == depth
    assert uniqueness_failure_depth(evidence) == depth
    assert all(
        max(len(path["indices"]) for path in support["paths"]) == depth
        for support in evidence.positive_support.values()
    )


def test_two_shallow_matches_cannot_borrow_depth_from_a_third_deep_match():
    story = sample_scene(3, 8, matches=3)
    solver = QuestionSolver()
    evidence = build_evidence(story.problems()[1], solver=solver)
    assert measure_depth(story.problems()[1], solver=solver)["depth"] == 8
    assert uniqueness_failure_depth(evidence) == 1


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    path = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots"
        / REVISION
    )
    if not path.is_dir():
        pytest.skip("pinned local tokenizer unavailable")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def policy(enabled=True):
    result = {
        "schema": SCHEMA,
        "interface": VERSION,
        "paired_omission_percent": 100,
    }
    if enabled:
        result["controlled_multi_selection"] = MULTI_SELECTION_DEPTH
    return result


@pytest.mark.parametrize("matches", (2, 3, 4))
def test_admission_is_opt_in_and_keeps_unique_escape_distinct(
    matches, tokenizer
):
    story = sample_scene(
        3, 8, matches=matches, shared_selection_support=True
    )
    entry = Entry(story, "train", 8, ("which",))
    problem = story.problems()[1]
    rejected = admit(entry, problem, policy(False), tokenizer, QuestionSolver())
    assert rejected["status"] == "depth-rejected"
    accepted = admit(entry, problem, policy(), tokenizer, QuestionSolver())
    assert accepted["status"] == "accepted", accepted.get("reason")
    primary = {
        row["contract"]: row for row in accepted["rows"] if row["menu_role"] == "primary"
    }
    unique = primary[Contract.UNIQUE.value]
    assert unique["answer"] == "E"
    assert unique["uniqueness_failure_depth"] == 8
    assert unique["substantive_depth_support"] is False
    assert unique["decision_depth_support"] is True
    for contract in (Contract.LISTED, Contract.EXHAUSTIVE):
        row = primary[contract.value]
        assert len(row["answer"]) == matches
        assert row["substantive_depth_support"] is True
    tampered = json.loads(json.dumps(unique))
    tampered["uniqueness_failure_depth"] = 1
    with pytest.raises(ValueError, match="uniqueness-failure"):
        audit_row(tampered, load_evidence(accepted), solver=QuestionSolver())


def test_admission_rejects_shallow_unique_failure_even_when_enabled(tokenizer):
    story = sample_scene(3, 8, matches=3)
    entry = Entry(story, "train", 8, ("which",))
    result = admit(entry, story.problems()[1], policy(), tokenizer, QuestionSolver())
    assert result["status"] == "depth-rejected", result.get("reason")
    assert "decision support" in result["reason"]


def test_unknown_policy_and_invalid_shared_support_parameters_fail():
    with pytest.raises(ValueError, match="depth policy"):
        common_policy({**policy(), "controlled_multi_selection": True})
    for matches in (0, 1, 5):
        with pytest.raises(ValueError, match="two through four"):
            sample_scene(3, 2, matches=matches, shared_selection_support=True)
    with pytest.raises(TypeError, match="bool"):
        sample_scene(3, 2, matches=2, shared_selection_support=1)
