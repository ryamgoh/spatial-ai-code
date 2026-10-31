"""Existing questions must retain identity, semantics and menus during replay."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"
)
SPEC = importlib.util.spec_from_file_location(
    "unique_pool_replay", DIRECTORY / "replay_unique_training_pool.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def existing():
    from spatial.v2.composition_pilot import Story
    from spatial.v2.entailment_scene_protocol import VERSION
    from spatial.v2.entailment_unique_study import training_allocation_policy
    from spatial.v2.solver import Direction
    from spatial.v2.tests.test_entailment_dataset import tokenizer
    from spatial.v2.tests.test_entailment_scene_protocol import scoped_problem

    problem = scoped_problem("direction")
    story = Story(
        "replay-fixture",
        "fixture-family",
        "fixture",
        problem.objects,
        problem.premise,
        "Library",
        "Park",
        tuple(o for o in problem.objects if o != "Park"),
        Direction.NORTH,
        (),
    )
    entry = MODULE.first.historical.Entry(story, "train", None, ("direction",))
    policy = training_allocation_policy(
        json.loads((DIRECTORY / "entailment-config-frozen-v1.json").read_text())
    )
    tok = tokenizer.__wrapped__()
    old = MODULE.first.production.admit(
        entry,
        problem,
        {**policy, "prompt_version": VERSION},
        tok,
        MODULE.first.QuestionSolver(),
    )
    assert old["status"] == "accepted"
    old.update(
        query_key=MODULE.first.query_profile(MODULE.first.all_scene_problem(problem))[
            "key"
        ],
        stratum="train/direction/semantic",
        source_stratum="train/direction/targeted",
    )
    return entry, old, policy, tok


def test_reconstruct_preserves_old_metadata_and_audits_new_explanation(
    existing, monkeypatch
):
    entry, old, policy, tok = existing
    monkeypatch.setattr(MODULE.first, "POLICY", policy)
    monkeypatch.setattr(MODULE.first, "TOKENIZER", tok)
    before = deepcopy(old)
    result = MODULE.reconstruct((entry, old))
    assert old == before
    assert result["semantics"] == old["semantics"]
    assert result["query_key"] == old["query_key"]
    assert all(row["selected_depth_audit"] == "passed" for row in result["rows"])
    assert len(result["rows"]) * 3 == len(old["rows"])


def test_changed_preserved_label_is_rejected(existing, monkeypatch):
    entry, old, policy, tok = existing
    monkeypatch.setattr(MODULE.first, "POLICY", policy)
    monkeypatch.setattr(MODULE.first, "TOKENIZER", tok)
    changed = deepcopy(old)
    changed["rows"][0]["answer"] = "E"
    with pytest.raises(ValueError, match="menu answer changed"):
        MODULE.reconstruct((entry, changed))
