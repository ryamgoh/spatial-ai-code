"""Short questions retain full semantics through an explicit system protocol."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from spatial.v2.composition_evaluation import score
from spatial.v2.composition_training import messages
from spatial.v2.entailment_contracts import VERSION as INTERFACE
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_dataset import build_attempt
from spatial.v2.entailment_external import (
    SelectionScope,
    adapt_problem,
    build_views,
    external_prompt,
    parse_external_prompt,
    source_paragraph,
    verify_view,
)
from spatial.v2.entailment_scene_protocol import (
    SCOPE,
    SHORT_VERSION,
    UNIQUE_INSTRUCTION,
    VERSION,
    all_scene_problem,
    parse_prompt,
    parse_row,
    render_prompt,
)
from spatial.v2.entailment_unique_study import STUDY_SCOPE, study_policy
from spatial.v2.solver import Direction
from spatial.v2.tests.test_entailment_scene_protocol import menu_for, scoped_problem


@pytest.mark.parametrize("kind", ("direction", "which", "count"))
@pytest.mark.parametrize("direction", tuple(Direction))
def test_short_questions_change_only_question_line(kind, direction):
    problem = all_scene_problem(scoped_problem(kind, direction))
    menu = menu_for(kind)
    old = render_prompt(problem, menu, Contract.UNIQUE)
    new = render_prompt(problem, menu, Contract.UNIQUE, prompt_version=SHORT_VERSION)
    assert old.splitlines()[:-7] == new.splitlines()[:-7]
    assert old.splitlines()[-6:] == new.splitlines()[-6:]
    assert "guaranteed" not in new
    assert "Select E" not in new and "Select F" not in new
    assert parse_prompt(new, prompt_version=SHORT_VERSION) == (
        problem,
        menu,
        Contract.UNIQUE,
    )
    with pytest.raises(ValueError):
        parse_prompt(new)
    with pytest.raises(ValueError):
        parse_prompt(old, prompt_version=SHORT_VERSION)
    assert parse_prompt(old) == (problem, menu, Contract.UNIQUE)


def test_system_contains_complete_rules_and_shared_style_scope():
    problem = all_scene_problem(scoped_problem())
    row = {
        "id": "short-example",
        "interface": INTERFACE,
        "study_scope": STUDY_SCOPE,
        "prompt_version": SHORT_VERSION,
        "scope_policy": SCOPE,
        "contract": Contract.UNIQUE.value,
        "answer": "E",
        "prompt": render_prompt(
            problem, menu_for("which"), Contract.UNIQUE, prompt_version=SHORT_VERSION
        ),
    }
    for style in ("natural", "symbolic"):
        system, user = messages(row, style)
        assert UNIQUE_INSTRUCTION in system["content"]
        assert "exactly one answer value is entailed" in system["content"]
        assert "another object that could qualify" in system["content"]
        assert user["content"] == row["prompt"]
        assert parse_row(row) == (problem, menu_for("which"), Contract.UNIQUE)
    bad = {**row, "contract": Contract.LISTED.value}
    with pytest.raises(ValueError):
        messages(bad, "natural")
    with pytest.raises(ValueError):
        parse_row(bad)
    with pytest.raises(ValueError):
        messages({**row, "study_scope": None}, "natural")
    with pytest.raises(ValueError):
        render_prompt(
            problem, menu_for("which"), Contract.LISTED, prompt_version=SHORT_VERSION
        )
    old = {**row, "id": "old", "prompt_version": VERSION}
    output = lambda r: {
        "id": r["id"],
        "stage1": {"text": "Answer: E", "token_ids": [], "input_tokens": 0},
        "stage2": {"text": "E", "token_ids": [], "input_tokens": 0},
    }
    assert score([row], [output(row)])["stage1"]["correct"] == 1
    with pytest.raises(ValueError, match="mix prompt versions"):
        score([row, old], [output(row), output(old)])


@pytest.fixture(scope="module")
def tokenizer():
    from spatial.v2.tests.test_entailment_dataset import tokenizer as fixture

    return fixture.__wrapped__()


@pytest.mark.parametrize("kind", ("direction", "which", "count"))
def test_short_synthetic_admission_preserves_unique_labels_evidence_and_menus(
    kind, tokenizer
):
    from spatial.v2.entailment_saved import load_evidence
    from spatial.v2.tests.test_entailment_artifact_audit import AUDIT

    problem = all_scene_problem(scoped_problem(kind))
    story = SimpleNamespace(
        story_id=f"short-{kind}",
        source_family="short-fixture",
        recipe="fixture",
        objects=problem.objects,
        premise=problem.premise,
        candidates=tuple(o for o in problem.objects if o != "Park"),
        distractors=(),
    )
    old = build_attempt(
        story, problem, None, tokenizer, omission_percent=100, prompt_version=VERSION
    )
    new = build_attempt(
        story,
        problem,
        None,
        tokenizer,
        omission_percent=100,
        prompt_version=SHORT_VERSION,
    )
    assert old["status"] == new["status"] == "accepted"
    assert old["semantics"] == new["semantics"]
    assert old["evidence"] == new["evidence"]
    assert load_evidence(new) == load_evidence(old)
    previous = [r for r in old["rows"] if r["contract"] == Contract.UNIQUE.value]
    assert len(previous) == len(new["rows"])
    for before, after in zip(previous, new["rows"]):
        for key in (
            "id",
            "menu",
            "answer",
            "references",
            "depth",
            "decision_depth_support",
        ):
            assert before[key] == after[key]
    AUDIT.audit_record(new)
    assert all(t["admitted"] for t in new["tokens"])
    assert {r["contract"] for r in new["rows"]} == {Contract.UNIQUE.value}


def test_short_external_verification_preserves_source_and_unique_references():
    from spatial.v2.tests.test_entailment_external import source

    for case in ("S5", "S6"):
        item = source(case)
        old = build_views(
            item, SelectionScope.ALL, omission_percent=100, prompt_version=VERSION
        )
        new = build_views(
            item, SelectionScope.ALL, omission_percent=100, prompt_version=SHORT_VERSION
        )
        assert old["semantics"] == new["semantics"]
        prior = [r for r in old["rows"] if r["contract"] == Contract.UNIQUE.value]
        assert len(new["rows"]) == len(prior)
        for before, after in zip(prior, new["rows"]):
            for key in (
                "id",
                "menu",
                "answer",
                "references",
                "scene_paragraph",
                "display_names",
            ):
                assert before[key] == after[key]
            verify_view(after)
            assert after["study_scope"] == STUDY_SCOPE
            assert after["prompt"].splitlines()[1].startswith("Shop O is in the map.")
        bad = deepcopy(new["rows"][0])
        bad["answer"] = "B" if bad["answer"] != "B" else "A"
        with pytest.raises(ValueError):
            verify_view(bad)


def test_all_1500_original_sources_roundtrip_under_short_unique_protocol():
    path = Path(__file__).resolve().parents[3] / "data/spatialeval_org.jsonl"
    if not path.is_file():
        pytest.skip("source data unavailable")
    count = 0
    for line in path.read_text().splitlines():
        item = json.loads(line)
        problem, names, _ = adapt_problem(
            item["text"], SelectionScope.ALL, preserve_source_names=True
        )
        problem = all_scene_problem(problem)
        kind = type(problem.query).__name__
        menu = (
            ("North", "East", "South", "West")
            if kind == "DirectionQuery"
            else (0, 1, 2, 3)
            if kind == "CountQuery"
            else problem.query.candidates[:4]
        )
        scene = source_paragraph(item["text"])
        prompt = external_prompt(
            problem,
            menu,
            Contract.UNIQUE,
            names,
            prompt_version=SHORT_VERSION,
            scene_paragraph=scene,
        )
        assert prompt.splitlines()[1:3] == [scene, ""]
        assert parse_external_prompt(
            prompt, prompt_version=SHORT_VERSION, names=names
        ) == (problem, menu, Contract.UNIQUE, names)
        count += 1
    assert count == 1500


def test_active_policy_uses_short_protocol_and_real_production_admission(tokenizer):
    from spatial.v2.composition_production import Entry
    from spatial.v2.composition_solver_reuse import QuestionSolver
    from spatial.v2.entailment_production import admit
    from spatial.v2.entailment_scene_recipes import controlled_scene

    root = Path(__file__).resolve().parents[3]
    frozen = json.loads(
        (
            root
            / "experiments/spatialentail-composition-v1/entailment-config-frozen-v1.json"
        ).read_text()
    )
    policy = study_policy(frozen)
    assert policy["prompt_version"] == SHORT_VERSION
    story = controlled_scene(
        2026111095,
        2,
        Direction.WEST,
        2,
        story_id="short-production",
        source_family="fixed-family",
    )
    result = admit(
        Entry(story, "train", 2, ("which",)),
        story.problems()[1],
        policy,
        tokenizer,
        QuestionSolver(),
    )
    assert result["status"] == "accepted", result.get("reason")
    assert all(r["study_scope"] == STUDY_SCOPE for r in result["rows"])
    assert all("guaranteed" not in r["prompt"] for r in result["rows"])
