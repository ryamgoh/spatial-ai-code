"""Ordinary names preserve all claims and support multiword places end to end."""

from pathlib import Path

import pytest

from spatial.v2.composition_contracts import validate_problem as legacy_validate
from spatial.v2.composition_execution_benchmark import development_jobs
from spatial.v2.composition_training import REVISION
from spatial.v2.entailment_contracts import Contract, parse_prompt, render_prompt
from spatial.v2.entailment_dataset import build_attempt
from spatial.v2.entailment_evidence import build_evidence, render_pair
from spatial.v2.entailment_names import (
    NAMES,
    rename_record,
    rename_structure,
    story_name_map,
)
from spatial.v2.tests.test_entailment_artifact_audit import AUDIT
from spatial.v2.tests.test_entailment_positions import fixture_record
from spatial.v2.tests.test_pivot_evidence import BY_ID


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


def test_fixed_vocabulary_covers_31_objects_without_collisions():
    objects = tuple(f"Object{i}" for i in range(31))
    names = story_name_map(objects, "story-a")
    assert len(names) == len(set(names.values())) == 31
    assert set(names.values()) == set(NAMES)
    assert "City Hall" in names.values()
    assert "Courthouse" in names.values()
    assert names == story_name_map(objects[::-1], "story-a")
    assert names != story_name_map(objects, "story-b")
    with pytest.raises(ValueError, match="fixed vocabulary"):
        story_name_map((*objects, "Object31"), "story-a")


def test_rename_only_exact_entity_values_and_keys():
    value = {"Object1": ["Object1", "Object10", "production-Object1"]}
    assert rename_structure(value, {"Object1": "City Hall"}) == {
        "City Hall": ["City Hall", "Object10", "production-Object1"]
    }


def test_independent_artifact_auditor_accepts_multiword_selection():
    original = fixture_record("S6", ("U", "V", "W", "T"))
    old = BY_ID["S6"].problem()
    names = dict(
        zip(
            old.objects,
            (
                "City Hall",
                "Train Station",
                "Veterinary Clinic",
                "Courthouse",
                "Bakery",
                "High School",
            ),
        )
    )
    problem = rename_structure(old, names)
    evidence = build_evidence(problem)
    record = rename_structure(original, names)
    for row in record["rows"]:
        contract, menu = Contract(row["contract"]), tuple(row["menu"])
        row["prompt"] = render_prompt(problem, menu, contract)
        row["references"] = render_pair(evidence, menu, contract)
    assert AUDIT.audit_record(record) == (1, 4)


@pytest.mark.parametrize("name", ("D1", "S6", "C2", "I1", "I3"))
def test_multiword_places_parse_and_have_checked_matched_evidence(name):
    old = BY_ID[name].problem()
    pool = (
        "City Hall",
        "Train Station",
        "Veterinary Clinic",
        "Courthouse",
        "Bakery",
        "High School",
    )
    names = dict(zip(old.objects, pool))
    problem = rename_structure(old, names)
    with pytest.raises(ValueError, match="ASCII identifiers"):
        legacy_validate(problem)
    evidence = build_evidence(problem)
    old_menu = (
        ("U", "V", "W", "T")
        if name == "S6"
        else (0, 1, 2, 3)
        if name == "C2"
        else ("North", "East", "South", "West")
    )
    menu = rename_structure(old_menu, names)
    for contract in Contract:
        prompt = render_prompt(problem, menu, contract)
        assert parse_prompt(prompt) == (problem, menu, contract)
        pair = render_pair(evidence, menu, contract)
        assert all("Object" not in text for text in pair.values())


@pytest.mark.parametrize("index", range(6))
def test_real_admission_name_revision_keeps_labels_and_proof_metadata(index, tokenizer):
    _, entry, problem = development_jobs()[index]
    original = build_attempt(
        entry.story, problem, None, tokenizer, omission_percent=100
    )
    assert original["status"] == "accepted"
    revised = rename_record(original, tokenizer)
    assert revised["base_id"] == original["base_id"]
    assert revised["family_id"] == original["family_id"]
    assert [row["answer"] for row in revised["rows"]] == [
        row["answer"] for row in original["rows"]
    ]
    assert all(token["admitted"] for token in revised["tokens"])
    assert revised["measurement"]["depth"] == original["measurement"]["depth"]
