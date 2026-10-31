"""External entailment views preserve source identities and declare domain changes."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from spatial.v2.composition_contracts import VerificationError
from spatial.v2.entailment_contracts import Contract
from spatial.v2.entailment_external import (
    SelectionScope,
    adapt_problem,
    build_views,
    external_prompt,
    parse_external_prompt,
    source_paragraph,
    verify_source_export,
    verify_view,
)
from spatial.v2.entailment_scene_protocol import VERSION as SCENE_VERSION
from spatial.v2.tests.test_pivot_evidence import BY_ID


def source(case="S5"):
    fixture = BY_ID[case]
    names = {obj: "Shop " + obj for obj in fixture.objects}
    facts = " ".join(
        f"{names[a]} is to the {direction} of {names[b]}."
        for a, direction, b in fixture.facts
    )
    text = (
        f"Consider a map with multiple objects: {names['O']} is in the map. "
        + facts
        + "\n\nPlease answer the question. Which object is in the North of Shop O? "
        "Available options:\nA. Shop U\nB. Shop V\nC. Shop W\nD. Shop T\n"
        "E. Cannot be determined."
    )
    return {"id": "fixture.scene.1", "text": text, "oracle_option": "E"}


def test_eligibility_is_explicit_and_independent_of_menus():
    row = source()
    original = deepcopy(row)
    narrowed, _, _ = adapt_problem(row["text"], SelectionScope.SOURCE)
    expanded, _, _ = adapt_problem(row["text"], SelectionScope.ALL)
    assert len(narrowed.query.candidates) == 4
    assert len(expanded.query.candidates) == 5
    assert narrowed.premise == expanded.premise
    assert row == original
    with pytest.raises(ValueError):
        adapt_problem(row["text"], "automatic")


@pytest.mark.parametrize("scope", list(SelectionScope))
def test_all_contracts_have_checked_source_and_legend_roundtrips(scope):
    result = build_views(source(), scope, omission_percent=100)
    assert {r["contract"] for r in result["rows"]} == {c.value for c in Contract}
    for row in result["rows"]:
        problem, menu, contract, names = parse_external_prompt(row["prompt"])
        assert names == row["entity_names"]
        assert list(menu) == row["menu"]
        assert contract.value == row["contract"]
        assert len(problem.objects) == 6
        verify_view(row)


def test_all_five_scope_changes_exhaustive_without_changing_premises():
    narrowed = build_views(source(), SelectionScope.SOURCE)
    expanded = build_views(source(), SelectionScope.ALL)
    narrow = next(
        r for r in narrowed["rows"] if r["contract"] == Contract.EXHAUSTIVE.value
    )
    broad = next(
        r for r in expanded["rows"] if r["contract"] == Contract.EXHAUSTIVE.value
    )
    assert narrow["answer"] == "ABCD"
    assert broad["answer"] == "F"
    assert broad["escape_reason"] == "incomplete-entailed-coverage"


def test_native_oracle_letter_is_not_reused():
    row = source("S6")
    first = build_views(row, SelectionScope.ALL)
    row["oracle_option"] = "A"
    second = build_views(row, SelectionScope.ALL)
    assert first == second
    assert all(
        r["answer"] in "ABCD" or r["answer"] == "F"
        for r in first["rows"]
        if r["contract"] == Contract.UNIQUE.value
    )


def test_swapped_source_legend_and_bad_gold_are_rejected():
    row = build_views(source(), SelectionScope.ALL)["rows"][0]
    corrupt = deepcopy(row)
    corrupt["prompt"] = corrupt["prompt"].replace('"Shop U"', '"Different shop"')
    with pytest.raises(VerificationError, match="source entities"):
        verify_view(corrupt)
    corrupt = deepcopy(row)
    corrupt["answer"] = "A"
    with pytest.raises(VerificationError, match="label differs"):
        verify_view(corrupt)


def test_frozen_source_projection_ignores_wrapper_fields_but_not_source_changes(
    tmp_path,
):
    original = source()
    corrected = {**original, "oracle_option": "A"}
    original_path = tmp_path / "original.jsonl"
    export_path = tmp_path / "export.jsonl"
    receipt_path = tmp_path / "receipt.json"
    original_bytes = (json.dumps(original) + "\n").encode()
    original_path.write_bytes(original_bytes)
    canonical = (
        json.dumps(corrected, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    receipt_path.write_text(
        json.dumps(
            {
                "input": {
                    "rows": 1,
                    "sha256": hashlib.sha256(original_bytes).hexdigest(),
                },
                "outputs": {
                    "corrected": {"sha256": hashlib.sha256(canonical).hexdigest()}
                },
            }
        )
    )
    export_path.write_text(
        json.dumps({**corrected, "evaluation_prompt": "old wrapper"}) + "\n"
    )
    assert verify_source_export(original_path, export_path, receipt_path) == [corrected]
    export_path.write_text(json.dumps({**corrected, "text": "changed source"}) + "\n")
    with pytest.raises(ValueError, match="projected corrected source hash"):
        verify_source_export(original_path, export_path, receipt_path)


def test_actual_historical_corr_projection_matches_its_release_receipt():
    root = Path(__file__).resolve().parents[3]
    export = root / "experiments/spatialentail-randomized-v2/prepared-corr-eval"
    original = root / "data/spatialeval_org.jsonl"
    if not export.is_dir() or not original.is_file():
        pytest.skip("historical external correction artifacts are unavailable")
    result = verify_source_export(
        original, export / "data/corr.jsonl", export / "correction-summary.json"
    )
    assert len(result) == 1500


def test_plain_external_prompts_use_original_names_and_all_scene_scope():
    old = build_views(source(), SelectionScope.ALL, omission_percent=100)
    new = build_views(
        source(), SelectionScope.ALL, omission_percent=100, prompt_version=SCENE_VERSION
    )
    assert old["semantics"]["entailed"] == new["semantics"]["entailed"]
    assert [r["answer"] for r in old["rows"]] == [r["answer"] for r in new["rows"]]
    for row in new["rows"]:
        assert "Entity" not in row["prompt"]
        assert "Eligible:" not in row["prompt"]
        assert row["prompt"].splitlines()[1:3] == [
            source_paragraph(source()["text"]),
            "",
        ]
        assert row["prompt"].splitlines()[1].startswith("Shop O is in the map.")
        assert "Shop O is to the" in row["prompt"] or "of Shop O." in row["prompt"]
        assert all("Entity" not in text for text in row["references"].values())
        parsed = parse_external_prompt(
            row["prompt"], prompt_version=SCENE_VERSION, names=row["entity_names"]
        )
        assert parsed[3] == row["entity_names"]
        verify_view(row)
    with pytest.raises(ValueError, match="all-nonreference"):
        build_views(source(), SelectionScope.SOURCE, prompt_version=SCENE_VERSION)
    altered = deepcopy(new["rows"][0])
    altered["display_names"]["Entity0"] = "Other shop"
    with pytest.raises(VerificationError, match="display mapping"):
        verify_view(altered)
    altered = deepcopy(new["rows"][0])
    altered["prompt"] = altered["prompt"].replace("Shop O is in the map. ", "")
    with pytest.raises(VerificationError, match="source paragraph"):
        verify_view(altered)


def test_all_original_source_paragraphs_roundtrip_without_sentence_changes():
    from spatial.v2.entailment_scene_protocol import all_scene_problem

    path = Path(__file__).resolve().parents[3] / "data/spatialeval_org.jsonl"
    if not path.is_file():
        pytest.skip("original source unavailable")
    count = 0
    for line in path.read_text().splitlines():
        item = json.loads(line)
        problem, names, original = adapt_problem(
            item["text"], SelectionScope.ALL, preserve_source_names=True
        )
        problem = all_scene_problem(problem)
        kind = type(original.query).__name__
        menu = (
            ("North", "East", "South", "West")
            if kind == "DirectionQuery"
            else (0, 1, 2, 3)
            if kind == "CountQuery"
            else problem.query.candidates[:4]
        )
        paragraph = source_paragraph(item["text"])
        for contract in Contract:
            prompt = external_prompt(
                problem,
                menu,
                contract,
                names,
                prompt_version=SCENE_VERSION,
                scene_paragraph=paragraph,
            )
            assert prompt.splitlines()[1:3] == [paragraph, ""]
            assert parse_external_prompt(
                prompt, prompt_version=SCENE_VERSION, names=names
            ) == (problem, menu, contract, names)
        count += 1
    assert count == 1500
