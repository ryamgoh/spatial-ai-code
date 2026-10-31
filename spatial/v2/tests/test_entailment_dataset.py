"""Entailment admissions with actual pinned tokenizer; no production families."""

from pathlib import Path

import pytest

from spatial.v2.composition_execution_benchmark import development_jobs
from spatial.v2.composition_training import ENTAILMENT_STYLES, REVISION, messages
from spatial.v2.entailment_contracts import VERSION, Contract, analyze
from spatial.v2.entailment_dataset import build_attempt, menus_for
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
        pytest.skip("pinned local tokenizer is unavailable")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.mark.parametrize("index", range(6))
def test_existing_cpu_development_admission(index, tokenizer):
    _, entry, problem = development_jobs()[index]
    record = build_attempt(entry.story, problem, None, tokenizer, omission_percent=100)
    assert record["status"] == "accepted", record.get("reason")
    assert record["interface"] == VERSION
    assert "possible" not in record["semantics"]
    assert len(record["rows"]) in (3, 6)
    assert all(t["admitted"] for t in record["tokens"])
    for row in record["rows"]:
        assert row["contract"] in {c.value for c in Contract}
        assert row["interface"] == VERSION
        for style in ("natural", "symbolic"):
            data = messages(row, style)
            assert data[0]["content"] == ENTAILMENT_STYLES[style]
            assert "references" not in data[1]
            assert "counterexamples" not in data[1]
            assert data[1]["content"] == row["prompt"]


def test_menu_omits_entailed_not_merely_compatible_value():
    semantics = analyze(BY_ID["S3"].problem())
    # With four eligible candidates there is no unlisted replacement value.
    assert len(menus_for(semantics, 0, omission_percent=100)) == 1
    semantics = analyze(BY_ID["S6"].problem())
    full, partial = menus_for(semantics, 0, omission_percent=100)
    assert semantics.entailed <= set(full)
    assert not semantics.entailed <= set(partial)
    assert sum(a != b for a, b in zip(full, partial)) == 1


def test_diagnostic_negatives_include_contingent_and_impossible():
    from spatial.v2.composition_solver_reuse import QuestionSolver

    categories = QuestionSolver().negative_categories(BY_ID["S4"].problem())
    assert categories["U"] == "contingent"
    assert categories["W"] == "impossible"
    assert categories["T"] == "impossible"


def test_selection_empty_entailment_not_forced_into_depth_quota():
    from spatial.v2.composition_contracts import UnsupportedEvidence
    from spatial.v2.entailment_dataset import measure_depth

    with pytest.raises(UnsupportedEvidence):
        measure_depth(BY_ID["S2"].problem())
