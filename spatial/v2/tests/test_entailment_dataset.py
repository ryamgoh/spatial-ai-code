"""Entailment admissions with actual pinned tokenizer; no production families."""

from pathlib import Path

import pytest

from spatial.v2.composition_production import Entry
from spatial.v2.composition_scene_sampler import sample_scene
from spatial.v2.composition_training import ENTAILMENT_STYLES, REVISION, messages
from spatial.v2.entailment_contracts import VERSION, Contract, analyze
from spatial.v2.entailment_dataset import build_attempt, menus_for
from spatial.v2.solver import Direction
from spatial.v2.tests.test_pivot_evidence import BY_ID


def development_jobs():
    """The same six admission cases formerly owned by the CPU timing runner."""
    scenes = [
        sample_scene(3, 2, matches=2),
        sample_scene(
            7,
            8,
            direction=Direction.NORTHEAST,
            proof_pattern="axis-split",
            matches=0,
        ),
    ]
    jobs = []
    for story in scenes:
        entry = Entry(
            story,
            "train",
            2 if story == scenes[0] else 8,
            ("direction", "which", "count"),
        )
        for problem in story.problems():
            jobs.append((len(jobs), entry, problem))
    return jobs


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


@pytest.mark.parametrize("case", ("sole", "no-entailed", "inconsistent"))
@pytest.mark.parametrize("count", range(5))
def test_count_pool_changes_options_not_complete_semantics(case, count):
    from spatial.v2.composition_contracts import answer_domain
    from spatial.v2.entailment_contracts import encode
    from spatial.v2.entailment_generator import SceneRequest, build_scene

    _, problem = build_scene(
        SceneRequest(1, Direction.NORTH, "count", case, count),
        identity=f"count-menu-{case}-{count}",
        family="menu-test",
        structure_seed=1234,
    )
    semantics = analyze(problem)
    assert len(answer_domain(problem)) > 5
    assert set(semantics.counterexamples) == (
        set(answer_domain(problem)) - semantics.entailed
        if semantics.consistent
        else set()
    )
    legacy = menus_for(semantics, 42, omission_percent=100)
    assert legacy == menus_for(
        semantics, 42, omission_percent=100, count_menu_values=None
    )
    for seed in range(20):
        menus = menus_for(
            semantics, seed, omission_percent=100, count_menu_values=range(5)
        )
        assert all(len(menu) == len(set(menu)) == 4 for menu in menus)
        assert all(set(menu) <= set(range(5)) for menu in menus)
        if case == "sole":
            assert len(menus) == 2
            full, omission = menus
            assert count in full and count not in omission
            assert sum(a != b for a, b in zip(full, omission)) == 1
            assert encode(semantics, full, Contract.UNIQUE).letters in "ABCD"
            assert encode(semantics, omission, Contract.UNIQUE).letters == "F"
        else:
            assert len(menus) == 1
            assert encode(semantics, menus[0], Contract.UNIQUE).letters == "E"


@pytest.mark.parametrize("pool", ((0, 1, 2, 2, 3), (False, 1, 2, 3, 4), (0, 1, 2, 99)))
def test_count_menu_pool_rejects_invalid_values(pool):
    semantics = analyze(BY_ID["C2"].problem())
    with pytest.raises(ValueError, match="invalid Count menu pool"):
        menus_for(semantics, 0, count_menu_values=pool)


def test_count_menu_pool_rejects_non_count_or_hidden_entailed_value():
    from spatial.v2.composition_contracts import answer_domain

    semantics = analyze(BY_ID["D1"].problem())
    with pytest.raises(ValueError, match="invalid Count menu pool"):
        menus_for(semantics, 0, count_menu_values=range(5))
    semantics = analyze(BY_ID["C2"].problem())
    excluded = tuple(
        v for v in answer_domain(semantics.problem) if v not in semantics.entailed
    )
    with pytest.raises(ValueError, match="entailed value outside pool"):
        menus_for(semantics, 0, count_menu_values=excluded)
