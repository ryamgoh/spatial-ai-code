"""Versioned admission, depth, quotas, and exports on development fixtures."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from spatial.v2 import composition_production as historical
from spatial.v2 import entailment_production as production
from spatial.v2.composition_scene_sampler import sample_scene
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.composition_training import REVISION
from spatial.v2.entailment_contracts import VERSION, Contract
from spatial.v2.tests import composition_policy

TOKENIZER = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots" / REVISION
)


@pytest.fixture
def policy():
    approved = composition_policy(coverage=True)
    contracts = {
        "single": Contract.UNIQUE.value,
        "visible-possible": Contract.LISTED.value,
        "all-possible": Contract.EXHAUSTIVE.value,
    }
    # Test-only policy: approval markers do not constitute a production approval.
    return {
        **approved,
        "schema": production.SCHEMA,
        "interface": VERSION,
        "status": "frozen",
        "coverage_review": {"status": "approved"},
        "coverage_minima": [
            {
                **rule,
                "match": {
                    **rule["match"],
                    "contract": contracts[rule["match"]["contract"]],
                },
            }
            for rule in approved["coverage_minima"]
        ],
        "source_batches": [
            {
                "generator": "sampled-forest",
                "seed": 3,
                "count": 1,
                "depth": 2,
                "split": "train",
                "query_kinds": ["direction", "which", "count"],
            }
        ],
        "development_exclusions": {
            "path": "test-only-exclusions.json",
            "sha256": "0" * 64,
        },
    }


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    if not TOKENIZER.is_dir():
        pytest.skip("pinned local tokenizer unavailable")
    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def entries():
    return (
        historical.Entry(
            sample_scene(3, 2, matches=1), "train", 2, ("direction", "which", "count")
        ),
    )


def test_policy_requires_semantic_version_and_review(policy):
    assert not production.plan(policy)["blockers"]
    before = deepcopy(policy)
    production.common_policy(policy)
    assert policy == before
    policy["coverage_review"] = {"status": "pending"}
    assert production.plan(policy)["blockers"]
    policy["schema"] = historical.SCHEMA
    with pytest.raises(ValueError, match="versioned"):
        production.plan(policy)


def test_plain_policy_names_objects_but_does_not_reuse_scoped_freeze(policy, tokenizer):
    from spatial.v2.entailment_scene_protocol import SCOPE, parse_row
    from spatial.v2.entailment_scene_protocol import VERSION as SCENE_VERSION

    policy.update(prompt_version=SCENE_VERSION, scope_policy=SCOPE)
    assert any(
        "All-scene allocation" in text for text in production.plan(policy)["blockers"]
    )
    entry = entries()[0]
    problem = entry.story.problems()[0]
    result = production.admit(entry, problem, policy, tokenizer, QuestionSolver())
    assert result["status"] == "accepted", result.get("reason")
    assert result["scope_policy"] == SCOPE
    for row in result["rows"]:
        assert "Object" not in row["prompt"]
        assert "Eligible:" not in row["prompt"]
        assert parse_row(row)[0].objects == tuple(
            sorted(result["entity_name_map"].values())
        )
    policy["scope_policy"] = "unknown"
    with pytest.raises(ValueError, match="scene protocol"):
        production.common_policy(policy)


@pytest.mark.parametrize("kind,matches", (("count", 0), ("count", 4), ("which", 4)))
def test_repaired_scene_preserves_nonhomogeneous_and_decision_depth_metadata(
    policy, tokenizer, kind, matches
):
    from spatial.v2.entailment_scene_protocol import SCOPE
    from spatial.v2.entailment_scene_protocol import VERSION as SCENE_VERSION
    from spatial.v2.entailment_scene_recipes import controlled_scene
    from spatial.v2.solver import Direction

    policy.update(
        prompt_version=SCENE_VERSION,
        scope_policy=SCOPE,
        controlled_multi_selection=production.MULTI_SELECTION_DEPTH,
    )
    story = controlled_scene(
        2026111094,
        10,
        Direction.NORTH,
        matches,
        story_id="repaired-test",
        source_family="fixed-family",
    )
    entry = historical.Entry(story, "extrapolation-test", 10, (kind,))
    problem = story.problems()[historical.KINDS.index(kind)]
    record = production.admit(entry, problem, policy, tokenizer, QuestionSolver())
    assert record["status"] == "accepted", record.get("reason")
    assert record["proof_pattern"] == "axis-split"
    primary = [r for r in record["rows"] if r["menu_role"] == "primary"]
    assert all(
        r["nonhomogeneous_proof"] and r["decision_depth_support"] for r in primary
    )
    if kind == "which":
        unique = next(r for r in primary if r["contract"] == Contract.UNIQUE.value)
        assert unique["answer"] == "E"
        assert unique["substantive_depth_support"] is False
        assert unique["uniqueness_failure_depth"] == 10
    else:
        assert record["semantics"]["entailed"] == [matches]


def test_impossible_or_historical_coverage_gate_rejected(policy):
    policy["coverage_minima"] = [
        {
            "match": {"query_kind": "count", "listed_exhaustive_differ": True},
            "minimum_bases": 1,
        }
    ]
    with pytest.raises(ValueError, match="impossible"):
        production.plan(policy)
    policy["coverage_minima"][0]["match"] = {"contract": "all-possible"}
    with pytest.raises(ValueError, match="historical"):
        production.plan(policy)


@pytest.mark.parametrize(
    "split,depth",
    (("train", 8), ("dev", 4), ("in-range-test", 8), ("extrapolation-test", 10)),
)
def test_all_split_policy_counts_deep_unique_e_without_claiming_substantive_output(
    policy, tokenizer, split, depth
):
    from spatial.v2.entailment_retention import matches
    from spatial.v2.entailment_scene_recipes import controlled_scene
    from spatial.v2.solver import Direction

    revised = production.all_scene_diagnostic_policy(policy)
    story = controlled_scene(
        2026111094,
        depth,
        Direction.NORTH,
        3,
        story_id=f"deep-unique-{split}",
        source_family="fixed-family",
    )
    entry = historical.Entry(story, split, depth, ("which",))
    record = production.admit(
        entry, story.problems()[1], revised, tokenizer, QuestionSolver()
    )
    assert record["status"] == "accepted", record.get("reason")
    rule = next(
        r
        for r in revised["coverage_minima"]
        if r["match"].get("split") == split
        and r["match"].get("query_kind") == "which"
        and r["match"].get("contract") == Contract.UNIQUE.value
        and r["match"].get("decision_depth_support") is True
    )
    assert matches(record, rule)
    unique = next(
        r
        for r in record["rows"]
        if r["contract"] == Contract.UNIQUE.value and r["menu_role"] == "primary"
    )
    assert unique["answer"] == "E"
    assert unique["uniqueness_failure_depth"] == depth
    assert not unique["substantive_depth_support"]


def test_full_admission_and_shared_quota_accounting(policy, tokenizer):
    entry = entries()[0]
    solver = QuestionSolver()
    for problem in entry.story.problems():
        solver.clear()
        record = production.admit(entry, problem, policy, tokenizer, solver)
        assert record["status"] == "accepted", record.get("reason")
        assert record["interface"] == VERSION
        assert all(
            row["substantive_depth_support"]
            for row in record["rows"]
            if row["menu_role"] == "primary"
        )
    records = list(
        production.generate(
            entries(), policy, {"family_ids": [], "story_keys": []}, tokenizer, solver
        )
    )
    assert len(records) == 3
    assert all(r["status"] == "accepted" for r in records)
    count = next(r for r in records if r["query_kind"] == "count")
    assert count["controlled_count_value"] == 1
    assert "possible" not in count["semantics"]
    summary = production.summarize(
        [historical.summary_record(r) for r in records], policy
    )
    assert summary["interface"] == VERSION
    assert summary["accepted_bases"] == 3
    assert production.has_shortfalls(summary)


def test_exports_only_new_contracts_without_gold_in_inputs(tmp_path, policy, tokenizer):
    records = list(
        production.generate(
            entries(),
            policy,
            {"family_ids": [], "story_keys": []},
            tokenizer,
            QuestionSolver(),
        )
    )
    assert len(records) == 3
    assert all(record["status"] == "accepted" for record in records)
    production.write_exports(records, tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "references.jsonl").read_text().splitlines()
    ]
    assert rows
    assert {row["contract"] for row in rows} == {c.value for c in Contract}
    assert all(row["interface"] == VERSION for row in rows)
    inputs = [
        json.loads(line)
        for line in (tmp_path / "train-natural-inputs.jsonl").read_text().splitlines()
    ]
    assert len(inputs) == len(rows)
    assert all(
        "counterexamples" not in row and "references" not in row for row in inputs
    )
