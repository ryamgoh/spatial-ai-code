"""Proof and distribution checks using existing development scaffold seeds."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from spatial.v2 import composition_balance
from spatial.v2 import composition_production as production
from spatial.v2.composition_contracts import analyze
from spatial.v2.composition_depth import measure_depth
from spatial.v2.composition_feasibility import verify_transformations
from spatial.v2.composition_scene_sampler import _support, sample_scene
from spatial.v2.composition_targeted_sampler import sample_targeted
from spatial.v2.composition_training import REVISION
from spatial.v2.solver import (
    And,
    Direction,
    DirectionQuery,
    SpatialProblem,
    SpatialSolverV2,
)

ROOT = Path(__file__).resolve().parents[3]
EMPTY_EXCLUSIONS = {"family_ids": [], "story_keys": []}
DIRECTIONS = tuple(Direction)
PATTERNS = [
    *(("axis-split", d) for d in Direction),
    *(("mixed-chain", d) for d in DIRECTIONS[1::2]),
]


@pytest.fixture
def policy():
    p = json.loads(
        (
            ROOT / "experiments/spatialentail-composition-v1/production-config-v1.json"
        ).read_text()
    )
    p["paired_omission_percent"] = 100
    return p


@pytest.fixture
def token_stub(monkeypatch):
    monkeypatch.setattr(
        composition_balance,
        "token_measurement",
        lambda t, r, style: {
            "id": r["id"],
            "style": style,
            "target_tokens": 10,
            "admitted": True,
        },
    )


@pytest.fixture(scope="module")
def tokenizer():
    path = (
        Path.home()
        / f".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/{REVISION}"
    )
    if not path.is_dir():
        pytest.skip("pinned tokenizer unavailable")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.mark.parametrize("pattern,direction", PATTERNS)
@pytest.mark.parametrize("depth", (2, 5, 8, 9, 10))
def test_varied_proof_semantics_and_measured_depth(pattern, direction, depth):
    story = sample_scene(3, depth, direction=direction, proof_pattern=pattern)
    assert story.source_family == sample_scene(3, 1).source_family
    assert len(story.objects) <= 31 and len(story.premise.operands) <= 32
    solver = SpatialSolverV2(backend="z3")
    for problem, expected in zip(
        story.problems(),
        (
            {direction.value},
            {story.target},
            {1},
        ),
    ):
        semantics = analyze(problem, solver=solver, include_witnesses=False)
        assert (
            semantics.consistent and semantics.possible == semantics.certain == expected
        )
        measurement = measure_depth(problem, solver=solver)
        assert measurement["depth"] == depth
    verify_transformations(story, story.problems()[2], semantics, solver)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_separate_axis_paths_are_both_necessary(direction):
    objects, atoms = _support(5, direction, "axis-split")
    problem = SpatialProblem(
        ("O", "C0", *objects), And(tuple(atoms)), DirectionQuery("C0", "O")
    )
    assert analyze(problem, include_witnesses=False).possible == {direction.value}
    for path in (atoms[:5], atoms[5:]):
        semantics = analyze(
            replace(problem, premise=And(tuple(path))), include_witnesses=False
        )
        assert semantics.consistent
        assert direction.value in semantics.possible and len(semantics.possible) > 1


@pytest.mark.parametrize("pattern,direction", PATTERNS)
@pytest.mark.parametrize("kind", production.KINDS)
def test_actual_depth_ten_readable_admission(
    policy, tokenizer, pattern, direction, kind
):
    story = sample_scene(3, 10, direction=direction, proof_pattern=pattern)
    e = production.Entry(story, "extrapolation-test", 10, (kind,))
    r = production.admit(
        e,
        story.problems()[production.KINDS.index(kind)],
        policy,
        tokenizer,
        SpatialSolverV2(backend="z3"),
    )
    assert r["status"] == "accepted", r.get("reason")
    assert r["proof_pattern"] == pattern
    assert all(row["nonhomogeneous_proof"] for row in r["rows"])
    assert all(
        row["substantive_depth_support"]
        for row in r["rows"]
        if row["menu_role"] == "primary"
    )
    assert all(t["admitted"] for t in r["tokens"])
    assert not any(row["contracts_differ"] for row in r["rows"])


@pytest.mark.parametrize("pattern", ("mixed-chain", "axis-split"))
@pytest.mark.parametrize("matches", (0, 2, 4))
def test_varied_count_values_preserve_depth(policy, tokenizer, pattern, matches):
    story = sample_scene(
        3, 10, direction=Direction.NORTHEAST, proof_pattern=pattern, matches=matches
    )
    r = production.admit(
        production.Entry(story, "extrapolation-test", 10, ("count",)),
        story.problems()[2],
        policy,
        tokenizer,
        SpatialSolverV2(backend="z3"),
    )
    assert r["status"] == "accepted", r.get("reason")
    assert r["semantics"]["possible"] == [matches]


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("depth", (2, 8, 10))
def test_zero_count_exclusion_requires_long_axis(direction, depth):
    story = sample_scene(
        3, depth, direction=direction, proof_pattern="axis-split", matches=0
    )
    solver = SpatialSolverV2(backend="z3")
    problem = story.problems()[2]
    assert analyze(problem, solver=solver, include_witnesses=False).possible == {0}
    assert measure_depth(problem, solver=solver)["depth"] == depth


@pytest.mark.parametrize(
    "kwargs",
    (
        {"proof_pattern": "unknown"},
        {"proof_pattern": "axis-split", "depth": 1},
        {"proof_pattern": "mixed-chain", "depth": 1},
        {"proof_pattern": "mixed-chain", "direction": Direction.NORTH},
    ),
)
def test_unsupported_proof_requests_reject(kwargs):
    with pytest.raises(ValueError):
        sample_scene(
            **{"seed": 3, "depth": 2, "direction": Direction.NORTHEAST, **kwargs}
        )


def test_proof_variants_cannot_cross_splits():
    entries = [
        production.Entry(
            sample_scene(3, 3, proof_pattern=pattern, direction=Direction.NORTHEAST),
            role,
            3,
            ("direction",),
        )
        for pattern, role in (("homogeneous", "train"), ("axis-split", "dev"))
    ]
    with pytest.raises(ValueError, match="family crosses"):
        production.preflight(entries, EMPTY_EXCLUSIONS)


def test_bounds_and_default_compatibility_across_development_scaffolds():
    for seed in range(32):
        for depth in (2, 8, 10):
            for pattern in ("homogeneous", "mixed-chain", "axis-split"):
                story = sample_scene(
                    seed, depth, direction=Direction.NORTHEAST, proof_pattern=pattern
                )
                assert len(story.objects) <= 31
                assert len(story.premise.operands) <= 32
                if pattern == "homogeneous":
                    assert story == sample_scene(
                        seed, depth, direction=Direction.NORTHEAST
                    )


def test_count_quotas_preserve_depth_totals_and_even_weights(policy):
    quotas = production.count_quotas(policy)
    for cell, total in production.quotas(policy).items():
        if "/count/" not in cell or cell.endswith("/semantic"):
            continue
        values = [quotas[f"{cell}/{n}"] for n in range(5)]
        assert sum(values) == total and max(values) - min(values) <= 1


@pytest.mark.parametrize(
    "weights",
    (None, {}, {"1": 1}, {"9": 1, "0": 1}, {"0": True, "1": 1}, {"0": 0, "1": 1}),
)
def test_invalid_count_policy_rejects(policy, weights):
    policy["controlled_count_weights"] = weights
    with pytest.raises(ValueError):
        production.plan(policy)


def test_count_schedule_cannot_silently_use_one(policy):
    batch = {
        "generator": "sampled-forest",
        "seed": 3,
        "count": 1,
        "depth": 2,
        "split": "train",
        "query_kinds": ["count"],
        "proof_pattern": "axis-split",
    }
    policy["source_batches"] = [batch]
    with pytest.raises(ValueError, match="explicit matches"):
        production.schedule(policy)
    batch["matches"] = 2
    entry = production.schedule(policy)[0]
    assert entry.story.recipe.endswith("/axis-split")
    assert analyze(entry.story.problems()[2], include_witnesses=False).possible == {2}


def test_count_cap_uses_verified_value_and_does_not_consume_other_capacity(
    policy, monkeypatch
):
    policy["base_targets"]["train"] = 600
    # At train/count/depth-2 this gives two slots for zero, two for one.
    quotas = production.count_quotas(policy)
    maximum = quotas["train/count/2/1"]
    entries = [
        production.Entry(
            sample_scene(seed, 2, direction=Direction.NORTHEAST),
            "train",
            2,
            ("count",),
        )
        for seed in range(maximum + 2)
    ]
    calls = iter([1] * (maximum + 1) + [0])
    monkeypatch.setattr(
        production,
        "admit",
        lambda *a: {
            "status": "accepted",
            "semantics": {"possible": [next(calls)]},
        },
    )
    records = list(production.generate(entries, policy, EMPTY_EXCLUSIONS, None, None))
    assert [r["status"] for r in records] == [
        *["accepted"] * maximum,
        "count-quota-excess",
        "accepted",
    ]
    assert records[-1]["controlled_count_value"] == 0


def test_difference_is_within_menu_not_across_omission(policy, token_stub):
    records = []
    story = sample_scene(3, 2, direction=Direction.NORTHEAST)
    entries = [production.Entry(story, "train", 2, ("which",))]
    records.extend(
        production.generate(
            entries, policy, EMPTY_EXCLUSIONS, None, SpatialSolverV2(backend="z3")
        )
    )
    story = sample_targeted(4, "multiple", matches=2)
    entries = [production.Entry(story, "train", None, ("which",))]
    records.extend(
        production.generate(
            entries, policy, EMPTY_EXCLUSIONS, None, SpatialSolverV2(backend="z3")
        )
    )
    assert all(r["status"] == "accepted" for r in records)
    # Singleton: primary A-D and omission F is NOT a contract distinction.
    assert not any(row["contracts_differ"] for row in records[0]["rows"])
    assert all(row["contracts_differ"] for row in records[1]["rows"])
    assert not any(
        row["listed_exhaustive_differ"]
        for row in records[1]["rows"]
        if row["menu_role"] == "primary"
    )
    assert all(
        row["listed_exhaustive_differ"]
        for row in records[1]["rows"]
        if row["menu_role"] == "omission"
    )
    summary = production.summarize(
        [production.summary_record(r) for r in records], policy
    )
    counts = summary["contract_differences_base_denominators"]["train/which"]
    assert counts == {
        "bases": 2,
        "contracts_differ": 1,
        **{flag: 1 for flag in production.CONTRACT_PAIRS},
    }
    # Forged flags cannot satisfy gates: summary recomputes from actual golds.
    for row in records[0]["rows"]:
        row.update({flag: True for flag in production.CONTRACT_PAIRS})
    assert (
        production.summarize(records, policy)["contract_differences_base_denominators"][
            "train/which"
        ]
        == counts
    )


def test_presence_only_is_not_enough_for_new_floors(policy):
    rules = production.diversity_coverage(policy)
    for role, expected in (("train", 120), ("dev", 24), ("in-range-test", 48)):
        rule = next(
            r
            for r in rules
            if r["match"]
            == {
                "split": role,
                "query_kind": "count",
                "listed_exhaustive_differ": True,
            }
        )
        assert rule["minimum_bases"] == expected
        assert rule["minimum_families"] > 1
    assert not any(r["match"].get("depth") == 1 for r in rules)
    assert any(r["match"].get("nonhomogeneous_proof") for r in rules)
    policy["diversity_checks"]["status"] = "draft"
    assert any("await approval" in b for b in production.plan(policy)["blockers"])
    assert (
        next(
            r
            for r in rules
            if r["match"]
            == {
                "split": "train",
                "query_kind": "direction",
                "semantic_case": "underdetermined",
            }
        )["minimum_bases"]
        == 30
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("contract_difference_semantic_percent", 0),
        ("contract_difference_semantic_percent", True),
        ("contract_difference_families", {}),
        ("nonhomogeneous_controlled_percent", 101),
        ("semantic_minimum_percent", True),
    ),
)
def test_invalid_diversity_settings_reject(policy, field, value):
    policy["diversity_checks"][field] = value
    with pytest.raises(ValueError):
        production.plan(policy)


def test_count_shortfall_is_independent_failure_gate(policy, token_stub):
    summary = production.summarize([], policy)
    for key in (
        "quota_shortfalls",
        "source_shortfalls",
        "family_shortfalls",
        "coverage_shortfalls",
        "direction_shortfalls",
        "unresolved_verification_failures",
    ):
        summary[key] = []
    assert production.has_shortfalls(summary)
    summary["controlled_count_shortfalls"] = {}
    assert not production.has_shortfalls(summary)
