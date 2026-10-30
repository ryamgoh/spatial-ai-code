"""Production assembly checks; test fixtures are not benchmark families."""

import json
import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from spatial.v2 import composition_balance
from spatial.v2 import composition_production as production
from spatial.v2.composition_contracts import Semantics, analyze
from spatial.v2.composition_feasibility import depth_story
from spatial.v2.composition_pilot import renamed_problem, story_key
from spatial.v2.composition_selection_expansion import make_story
from spatial.v2.composition_training import REVISION
from spatial.v2.solver import Direction, SpatialSolverV2

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/spatialentail-composition-v1/production-config-v1.json"
EMPTY_EXCLUSIONS = {"family_ids": [], "story_keys": []}


@pytest.fixture
def policy():
    policy = {
        **json.loads(CONFIG.read_text()),
        "paired_omission_percent": 0,
        "coverage_minima": [
            {
                "match": {"split": "train", "semantic_case": "inconsistent"},
                "minimum_bases": 1,
            }
        ],
    }
    policy["depth_family_minima"] = {
        k: 1
        for k, n in production.quotas(policy).items()
        if n and not k.endswith("/semantic")
    }
    policy["diversity_checks"]["status"] = "approved"
    return policy


@pytest.fixture
def token_stub(monkeypatch):
    monkeypatch.setattr(
        composition_balance,
        "token_measurement",
        lambda tokenizer, row, style: {
            "id": row["id"],
            "style": style,
            "target_tokens": 10,
            "admitted": True,
        },
    )


def entry(depth=2, split="train"):
    story = make_story(
        "rival-chain", depth, 0, random.Random(42), direction=Direction.SOUTHWEST
    )
    return production.Entry(story, split, depth, ("which",))


def test_exact_quota_totals_and_depth_weights(policy):
    result = production.quotas(policy)
    for role, n in policy["base_targets"].items():
        assert sum(v for k, v in result.items() if k.startswith(role + "/")) == n
    assert result["train/which/1"] == 112
    assert result["train/which/8"] == 266
    assert result["train/which/semantic"] == 600
    assert result["extrapolation-test/which/9"] == 200
    assert result["extrapolation-test/which/10"] == 200
    assert not any("extrapolation-test" in k and "semantic" in k for k in result)
    assert production.plan(policy)["primary_views_per_style"]["train"] == 18000


@pytest.mark.parametrize("weights", ({}, {"a": 0}, {"a": -1}, {"a": True}, {"a": 0.5}))
def test_invalid_allocation_weights_rejected(weights):
    with pytest.raises(ValueError):
        production.allocate(10, weights)


def test_largest_remainder_ties_use_declared_order():
    assert production.allocate(2, {"a": 1, "b": 1, "c": 1}) == {"a": 1, "b": 1, "c": 0}


def test_draft_plan_does_not_construct_stories(policy, monkeypatch):
    monkeypatch.setattr(
        production, "schedule", lambda _: pytest.fail("must not sample")
    )
    result = production.plan(policy)
    assert result["status"] == "blocked-before-generation"
    assert len(result["blockers"]) == 3


def test_draft_cli_cannot_generate(tmp_path, monkeypatch):
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        "sys.argv", ["production", "--config", str(CONFIG), "--output", str(output)]
    )
    with pytest.raises(SystemExit) as exc:
        production.main()
    assert exc.value.code == 2
    assert not output.exists()


def test_schedule_reuses_constructors_without_inventing_family_ids(policy):
    policy["source_batches"] = [
        {
            "generator": "controlled-depth",
            "seed": seed,
            "count": 1,
            "sample": 0,
            "depth": depth,
            "split": role,
            "query_kinds": ["direction"],
        }
        for seed, depth, role in ((1, 2, "train"), (2, 9, "extrapolation-test"))
    ]
    entries = production.schedule(policy)
    assert entries == production.schedule(policy)
    assert {e.story.source_family for e in entries} == {"controlled-depth/sample-0"}
    with pytest.raises(ValueError, match="family crosses"):
        production.preflight(entries, EMPTY_EXCLUSIONS)


@pytest.mark.parametrize(
    "family",
    (
        "depth-in-range-family-0",
        "depth-extrapolation-family-0",
        "balance-depth-in-range-family-0",
    ),
)
def test_historical_depth_family_aliases_are_excluded(policy, family):
    policy["source_batches"] = [
        {
            "generator": "controlled-depth",
            "seed": 4,
            "count": 1,
            "sample": 0,
            "depth": 9,
            "split": "extrapolation-test",
            "query_kinds": ["count"],
        }
    ]
    with pytest.raises(ValueError, match="development source"):
        production.preflight(
            production.schedule(policy), {"family_ids": [family], "story_keys": []}
        )


def test_renamed_story_cannot_cross_splits():
    original = entry()
    changed, _ = renamed_problem(original.story.problems()[1])
    copy = replace(
        original.story,
        story_id="renamed",
        source_family="different-label",
        objects=changed.objects,
        premise=changed.premise,
    )
    with pytest.raises(ValueError, match="canonical story crosses"):
        production.preflight(
            [original, replace(original, story=copy, split="dev")], EMPTY_EXCLUSIONS
        )


def test_development_story_key_excluded():
    e = entry(split="dev")
    with pytest.raises(ValueError, match="development source"):
        production.preflight(
            [e],
            {
                "family_ids": [],
                "story_keys": [story_key(e.story.objects, e.story.premise)],
            },
        )


@pytest.mark.parametrize(
    "depth,split",
    ((9, "train"), (1, "extrapolation-test"), (None, "extrapolation-test")),
)
def test_invalid_depth_role_is_rejected_before_admission(depth, split):
    with pytest.raises(ValueError):
        production.preflight(
            [replace(entry(), depth=depth, split=split)], EMPTY_EXCLUSIONS
        )


def test_real_admission_preserves_three_contracts_and_deep_rival(policy, token_stub):
    e = entry()
    record = production.admit(
        e, e.story.problems()[1], policy, None, SpatialSolverV2(backend="z3")
    )
    assert record["status"] == "accepted", record.get("reason")
    assert len(record["rows"]) == 3
    assert {r["contract"] for r in record["rows"]} == production.CONTRACTS
    assert all(
        r["split"] == "train" and r["substantive_depth_support"] for r in record["rows"]
    )
    assert len({tuple(r["menu"]) for r in record["rows"]}) == 1
    assert all(set(r["references"]) == {"natural", "symbolic"} for r in record["rows"])


def test_missing_printed_depth_support_is_rejected(policy, token_stub, monkeypatch):
    e = entry()
    solver = SpatialSolverV2(backend="z3")
    real = production.admit(e, e.story.problems()[1], policy, None, solver)
    real["rows"][1]["substantive_depth_support"] = False
    monkeypatch.setattr(production, "build_attempt", lambda *a, **k: deepcopy(real))
    assert (
        production.admit(e, e.story.problems()[1], policy, None, solver)["status"]
        == "depth-rejected"
    )


def test_incidental_long_depth_cannot_enter_semantic_training_cohort(
    policy, token_stub
):
    e = replace(entry(9), depth=None)
    record = production.admit(
        e, e.story.problems()[1], policy, None, SpatialSolverV2(backend="z3")
    )
    assert record["status"] == "depth-rejected"
    assert record["measurement"]["depth"] == 9


def test_solver_failure_is_not_an_answer(policy, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unknown: timeout")

    monkeypatch.setattr(composition_balance, "analyze", fail)
    e = entry()
    records = list(production.generate([e], policy, EMPTY_EXCLUSIONS, None, None))
    assert len(records) == 1
    assert records[0]["status"] == "verification-failed"
    assert not records[0].get("rows")


def test_bounds_checked_before_solver(policy, monkeypatch):
    policy["scene_limits"]["objects"] = 1
    monkeypatch.setattr(production, "admit", lambda *a: pytest.fail("must not solve"))
    records = list(production.generate([entry()], policy, EMPTY_EXCLUSIONS, None, None))
    assert records[0]["status"] == "bounds-rejected"


def test_fixed_attempts_duplicates_and_quota_excess_are_reported(policy, token_stub):
    e = entry()
    duplicate = replace(e, story=replace(e.story, story_id="duplicate"))
    policy["base_targets"]["train"] = 1
    policy["depth_family_minima"] = None
    records = list(
        production.generate(
            [e, duplicate],
            policy,
            EMPTY_EXCLUSIONS,
            None,
            SpatialSolverV2(backend="z3"),
        )
    )
    assert [r["status"] for r in records] == ["quota-excess", "duplicate"]
    result = production.summarize(records, policy)
    assert result["attempts"] == 2
    assert result["accepted_bases"] == 0
    assert result["quota_shortfalls"]
    assert result["coverage_shortfalls"]


def test_streamed_exports_keep_gold_out_of_model_inputs(policy, token_stub, tmp_path):
    records = list(
        production.generate(
            [entry()], policy, EMPTY_EXCLUSIONS, None, SpatialSolverV2(backend="z3")
        )
    )
    assert records[0]["status"] == "accepted"
    production.write_exports(iter(records), tmp_path)
    refs = [
        json.loads(line)
        for line in (tmp_path / "references.jsonl").read_text().splitlines()
    ]
    ids = {}
    for style in ("natural", "symbolic"):
        values = [
            json.loads(line)
            for line in (tmp_path / f"train-{style}-inputs.jsonl")
            .read_text()
            .splitlines()
        ]
        ids[style] = [r["id"] for r in values]
        assert all(set(r) == {"id", "messages"} for r in values)
        assert all(
            [m["role"] for m in r["messages"]] == ["system", "user"] for r in values
        )
        assert (tmp_path / f"dev-{style}-inputs.jsonl").read_text() == ""
    assert ids["natural"] == ids["symbolic"] == [r["id"] for r in refs]
    result = production.summarize(
        [production.summary_record(r) for r in records], policy
    )
    assert result["accepted_bases"] == 1 and result["views_per_style"] == 3
    assert result["quota_shortfalls"] and result["coverage_shortfalls"]
    assert result["tokens"]["natural"]["max"] == 10
    with pytest.raises(FileExistsError):
        production.write_exports(iter(records), tmp_path)


def test_mandatory_coverage_cannot_be_removed_by_custom_minima():
    rules = production.required_coverage()
    semantic = [
        r
        for r in rules
        if "semantic_case" in r["match"] and "contains_zero" not in r["match"]
    ]
    assert len(semantic) == 33 * 3
    assert all(r["match"]["split"] != "extrapolation-test" for r in rules)
    assert any(r["match"].get("answer") == "ABCD" for r in rules)
    assert any(r["match"].get("contains_zero") for r in rules)


def test_frozen_cli_writes_only_unsealed_outputs_and_reports_shortfalls(
    policy, tmp_path, monkeypatch
):
    snapshot = (
        Path.home()
        / f".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/{REVISION}"
    )
    if not snapshot.is_dir():
        pytest.skip("pinned tokenizer not available locally")
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps(EMPTY_EXCLUSIONS))
    policy.update(
        status="frozen",
        attempt_budget=3,
        development_exclusions={
            "path": exclusions.name,
            "sha256": production.digest(exclusions),
        },
        source_batches=[
            {
                "generator": "controlled-depth",
                "seed": 11,
                "count": 1,
                "sample": 0,
                "depth": 2,
                "split": "train",
                "query_kinds": list(production.KINDS),
            }
        ],
    )
    config = tmp_path / "test-policy.json"
    config.write_text(json.dumps(policy))
    output = tmp_path / "candidate"
    monkeypatch.setattr(
        "sys.argv",
        [
            "production",
            "--config",
            str(config),
            "--output",
            str(output),
            "--tokenizer",
            str(snapshot),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        production.main()
    assert exc.value.code == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed-unsealed-candidate"
    assert manifest["manual_audit"] == "pending"
    assert manifest["training_authorized"] is False
    assert manifest["attempt_budget"] == 3
    assert all(
        production.digest(output / name) == expected
        for name, expected in manifest["output_sha256"].items()
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["accepted_bases"] == 3
    assert summary["views_per_style"] == 9
    assert summary["quota_shortfalls"]
    assert summary["family_shortfalls"]
    assert summary["direction_shortfalls"]
    assert not list(output.glob("*seal*"))


@pytest.mark.parametrize("depth,role", ((2, "train"), (9, "extrapolation-test")))
@pytest.mark.parametrize("kind", production.KINDS)
@pytest.mark.parametrize("paired_percent", (0, 100))
def test_admission_with_actual_pinned_tokenizer(
    policy, depth, role, kind, paired_percent
):
    policy["paired_omission_percent"] = paired_percent
    snapshot = (
        Path.home()
        / f".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/{REVISION}"
    )
    if not snapshot.is_dir():
        pytest.skip("pinned tokenizer not available locally")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    story = depth_story(depth, 0, random.Random(11))
    e = production.Entry(story, role, depth, (kind,))
    problem = story.problems()[production.KINDS.index(kind)]
    record = production.admit(
        e, problem, policy, tokenizer, SpatialSolverV2(backend="z3")
    )
    assert record["status"] == "accepted", record.get("reason")
    assert len(record["tokens"]) == (12 if paired_percent else 6)
    assert all(t["admitted"] and t["target_tokens"] > 0 for t in record["tokens"])


@pytest.mark.parametrize("value", (-1, 101, True, 0.5))
def test_invalid_omission_percentage_is_rejected(policy, value):
    policy["paired_omission_percent"] = value
    with pytest.raises(ValueError, match="paired_omission_percent"):
        production.plan(policy)


def test_plan_reports_extra_views_without_claiming_exact_pair_count(policy):
    policy["paired_omission_percent"] = 20
    result = production.plan(policy)
    assert result["primary_views_per_style"]["train"] == 18000
    assert result["maximum_views_per_style"]["train"] == 36000
    assert result["paired_omission_percent"] == 20


def test_family_minima_are_required_not_implicitly_approved(policy):
    policy["depth_family_minima"] = None
    assert any("family" in b for b in production.plan(policy)["blockers"])
    approved = json.loads(CONFIG.read_text())
    assert approved["status"] == "numeric-approved"
    assert approved["depth_family_minima"]["train/which/2"] == 50


@pytest.mark.parametrize("value", (0, -1, True, 0.5, 100000))
def test_invalid_family_floor_is_rejected(policy, value):
    policy["depth_family_minima"]["train/which/2"] = value
    with pytest.raises(ValueError, match="family minima"):
        production.plan(policy)


def test_family_minima_cannot_skip_or_add_a_depth_cell(policy):
    policy["depth_family_minima"].pop("train/which/2")
    with pytest.raises(ValueError, match="exactly the depth strata"):
        production.plan(policy)
    policy["depth_family_minima"]["train/which/semantic"] = 1
    with pytest.raises(ValueError, match="exactly the depth strata"):
        production.plan(policy)


@pytest.mark.parametrize("kind", production.KINDS)
def test_matched_omission_keeps_depth_base_and_contracts(policy, token_stub, kind):
    policy["paired_omission_percent"] = 100
    story = depth_story(2, 0, random.Random(11))
    e = production.Entry(story, "train", 2, (kind,))
    problem = story.problems()[production.KINDS.index(kind)]
    solver = SpatialSolverV2(backend="z3")
    result = production.admit(e, problem, policy, None, solver)
    assert result["status"] == "accepted", result.get("reason")
    assert result["paired_omission"] is True
    assert len(result["rows"]) == 6 and len(result["tokens"]) == 12
    assert {r["base_id"] for r in result["rows"]} == {result["base_id"]}
    assert {r["depth"] for r in result["rows"]} == {2}
    assert {r["family_id"] for r in result["rows"]} == {story.source_family}
    primary = [r for r in result["rows"] if r["menu_role"] == "primary"]
    omission = [r for r in result["rows"] if r["menu_role"] == "omission"]
    assert {r["contract"] for r in primary} == production.CONTRACTS
    assert {r["contract"] for r in omission} == production.CONTRACTS
    assert all(r["substantive_depth_support"] for r in primary)
    assert all(
        r["answer"] == "F" and not r["substantive_depth_support"] for r in omission
    )
    production.check_omission_pair(
        analyze(problem, solver=solver, include_witnesses=False),
        tuple(primary[0]["menu"]),
        tuple(omission[0]["menu"]),
    )
    for row in result["rows"]:
        assert production.parse_prompt(row["prompt"])[0] == problem
        assert set(row["references"]) == {"natural", "symbolic"}


@pytest.mark.parametrize(
    "possible",
    (
        frozenset(),
        frozenset({"North"}),
        frozenset({"North", "East"}),
        frozenset({"North", "East", "South", "West"}),
        frozenset(d.value for d in Direction),
    ),
)
def test_pair_inventory_handles_empty_unique_multi_and_overfull_domains(possible):
    problem = entry().story.problems()[0]
    semantics = Semantics(
        problem,
        bool(possible),
        possible,
        possible if len(possible) == 1 else frozenset(),
        "test",
        {},
    )
    kwargs = {"controlled": False, "paired_percent": 100, "solver": None}
    menus = production.select_menu(semantics, 71, **kwargs)
    assert menus == production.select_menu(semantics, 71, **kwargs)
    if 0 < len(possible) <= 4:
        assert len(menus) == 2
        production.check_omission_pair(semantics, *menus)
        assert all(a == b for a, b in zip(*menus) if a in set(menus[0]) & set(menus[1]))
    else:
        assert len(menus) == 1
    assert (
        production.select_menu(
            semantics, 71, controlled=False, paired_percent=0, solver=None
        )
        == menus[:1]
    )


def test_cannot_make_omission_without_unused_impossible_value():
    problem = entry().story.problems()[1]
    problem = replace(
        problem, query=replace(problem.query, candidates=problem.query.candidates[:4])
    )
    semantics = Semantics(
        problem,
        True,
        frozenset({problem.query.candidates[0]}),
        frozenset({problem.query.candidates[0]}),
        "test",
        {},
    )
    assert (
        len(
            production.select_menu(
                semantics, 71, controlled=False, paired_percent=100, solver=None
            )
        )
        == 1
    )


def test_omission_mutations_are_rejected():
    problem = entry().story.problems()[0]
    possible = frozenset({"North", "East"})
    semantics = Semantics(problem, True, possible, frozenset(), "test", {})
    full = ("North", "East", "South", "West")
    good = ("Northeast", "East", "South", "West")
    production.check_omission_pair(semantics, full, good)
    for partial in (
        full,
        ("Northeast", "South", "East", "West"),
        ("Northeast", "Southeast", "South", "West"),
        ("North", "East", "South", "Northeast"),
    ):
        with pytest.raises(ValueError):
            production.check_omission_pair(semantics, full, partial)


def test_omission_cannot_hide_missing_primary_depth(policy, token_stub, monkeypatch):
    policy["paired_omission_percent"] = 100
    e = entry()
    solver = SpatialSolverV2(backend="z3")
    real = production.admit(e, e.story.problems()[1], policy, None, solver)
    assert real["status"] == "accepted" and len(real["rows"]) == 6
    real["rows"][0]["substantive_depth_support"] = False
    monkeypatch.setattr(production, "build_attempt", lambda *a, **k: deepcopy(real))
    assert (
        production.admit(e, e.story.problems()[1], policy, None, solver)["status"]
        == "depth-rejected"
    )


def test_incomplete_companion_contract_bundle_fails(policy, token_stub, monkeypatch):
    policy["paired_omission_percent"] = 100
    e = entry()
    solver = SpatialSolverV2(backend="z3")
    real = production.admit(e, e.story.problems()[1], policy, None, solver)
    real["rows"].pop()
    monkeypatch.setattr(production, "build_attempt", lambda *a, **k: deepcopy(real))
    with pytest.raises(ValueError, match="three matched contracts per menu"):
        production.admit(e, e.story.problems()[1], policy, None, solver)


def test_paired_exports_count_one_base_and_family(policy, token_stub, tmp_path):
    policy["paired_omission_percent"] = 100
    records = list(
        production.generate(
            [entry()], policy, EMPTY_EXCLUSIONS, None, SpatialSolverV2(backend="z3")
        )
    )
    production.write_exports(records, tmp_path)
    summary = production.summarize(
        [production.summary_record(r) for r in records], policy
    )
    assert summary["accepted_bases"] == summary["accepted_families"] == 1
    assert summary["views_per_style"] == 6
    assert summary["paired_omission_bases"] == 1
    assert summary["family_counts_by_stratum"]["train/which/2"] == 1
    refs = [
        json.loads(line)
        for line in (tmp_path / "references.jsonl").read_text().splitlines()
    ]
    assert len(refs) == 6
    for style in ("natural", "symbolic"):
        values = [
            json.loads(line)
            for line in (tmp_path / f"train-{style}-inputs.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [r["id"] for r in refs] == [r["id"] for r in values]
        assert all(
            [m["role"] for m in r["messages"]] == ["system", "user"] for r in values
        )


def test_two_base_variants_do_not_satisfy_two_family_minimum(policy, token_stub):
    record = next(
        production.generate(
            [entry()], policy, EMPTY_EXCLUSIONS, None, SpatialSolverV2(backend="z3")
        )
    )
    second = deepcopy(record)
    second["base_id"] = "another-base"
    for row in second["rows"]:
        row["base_id"] = second["base_id"]
    policy["depth_family_minima"]["train/which/2"] = 2
    rule = {
        "match": {"split": "train", "query_kind": "which", "depth": 2},
        "minimum_bases": 2,
        "minimum_families": 2,
    }
    policy["coverage_minima"] = [rule]
    summary = production.summarize([record, second], policy)
    assert {
        "stratum": "train/which/2",
        "minimum_families": 2,
        "actual_families": 1,
    } in summary["family_shortfalls"]
    assert {**rule, "actual_bases": 2, "actual_families": 1} in summary[
        "coverage_shortfalls"
    ]
    second["family_id"] = "another-family"
    for row in second["rows"]:
        row["family_id"] = second["family_id"]
    summary = production.summarize([record, second], policy)
    assert not any(
        r["stratum"] == "train/which/2" for r in summary["family_shortfalls"]
    )
    assert not any(r["match"] == rule["match"] for r in summary["coverage_shortfalls"])


def test_direction_presence_does_not_require_every_depth_or_equal_counts():
    records = [
        {
            "split": role,
            "query_kind": kind,
            "direction": direction.value,
            "direction_group": "cardinal"
            if direction.value in production.CARDINALS
            else "diagonal",
            "family_id": f"family-{role}-{kind}",
            "measurement": {"depth": 9 if role == "extrapolation-test" else 1},
        }
        for role in production.ROLES
        for kind in production.KINDS
        for direction in Direction
    ]
    records.extend([deepcopy(records[0]) for _ in range(7)])
    report = production.direction_coverage(records)
    assert not report["direction_shortfalls"]
    assert report["direction_counts_base_denominators"]["train/direction/North"] == 8
    assert report["direction_family_counts"]["train/direction/North"] == 1
    records = [
        r
        for r in records
        if not (
            r["split"] == "dev"
            and r["query_kind"] == "which"
            and r["direction"] == "Southwest"
        )
    ]
    assert production.direction_coverage(records)["direction_shortfalls"] == [
        "dev/which/Southwest"
    ]


def test_ambiguous_direction_menu_does_not_count_as_direction_presence(
    policy, token_stub
):
    story = depth_story(1, 0, random.Random(11))
    problem = story.problems()[0]
    # Empty premises are outside composition scope; keep a relation disconnected
    # from the queried pair to make its direction genuinely unconstrained.
    from spatial.v2.solver import And, direction_constraint

    others = [
        o
        for o in story.objects
        if o not in (problem.query.target, problem.query.reference)
    ]
    premise = And((direction_constraint(others[0], others[1], Direction.NORTH),))
    story = replace(story, premise=premise, distractors=())
    problem = replace(problem, premise=premise)
    result = production.admit(
        production.Entry(story, "train", None, ("direction",)),
        problem,
        policy,
        None,
        SpatialSolverV2(backend="z3"),
    )
    assert result["status"] == "accepted", result.get("reason")
    assert len(result["semantics"]["possible"]) == 8
    assert result["direction"] is None and not result["paired_omission"]


@pytest.mark.parametrize(
    "key",
    (
        "quota_shortfalls",
        "controlled_count_shortfalls",
        "source_shortfalls",
        "family_shortfalls",
        "coverage_shortfalls",
        "direction_shortfalls",
        "unresolved_verification_failures",
    ),
)
def test_each_shortfall_independently_blocks_success(key):
    summary = {
        field: []
        for field in (
            "quota_shortfalls",
            "controlled_count_shortfalls",
            "source_shortfalls",
            "family_shortfalls",
            "coverage_shortfalls",
            "direction_shortfalls",
            "unresolved_verification_failures",
        )
    }
    summary["family_minima_configured"] = True
    assert not production.has_shortfalls(summary)
    summary[key] = ["missing"]
    assert production.has_shortfalls(summary)


def test_companion_token_failure_rejects_whole_base(policy, monkeypatch):
    policy["paired_omission_percent"] = 100
    monkeypatch.setattr(
        composition_balance,
        "token_measurement",
        lambda tokenizer, row, style: {
            "id": row["id"],
            "style": style,
            "target_tokens": 10,
            "admitted": "-m1-" not in row["id"],
        },
    )
    result = list(
        production.generate(
            [entry()], policy, EMPTY_EXCLUSIONS, None, SpatialSolverV2(backend="z3")
        )
    )
    assert result[0]["status"] == "token-rejected"
    assert len(result[0]["rows"]) == 6
    summary = production.summarize(result, policy)
    assert summary["accepted_bases"] == summary["views_per_style"] == 0


def test_family_aliases_do_not_inflate_summary_minima(policy, token_stub):
    original = next(
        production.generate(
            [entry()], policy, EMPTY_EXCLUSIONS, None, SpatialSolverV2(backend="z3")
        )
    )
    records = []
    for i, family in enumerate(
        (
            "depth-in-range-family-0",
            "depth-extrapolation-family-0",
        )
    ):
        record = deepcopy(original)
        record.update(base_id=f"base-{i}", family_id=family)
        for row in record["rows"]:
            row.update(base_id=record["base_id"], family_id=family)
        records.append(record)
    summary = production.summarize(records, policy)
    assert summary["accepted_bases"] == 2
    assert summary["accepted_families"] == 1
    assert summary["family_counts_by_stratum"]["train/which/2"] == 1


def test_no_match_selection_does_not_fabricate_an_omission():
    semantics = Semantics(
        entry().story.problems()[1],
        True,
        frozenset(),
        frozenset(),
        "test",
        {},
    )
    assert (
        len(
            production.select_menu(
                semantics,
                1,
                controlled=False,
                paired_percent=100,
                solver=None,
            )
        )
        == 1
    )


@pytest.mark.parametrize("kind", ("which", "count"))
def test_ambiguous_semantic_base_keeps_matched_control(policy, token_stub, kind):
    policy["paired_omission_percent"] = 100
    story = next(
        s
        for s in production.construct_stories(11, 2)
        if s.source_family == "contingent-1"
    )
    problem = story.problems()[production.KINDS.index(kind)]
    result = production.admit(
        production.Entry(story, "train", None, (kind,)),
        problem,
        policy,
        None,
        SpatialSolverV2(backend="z3"),
    )
    assert result["status"] == "accepted", result.get("reason")
    assert result["semantics"]["case"] == "underdetermined"
    assert result["paired_omission"] and len(result["rows"]) == 6
    assert all(r["answer"] == "E" for r in result["rows"] if r["contract"] == "single")
    assert all(
        r["answer"] == "F"
        for r in result["rows"]
        if r["contract"] == "all-possible" and r["menu_role"] == "omission"
    )


def test_approved_source_and_family_allocations():
    policy = json.loads(CONFIG.read_text())
    result = production.plan(policy)
    expected = {
        "train": (4200, 900, 600, 300),
        "dev": (840, 180, 120, 60),
        "in-range-test": (1680, 360, 240, 120),
        "extrapolation-test": (1200, 0, 0, 0),
    }
    assert result["source_totals_by_split"] == {
        role: dict(zip(production.SOURCES, values)) for role, values in expected.items()
    }
    for role, floor in zip(production.ROLES, (50, 10, 20, 50)):
        values = {
            n
            for k, n in result["depth_family_minima"].items()
            if k.startswith(role + "/")
        }
        assert values == {floor}
    assert len(result["depth_family_minima"]) == 78
    assert policy["paired_omission_percent"] == 20
    assert result["status"] == "blocked-before-generation"
    assert policy["approval"]["production_freeze"] is False
    assert policy["approval"]["training_authorized"] is False
    assert not any("Numeric policy has not" in b for b in result["blockers"])


@pytest.mark.parametrize(
    "mix",
    (
        None,
        {},
        {"controlled": 70, "random-world": 15, "random-atoms": 10, "targeted": 6},
        {"controlled": 60, "random-world": 20, "random-atoms": 10, "targeted": 10},
        {"controlled": 70, "random-world": 19, "random-atoms": 10, "targeted": True},
        {"controlled": 70, "random-world": 20, "random-atoms": 10, "targeted": 0},
    ),
)
def test_invalid_source_mix_cannot_be_frozen(policy, mix):
    policy["source_mix_percent"] = mix
    with pytest.raises(ValueError, match="source mix"):
        production.plan(policy)


@pytest.mark.parametrize("total", (1, 7, 31, 60, 100))
def test_source_rounding_preserves_query_and_cohort_totals(policy, total):
    policy["base_targets"]["train"] = total
    source_targets = production.source_quotas(policy)
    base_targets = production.quotas(policy)
    for kind in production.KINDS:
        prefix = f"train/{kind}/"
        assert sum(n for k, n in source_targets.items() if k.startswith(prefix)) == sum(
            n for k, n in base_targets.items() if k.startswith(prefix)
        )
        assert sum(
            n
            for k, n in source_targets.items()
            if k.startswith(prefix) and not k.endswith("/controlled")
        ) == base_targets.get(prefix + "semantic", 0)


def test_source_caps_do_not_refill_from_another_recipe(policy, token_stub):
    policy["base_targets"]["train"] = 60
    policy["paired_omission_percent"] = 100
    policy["depth_family_minima"] = {
        k: 1
        for k, n in production.quotas(policy).items()
        if n and not k.endswith("/semantic")
    }
    entries = [replace(entry(depth), depth=None) for depth in (2, 3)]
    records = list(
        production.generate(
            entries, policy, EMPTY_EXCLUSIONS, None, SpatialSolverV2(backend="z3")
        )
    )
    assert [r["status"] for r in records] == ["accepted", "source-quota-excess"]
    assert all(r["source"] == "targeted" for r in records)
    summary = production.summarize(
        [production.summary_record(r) for r in records], policy
    )
    assert summary["source_counts_base_denominators"]["train/which/targeted"] == 1
    assert summary["source_shortfalls"]["train/which/random-world"] == 3
    assert summary["source_shortfalls"]["train/which/random-atoms"] == 2
    assert summary["quota_shortfalls"]["train/which/semantic"] == 5
    assert summary["accepted_bases"] == 1 and summary["views_per_style"] == 6
    assert all(row["source"] == "targeted" for row in records[0]["rows"])


def test_source_excess_does_not_consume_other_source_capacity(policy, monkeypatch):
    policy["base_targets"]["train"] = 60
    sources = [replace(entry(d), depth=None) for d in (2, 3, 4)]
    sources[-1] = replace(
        sources[-1], story=replace(sources[-1].story, recipe="random-world")
    )
    monkeypatch.setattr(production, "admit", lambda *a: {"status": "accepted"})
    records = list(production.generate(sources, policy, EMPTY_EXCLUSIONS, None, None))
    assert [r["status"] for r in records] == [
        "accepted",
        "source-quota-excess",
        "accepted",
    ]


def test_all_legacy_targeted_families_are_development_lineages(policy):
    from spatial.v2.composition_pilot import RECIPES

    excluded = {f"{recipe}-{sample}" for recipe in RECIPES for sample in range(5)}
    stories = production.construct_stories(20261029, 5)
    assert len(excluded) == 45
    assert {story.source_family for story in stories} == excluded
    assert {
        story.source_family for story in production.construct_stories(999, 5)
    } == excluded
    # A new seed cannot make any of this constructor's family IDs held-out.
    entries = [production.Entry(story, "dev", None, ("which",)) for story in stories]
    with pytest.raises(ValueError, match="development source"):
        production.preflight(
            entries, {"family_ids": sorted(excluded), "story_keys": []}
        )
