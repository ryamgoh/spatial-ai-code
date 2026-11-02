"""Release receipts authorize immutable candidates, not unchecked readiness flags."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
import generation_policy as policy

from spatial.v2 import entailment_release as release
from spatial.v2 import entailment_sft as sft
from spatial.v2.entailment_generation import payload_hashes
from spatial.v2.serialization import atomic_json


@pytest.fixture
def candidate(tmp_path):
    """Small-content fixture with full counts; attestations are test doubles."""
    artifacts, evidence = tmp_path / "artifacts", tmp_path / "evidence"
    for folder in ("data", "audit", "inputs", "tokenizers"):
        (artifacts / folder).mkdir(parents=True)
    evidence.mkdir()
    cfg = policy.workflow_config()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source = subprocess.check_output(
        ["git", "show", f"{revision}:CONTEXT.md"], cwd=ROOT
    )
    case_rows = []
    for kind in ("direction", "which", "count"):
        for case in (
            "A-D",
            "F",
            "ambiguous",
            "inconsistent",
            *(("multiple",) if kind == "which" else ()),
        ):
            for level in range(1, 11):
                case_rows.append(
                    {
                        "id": f"review-{kind}-{case}-{level}",
                        "query_kind": kind,
                        "case": case,
                        "construction_level": level,
                        "sole_entailed_with_contingent": kind == "which"
                        and case == "A-D",
                    }
                )
    ids = []
    for split, count in release.COUNTS.items():
        rows = list(case_rows) if split == "train" else []
        rows += [
            {
                "id": f"{split}-{i}",
                "query_kind": "direction",
                "case": "A-D",
                "construction_level": 1,
            }
            for i in range(count - len(rows))
        ]
        directory = artifacts / "data" / split
        directory.mkdir()
        (directory / "references.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        ids.extend(row["id"] for row in rows)
    selection = artifacts / "audit/review-selection.json"
    atomic_json(
        selection,
        {
            "selected": [
                {
                    "id": row["id"],
                    "reasons": [f"case/{row['query_kind']}/{row['case']}"],
                }
                for row in case_rows
            ],
        },
    )
    manifest = {
        "schema": cfg.manifest_schema,
        "status": "candidate-audited",
        "mode": "full",
        "model": policy.MODEL,
        "training_ready": False,
        "pending_release_checks": sorted(release.PENDING),
        "source_sha256": {"CONTEXT.md": hashlib.sha256(source).hexdigest()},
        "payload_sha256": payload_hashes(artifacts),
        "split_counts": release.COUNTS,
    }
    atomic_json(artifacts / "manifest.json", manifest)
    sha = release.digest(artifacts / "manifest.json")
    atomic_json(
        artifacts / "generation.json",
        {"status": "candidate-audited", "manifest_sha256": sha},
    )
    atomic_json(
        evidence / "review.json",
        {
            "status": "passed",
            "manifest_sha256": sha,
            "reviewer": {"kind": "ai-assisted", "name": "test fixture"},
            "unresolved_issues": [],
            "reviewed_ids": [row["id"] for row in case_rows],
            "review_selection_sha256": release.digest(selection),
        },
    )
    results = []
    for kind in ("direction", "which", "count"):
        for target in ("letter", "answer_class"):
            for features in ("lengths", "names", "menus", "combined"):
                majority = 0.3 if target == "letter" else 0.6
                results.append(
                    {
                        "query_kind": kind,
                        "target": target,
                        "feature_set": features,
                        "n": 2000,
                        "accuracy": majority,
                        "majority_accuracy": majority,
                        "lift": 0,
                        "investigate": False,
                    }
                )
    atomic_json(
        evidence / "diagnostics.json",
        {
            "schema": "spatialentail-nuisance-probe-v1",
            "manifest_sha256": sha,
            "status": "no-trigger",
            "train_sha256": release.digest(artifacts / "data/train/references.jsonl"),
            "policy": {
                "data": "train-only",
                "groups": "family_id",
                "folds": 5,
                "seed": 42,
                "classifier": "DecisionTreeClassifier(max_depth=3,min_samples_leaf=30)",
                "investigate_lift": 0.10,
                "sklearn": "1.7.2",
            },
            "results": results,
        },
    )
    atomic_json(
        evidence / "replay.json",
        {
            "manifest_sha256": sha,
            "status": "passed",
            "counts": release.COUNTS,
            "ordered_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        },
    )
    atomic_json(
        evidence / "profile.json",
        {
            "manifest_sha256": sha,
            "status": "passed",
            "splits": {
                split: {
                    "questions": count,
                    "tokens": {
                        f"{style}/{field}": {"min": 1, "max": 100}
                        for style in ("natural", "symbolic")
                        for field in (
                            "supervised_tokens",
                            "training_tokens",
                            "stage2_reserved_tokens",
                        )
                    },
                }
                for split, count in release.COUNTS.items()
            },
        },
    )
    return cfg, artifacts, evidence, revision


def test_seal_preserves_candidate_and_enables_four_configs(candidate, tmp_path):
    cfg, artifacts, evidence, revision = candidate
    before = {
        path: path.read_bytes() for path in artifacts.rglob("*") if path.is_file()
    }
    receipt = release.seal_release(cfg, artifacts, evidence, revision)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (
        json.loads((artifacts / "manifest.json").read_text())["training_ready"] is False
    )
    campaign = tmp_path / "campaign"
    sft.prepare(cfg, artifacts, campaign, release=receipt)
    for style in sft.STYLES:
        for seed in sft.SEEDS:
            meta = sft.verify_run(campaign / sft.run_key(style, seed))
            assert meta["release_sha256"] == release.digest(receipt)
    with pytest.raises(ValueError, match="already exists"):
        release.seal_release(cfg, artifacts, evidence, revision)
    receipt.write_text(receipt.read_text() + "\n")
    with pytest.raises(ValueError, match="receipt changed"):
        sft.verify_run(campaign / "natural-seed-42")


@pytest.mark.parametrize(
    "fault",
    (
        "missing-review",
        "open-issue",
        "diagnostic-trigger",
        "other-candidate",
        "incomplete-replay",
        "token-overflow",
    ),
)
def test_incomplete_evidence_cannot_be_sealed(candidate, fault):
    cfg, artifacts, evidence, revision = candidate
    name = {
        "missing-review": "review.json",
        "open-issue": "review.json",
        "diagnostic-trigger": "diagnostics.json",
        "other-candidate": "diagnostics.json",
        "incomplete-replay": "replay.json",
        "token-overflow": "profile.json",
    }[fault]
    path = evidence / name
    report = json.loads(path.read_text())
    if fault == "missing-review":
        report["reviewed_ids"].pop()
    elif fault == "open-issue":
        report["unresolved_issues"] = ["unchecked rationale"]
    elif fault == "diagnostic-trigger":
        report["results"][0].update(accuracy=0.8, lift=0.5, investigate=True)
    elif fault == "other-candidate":
        report["manifest_sha256"] = "wrong"
    elif fault == "incomplete-replay":
        report["counts"].pop("external-test")
    else:
        report["splits"]["train"]["tokens"]["natural/training_tokens"]["max"] = 100000
    atomic_json(path, report)
    with pytest.raises(ValueError):
        release.seal_release(cfg, artifacts, evidence, revision)
    assert not (evidence / "release.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_ready", True),
        ("mode", "smoke"),
        ("pending_release_checks", ["unknown-check"]),
    ],
)
def test_manual_readiness_or_incomplete_candidate_rejected(candidate, field, value):
    cfg, artifacts, evidence, revision = candidate
    path = artifacts / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    atomic_json(path, manifest)
    atomic_json(
        artifacts / "generation.json",
        {
            "status": "candidate-audited",
            "manifest_sha256": release.digest(path),
        },
    )
    with pytest.raises(ValueError):
        release.seal_release(cfg, artifacts, evidence, revision)


def test_receipt_is_portable_and_detects_evidence_and_payload_tampering(
    candidate, tmp_path
):
    cfg, artifacts, evidence, revision = candidate
    receipt = release.seal_release(cfg, artifacts, evidence, revision)
    copied = tmp_path / "portable"
    shutil.copytree(evidence, copied)
    release.verify_release(
        artifacts, copied / receipt.name, root=cfg.root, schema=cfg.manifest_schema
    )
    (copied / "review.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        release.verify_release(
            artifacts, copied / receipt.name, root=cfg.root, schema=cfg.manifest_schema
        )
    with (artifacts / "data/train/references.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="payload changed"):
        release.verify_release(
            artifacts, receipt, root=cfg.root, schema=cfg.manifest_schema
        )


def test_wrong_generation_commit_rejected(candidate):
    cfg, artifacts, evidence, _ = candidate
    with pytest.raises(ValueError, match="missing Git source"):
        release.seal_release(cfg, artifacts, evidence, "0" * 40)


@pytest.mark.parametrize("command", ("release", "verify-release"))
@pytest.mark.parametrize("invalid", (False, True))
def test_release_cli_verifies_once_and_propagates_failures(
    candidate, monkeypatch, capsys, command, invalid
):
    from spatial.v2.entailment_generation import main

    cfg, artifacts, evidence, revision = candidate
    arguments = [command, "--run-dir", str(artifacts)]
    if command == "release":
        arguments += ["--evidence", str(evidence), "--generation-revision", revision]
    else:
        receipt = release.seal_release(cfg, artifacts, evidence, revision)
        arguments += ["--release", str(receipt)]
    if invalid:
        (evidence / "review.json").write_text("{}")
        with pytest.raises((ValueError, KeyError)):
            main(cfg, arguments)
        assert capsys.readouterr().out == ""
        return

    verify = release.verify_release
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return verify(*args, **kwargs)

    monkeypatch.setattr(release, "verify_release", counted)
    main(cfg, arguments)
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "status": "data-ready",
        "release": str(evidence / "release.json"),
        "runtime_checks": "required-inside-SFT",
    }
    assert len(calls) == 1


def test_nuisance_features_ignore_gold_and_hidden_metadata():
    from spatial.v2.entailment_dataset import build_attempt
    from spatial.v2.entailment_generation import GenerationWorkflow
    from spatial.v2.entailment_release_checks import nuisance_features
    from spatial.v2.entailment_scene_protocol import SHORT_VERSION

    slot = next(s for s in policy.schedule() if s.level == 1 and s.kind == "direction")
    story, problem, _, _ = GenerationWorkflow(policy.workflow_config()).construct(
        slot, 0
    )
    record = build_attempt(
        story,
        problem,
        slot.requested_depth,
        None,
        omission_percent=0,
        prompt_version=SHORT_VERSION,
        measure_tokens=lambda *_: {"admitted": True},
    )
    assert record["status"] == "accepted"
    row = record["rows"][0]
    expected = nuisance_features(row, 500)
    row.update(
        answer="E",
        case="inconsistent",
        construction_level=999,
        direction="North",
        references={"natural": "gold", "symbolic": "gold"},
        id="gold-ID",
    )
    assert nuisance_features(row, 500) == expected
    assert set(expected) == {"lengths", "names", "menus", "combined"}
