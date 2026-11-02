"""Receipt-backed data release; original candidate manifests remain unchanged."""

import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

from spatial.v2.entailment_artifact_audit import require
from spatial.v2.entailment_generation import GenerationWorkflow, digest, read_jsonl
from spatial.v2.serialization import atomic_json

SCHEMA = "spatialentail-data-release-v1"
COUNTS = {"train": 6000, "dev": 1200, "sft-bench": 3600, "external-test": 1500}
EVIDENCE = ("review.json", "diagnostics.json", "replay.json", "profile.json")
PENDING = {
    "nuisance-shortcut-probes-and-paired-example-review",
    "actual-axolotl-loader-collator-and-gpu-validation",
    "final-v1-training-and-epoch-evaluation-integration",
}
RUNTIME_CHECKS = [
    "pinned-environment-loader-collator-model-before-first-update",
    "finite-loss-gradients-and-online-wandb-during-training",
    "both-epoch-dev-evaluation-selection-and-adapter-reload",
]


def original_sources(root, manifest, revision):
    """Resolve recorded files from Git, not from a subsequently changed checkout."""
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "full generation commit required"
    )
    names = sorted(manifest["source_sha256"])
    require(bool(names), "empty generation source inventory")
    for name in names:
        path = PurePosixPath(name)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and "\n" not in name
            and "\r" not in name,
            "invalid generation source path",
        )
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        input="".join(f"{revision}:{name}\n" for name in names).encode(),
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    sources = {}
    position = 0
    for name in names:
        end = result.index(b"\n", position)
        header = result[position:end].split()
        require(
            len(header) == 3 and header[1] == b"blob", f"missing Git source: {name}"
        )
        size = int(header[2])
        position = end + 1
        sources[name] = hashlib.sha256(result[position : position + size]).hexdigest()
        position += size + 1
    require(
        sources == manifest["source_sha256"],
        "generation commit does not match candidate",
    )
    return sources


def verify_candidate(artifacts, root, schema, revision):
    manifest = json.loads((artifacts / "manifest.json").read_text())
    sources = original_sources(root, manifest, revision)
    manifest = GenerationWorkflow.verify_saved(artifacts, sources, schema)
    require(
        manifest["mode"] == "full"
        and manifest["status"] == "candidate-audited"
        and manifest["split_counts"] == COUNTS,
        "release requires the complete full candidate",
    )
    require(
        manifest["training_ready"] is False
        and set(manifest["pending_release_checks"]) == PENDING,
        "candidate readiness fields changed or unknown checks remain",
    )
    return manifest


def validate_evidence(artifacts, reports):
    manifest_sha = digest(artifacts / "manifest.json")
    for report in reports.values():
        require(
            report["manifest_sha256"] == manifest_sha,
            "evidence belongs to another candidate",
        )
    review, diagnostics, replay = (
        reports[name] for name in ("review.json", "diagnostics.json", "replay.json")
    )
    profile = reports["profile.json"]
    require(
        profile["status"] == "passed"
        and {name: value["questions"] for name, value in profile["splits"].items()}
        == COUNTS,
        "incomplete data profile",
    )
    model = json.loads((artifacts / "manifest.json").read_text())["model"]
    for split in ("train", "dev"):
        tokens = profile["splits"][split]["tokens"]
        for style in ("natural", "symbolic"):
            require(
                tokens[f"{style}/supervised_tokens"]["min"] > 0
                and tokens[f"{style}/supervised_tokens"]["max"]
                <= model["stage1_max_new_tokens"]
                and tokens[f"{style}/training_tokens"]["max"] <= model["context_length"]
                and tokens[f"{style}/stage2_reserved_tokens"]["max"]
                <= model["context_length"],
                "profile contains invalid native token budgets",
            )
    rows = [
        row
        for split in COUNTS
        for row in read_jsonl(artifacts / f"data/{split}/references.jsonl")
    ]
    by_id = {row["id"]: row for row in rows}
    require(len(by_id) == sum(COUNTS.values()), "release reference inventory differs")
    require(
        replay["status"] == "passed"
        and replay["counts"] == COUNTS
        and replay["ordered_ids_sha256"]
        == hashlib.sha256(json.dumps([row["id"] for row in rows]).encode()).hexdigest(),
        "complete independent replay required",
    )
    require(
        review["status"] == "passed"
        and not review["unresolved_issues"]
        and review["reviewer"]["kind"] in {"human", "ai-assisted"}
        and bool(review["reviewer"]["name"]),
        "completed attributed paired review required",
    )
    ids = review["reviewed_ids"]
    require(
        len(ids) == len(set(ids)) and set(ids) <= set(by_id),
        "invalid reviewed IDs",
    )
    selection_path = artifacts / "audit/review-selection.json"
    require(
        review["review_selection_sha256"] == digest(selection_path),
        "review selection changed",
    )
    selection = json.loads(selection_path.read_text())
    required = {
        item["id"]
        for item in selection["selected"]
        if any(reason.startswith(("case/", "longest/")) for reason in item["reasons"])
    }
    require(required <= set(ids), "case-sampled or longest paired reviews missing")
    reviewed = [by_id[identity] for identity in ids]
    require(
        {
            row["construction_level"]
            for row in reviewed
            if row.get("sole_entailed_with_contingent")
        }
        == set(range(1, 11)),
        "contingent Which review must cover every level",
    )
    cells = Counter((row["query_kind"], row.get("case")) for row in reviewed)
    expected_cells = {
        (kind, case)
        for kind in ("direction", "which", "count")
        for case in ("A-D", "F", "ambiguous", "inconsistent")
    } | {("which", "multiple")}
    require(
        all(cells[cell] >= 10 for cell in expected_cells),
        "insufficient paired case review",
    )
    require(
        any(
            row["query_kind"] == "which"
            and row.get("case") == "multiple"
            and row["construction_level"] == 10
            for row in reviewed
        ),
        "deep multiple-Which review missing",
    )
    policy = diagnostics["policy"]
    require(
        diagnostics["schema"] == "spatialentail-nuisance-probe-v1"
        and policy["data"] == "train-only"
        and policy["groups"] == "family_id"
        and policy["folds"] == 5
        and policy["seed"] == 42
        and policy["classifier"]
        == "DecisionTreeClassifier(max_depth=3,min_samples_leaf=30)"
        and policy["investigate_lift"] == 0.10
        and policy["sklearn"] == "1.7.2"
        and diagnostics["train_sha256"]
        == digest(artifacts / "data/train/references.jsonl"),
        "wrong diagnostic policy or training data",
    )
    expected = {
        (kind, target, features)
        for kind in ("direction", "which", "count")
        for target in ("letter", "answer_class")
        for features in ("lengths", "names", "menus", "combined")
    }
    results = diagnostics["results"]
    require(
        len(results) == len(expected)
        and {(r["query_kind"], r["target"], r["feature_set"]) for r in results}
        == expected,
        "incomplete nuisance diagnostics",
    )
    for result in results:
        accuracy, majority = result["accuracy"], result["majority_accuracy"]
        require(
            result["n"] == 2000
            and math.isfinite(accuracy)
            and 0 <= accuracy <= 1
            and majority == (0.3 if result["target"] == "letter" else 0.6)
            and abs(result["lift"] - (accuracy - majority)) < 1e-12,
            "invalid diagnostic measurements",
        )
        require(
            accuracy - majority < policy["investigate_lift"]
            and not result["investigate"],
            "unresolved nuisance shortcut trigger",
        )
    require(
        diagnostics["status"] == "no-trigger", "diagnostic investigation remains open"
    )


def verify_release(artifacts, receipt, *, root, schema):
    artifacts, receipt = Path(artifacts).resolve(), Path(receipt).resolve()
    release = json.loads(receipt.read_text())
    require(
        release["schema"] == SCHEMA
        and release["status"] == "data-ready"
        and release["training_ready"] is True
        and release["runtime_checks"] == RUNTIME_CHECKS,
        "invalid data release",
    )
    require(
        release["generation_manifest_sha256"] == digest(artifacts / "manifest.json"),
        "release candidate changed",
    )
    manifest = verify_candidate(artifacts, root, schema, release["generation_revision"])
    require(
        set(release["evidence_sha256"]) == set(EVIDENCE), "incomplete release evidence"
    )
    reports = {}
    for name, sha in release["evidence_sha256"].items():
        path = (receipt.parent / name).resolve()
        require(path.is_relative_to(receipt.parent), "invalid evidence path")
        require(digest(path) == sha, f"release evidence changed: {name}")
        reports[name] = json.loads(path.read_text())
    validate_evidence(artifacts, reports)
    return manifest


def seal_release(config, artifacts, evidence, revision):
    """Seal checked attestations beside, never inside, the immutable candidate."""
    artifacts, evidence = artifacts.resolve(), evidence.resolve()
    require(
        not evidence.is_relative_to(artifacts),
        "release evidence must be outside candidate",
    )
    output = evidence / "release.json"
    require(not output.exists(), "release receipt already exists")
    verify_candidate(artifacts, config.root, config.manifest_schema, revision)
    reports = {name: json.loads((evidence / name).read_text()) for name in EVIDENCE}
    validate_evidence(artifacts, reports)
    release = {
        "schema": SCHEMA,
        "status": "data-ready",
        "training_ready": True,
        "generation_manifest_sha256": digest(artifacts / "manifest.json"),
        "generation_revision": revision,
        "evidence_sha256": {name: digest(evidence / name) for name in EVIDENCE},
        "runtime_checks": RUNTIME_CHECKS,
    }
    atomic_json(output, release)
    verify_release(artifacts, output, root=config.root, schema=config.manifest_schema)
    return output
