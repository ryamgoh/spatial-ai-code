"""Production audit fixtures preserve actual paired admission and overlap checks."""

import json
from pathlib import Path

import pytest
import yaml

from spatial.v2.study.build import build_release
from spatial.v2.study.jobs import prepare_jobs
from spatial.v2.study.seal import audit_release
from spatial.v2.tests.test_context_budget import FixtureTokenizer
from spatial.v2.tests.test_study_release import config


def production_fixture(tmp_path):
    path = config(tmp_path, engineering=False)
    source = yaml.safe_load(path.read_text())
    source["roles"]["bench-core"] = [{"cell": "DIR-ARC-UNI", "count": 1}]
    source["roles"]["bench-comp"] = [
        {
            "stratum": "comp",
            "count": 1,
            "policy": {
                "semantic_shape": "unique",
                "composition": "mp-trc",
                "num_entities": 6,
                "num_premises": 10,
            },
        }
    ]
    source["roles"]["bench-composition"] = [
        {
            "stratum": "held",
            "count": 1,
            "policy": {
                "semantic_shape": "unique",
                "composition": "mp-dje-trc",
                "num_entities": 6,
                "num_premises": 10,
            },
        }
    ]
    source["roles"]["bench-depth"] = [
        {
            "stratum": "count-depth-3",
            "count": 1,
            "policy": {
                "query_kind": "count",
                "semantic_shape": "unique",
                "num_entities": 9,
                "num_premises": 12,
                "omit_direct_query_relation": True,
                "min_membership_depth": 3,
                "max_membership_depth": 3,
                "membership_candidate_limit": 3,
                "measure_support": False,
                "oracle_only": True,
            },
        }
    ]
    source["attempts_per_base"] = 20
    path.write_text(yaml.safe_dump(source))
    release = tmp_path / "release"
    build_release(path, release, tokenizer=FixtureTokenizer())
    return release


def test_production_audit_checks_all_roles_and_hashes(tmp_path):
    release = production_fixture(tmp_path)
    audit = audit_release(release)
    assert audit["overlap"] == 0
    assert audit["roles"]["train"]["bases"] == 2
    with (release / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        audit_release(release)


def test_six_jobs_have_no_benchmark_training_and_transferable_paths(tmp_path):
    release = production_fixture(tmp_path)
    audit = audit_release(release)
    audit_path = release / "release-audit.json"
    audit_path.write_text(json.dumps(audit))
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    import hashlib

    manifest.update(
        status="sealed",
        release_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="quality index"):
        prepare_jobs(release, tmp_path / "jobs", repo_root=tmp_path)


@pytest.mark.parametrize("use_appeal", [False, True])
def test_seal_validates_actual_judge_payloads_then_prepares_jobs(tmp_path, use_appeal):
    from spatial.v2.hitl.review import (
        REVIEW_CRITERIA,
        build_review_packet,
        write_review_packet,
    )
    from spatial.v2.study.seal import seal_release

    release = production_fixture(tmp_path)
    manifest = json.loads((release / "manifest.json").read_text())
    review = tmp_path / "review"
    review.mkdir()
    batches = []
    rejected_packet = None
    for role in manifest["quotas"]:
        if role == "bench-depth":
            continue
        rows = [
            json.loads(line)
            for line in (release / (role + ".jsonl")).read_text().splitlines()
        ]
        natural = [
            r
            for r in rows
            if r["metadata"]["trace_format"] == "natural"
            and r["metadata"]["supervision_arm"] == "checked-trace"
        ]
        packet, crosswalk = build_review_packet(
            natural,
            packet_id=role,
            registry_fingerprint=manifest["config_sha256"],
            rubric_version="quality-judge-v1",
            minimum_per_cell=1,
            sample_fraction=1,
        )
        write_review_packet(
            packet,
            crosswalk,
            review / (role + ".json"),
            review / (role + "-crosswalk.json"),
        )
        for model, position, key in [
            ("DeepSeek-V4-Pro", 1, "deepseek"),
            ("GPT-5.5", 2, "gpt"),
        ]:
            receipt = {
                "schema": "spatialentail-judge-run-v1",
                "provider": "trae",
                "model": model,
                "run_id": role + key,
                "packet_fingerprint": packet.fingerprint,
                "judge_prompt_version": "quality-judge-v1",
                "sequence_position": position,
                "isolated_context": True,
                "decisions": [
                    {
                        "review_item_id": i.id,
                        "payload_hash": i.payload_hash,
                        "verdict": "approve",
                        "reason_codes": [],
                        "criteria": dict.fromkeys(REVIEW_CRITERIA, True),
                    }
                    for i in packet.items
                ],
            }
            if use_appeal and rejected_packet is None:
                receipt["decisions"][0].update(
                    verdict="reject",
                    reason_codes=["TRACE_UNREADABLE"],
                    criteria=dict.fromkeys(REVIEW_CRITERIA, False),
                )
            (review / (role + "-" + key + ".json")).write_text(json.dumps(receipt))
        if use_appeal and rejected_packet is None:
            from dataclasses import asdict

            from spatial.v2.hitl.review import HumanAppeal

            appeal = HumanAppeal.create(
                packet, packet.items[0].id, "approve", reviewer="user"
            )
            (review / (role + "-appeals.jsonl")).write_text(
                json.dumps(asdict(appeal)) + "\n"
            )
            rejected_packet = role
        batches.append(
            {
                "packet": role + ".json",
                "crosswalk": role + "-crosswalk.json",
                "deepseek": role + "-deepseek.json",
                "gpt": role + "-gpt.json",
            }
        )
    if use_appeal:
        batches[0]["appeals"] = rejected_packet + "-appeals.jsonl"
    index = {
        "schema": "spatialentail-release-review-v1",
        "release_artifacts_sha256": {
            r: v["sha256"] for r, v in manifest["artifacts"].items()
        },
        "minimum_review_items": 6,
        "batches": batches,
    }
    (review / "index.json").write_text(json.dumps(index))
    if use_appeal:
        appeal_path = review / (rejected_packet + "-appeals.jsonl")
        original = appeal_path.read_text()
        damaged = json.loads(original)
        damaged["payload_hash"] = "wrong"
        appeal_path.write_text(json.dumps(damaged) + "\n")
        with pytest.raises(ValueError, match="payload"):
            seal_release(release, review / "index.json")
        appeal_path.write_text(original)
    audit = seal_release(release, review / "index.json")
    assert audit["quality"]["counts"] == {"accept": 6}
    assert json.loads((release / "manifest.json").read_text())["status"] == "sealed"

    import shutil

    template = tmp_path / "experiments/15-v2-ablation/templates/train.yaml"
    template.parent.mkdir(parents=True)
    shutil.copy(
        Path(__file__).parents[3] / "experiments/15-v2-ablation/templates/train.yaml",
        template,
    )
    report = prepare_jobs(release, tmp_path / "jobs", repo_root=tmp_path)
    assert len(report["jobs"]) == 6 and not report["submitted"]
    for job in report["jobs"]:
        cfg = yaml.safe_load((tmp_path / job["config"]).read_text())
        assert cfg["seed"] in (42, 43)
        assert cfg["revision_of_model"] == "fixed"
        assert not cfg["load_best_model_at_end"]
        assert cfg["use_wandb"] is True
        assert cfg["wandb_project"] == "spatialentail-sft"
        assert cfg["wandb_mode"] == "online"
        assert cfg["wandb_name"].endswith(job["name"])
        assert cfg["wandb_run_id"].endswith(job["name"])
        assert all(
            "bench" not in x["path"] for x in cfg["datasets"] + cfg["test_datasets"]
        )
        script = (tmp_path / job["script"]).read_text()
        assert str(tmp_path) not in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --time=03:00:00" in script
        assert "#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90" in script
        assert "#SBATCH --mail-user=e0958116@u.nus.edu" in script
        assert "/logs/%x-%j.out" in script
        assert "/logs/%x-%j.err" in script
        assert "WANDB_RESUME=allow" in script
        import subprocess

        subprocess.run(["bash", "-n", str(tmp_path / job["script"])], check=True)

    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copytree(release, moved / "release")
    moved_template = moved / "experiments/15-v2-ablation/templates/train.yaml"
    moved_template.parent.mkdir(parents=True)
    shutil.copy(template, moved_template)
    copied = prepare_jobs(moved / "release", moved / "jobs", repo_root=moved)
    assert len(copied["jobs"]) == 6
    assert (moved / "jobs/logs/.gitkeep").is_file()


def test_nested_partial_group_cannot_pass_with_rehashed_artifact(tmp_path):
    release = production_fixture(tmp_path)
    path = release / "train-0.5.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    import hashlib

    manifest["artifacts"]["train-0.5"].update(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=1, bases=1
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="paired groups"):
        audit_release(release)


def test_rehashed_corrupted_symbolic_trace_cannot_seal(tmp_path):
    release = production_fixture(tmp_path)
    path = release / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    symbolic = next(r for r in rows if r["metadata"]["trace_format"] == "symbolic")
    content = symbolic["messages"][-1]["content"]
    records = content.split("<think>\n", 1)[1].split("\n</think>\n", 1)[0].splitlines()
    records[0] = "{}"
    symbolic["messages"][-1]["content"] = (
        "<think>\n" + "\n".join(records) + "\n</think>\n" + content.rsplit("\n", 1)[-1]
    )
    # Re-admit and rehash to test semantic replay, not merely integrity hashes.
    from spatial.v2.context_budget import ContextBudget, admit_row_group
    from spatial.v2.tests.test_context_budget import FixtureTokenizer

    same = [
        r for r in rows if r["metadata"]["base_id"] == symbolic["metadata"]["base_id"]
    ]
    admitted = admit_row_group(
        same,
        FixtureTokenizer(),
        ContextBudget(**symbolic["metadata"]["context_budget"]),
    )
    replacements = {r["id"]: r for r in admitted.rows}
    rows = [replacements.get(r["id"], r) for r in rows]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    import hashlib

    manifest["artifacts"]["train"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="replay"):
        audit_release(release)
