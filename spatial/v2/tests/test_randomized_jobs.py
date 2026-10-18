import hashlib
import json

import yaml

from spatial.v2.study.randomized_jobs import prepare_jobs


def test_prepare_jobs_exports_lean_data_and_six_4096_configs(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    files = {}
    for contract, package in (("single", "natural"), ("single", "answer-only"), ("multi", "natural")):
        for split in ("train", "development"):
            relative = f"{contract}/{split}-{package}.jsonl"
            row = {
                "id": f"{contract}-{split}-{package}",
                "messages": [{"role": "assistant", "content": "Answer: A"}],
                "metadata": {"ignored": True},
                "audit": {"ignored": True},
            }
            content = json.dumps(row) + "\n"
            path = candidate / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            files[relative] = {
                "bytes": len(content.encode()),
                "rows": 1,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
    manifest = {"files": files}
    manifest_text = json.dumps(manifest)
    (candidate / "manifest.json").write_text(manifest_text)
    token_audit = tmp_path / "token-audit.json"
    token_audit.write_text(
        json.dumps(
            {
                "status": "passed",
                "recommended_train_max_tokens": 4096,
                "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
            }
        )
    )

    receipt = prepare_jobs(candidate, token_audit, tmp_path / "jobs", tmp_path)

    assert len(receipt["jobs"]) == 6
    assert {job["package"] for job in receipt["jobs"]} == {
        "single-natural",
        "single-answer-only",
        "multi-natural",
    }
    config = yaml.safe_load(
        (tmp_path / "jobs/configs/multi-natural-seed-42.yaml").read_text()
    )
    assert config["sequence_len"] == 4096
    assert config["datasets"][0]["type"] == "chat_template"
    lean = json.loads(
        (tmp_path / "jobs/data/multi-natural-train.jsonl").read_text()
    )
    assert set(lean) == {"id", "messages"}
