import hashlib
import json
import subprocess
from pathlib import Path

from spatial.v2.study import role_path
from spatial.v2.study.evaluation import BENCH_MODULES, prepare_evaluation


def _row(base_id: str, role: str, trace: str, arm: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Solve."},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer: A"},
        ],
        "metadata": {
            "base_id": base_id,
            "study_role": role,
            "benchmark_cell": "DIR-ARC-UNI",
            "query_kind": "direction",
            "trace_format": trace,
            "supervision_arm": arm,
        },
    }


def _release(root: Path) -> Path:
    release = root / "release"
    release.mkdir()
    artifacts = {}
    for role in BENCH_MODULES.values():
        rows = [_row(role, role, "natural", "answer-only")]
        if role != "bench-depth":
            rows.extend(
                [
                    _row(role, role, "natural", "checked-trace"),
                    _row(role, role, "symbolic", "checked-trace"),
                ]
            )
        path = role_path(release, role)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        artifacts[role] = {
            "bases": 1,
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": "spatialentail-study-release-v1",
        "status": "sealed",
        "artifacts": artifacts,
        "context_budget": {
            "tokenizer_name": "Qwen/Qwen3.5-4B",
            "tokenizer_revision": "fixed",
            "train_max_tokens": 100,
            "eval_max_tokens": 200,
            "max_new_tokens": 100,
        },
    }
    (release / "manifest.json").write_text(json.dumps(manifest))
    return release


def test_frozen_evaluation_selects_one_answer_row_per_base(tmp_path: Path) -> None:
    release = _release(tmp_path)
    output = tmp_path / "prepared"

    receipt = prepare_evaluation(release, output, repo_root=tmp_path)

    assert receipt["total_bases"] == 4
    assert len(receipt["tasks"]) == 4
    assert {item["role"] for item in receipt["tasks"]} == set(BENCH_MODULES)
    assert [job["name"] for job in receipt["jobs"]] == [
        "untuned",
        "answer-only-seed-42",
        "natural-seed-42",
    ]
    for data_path in sorted((output / "data").glob("*.jsonl")):
        rows = [json.loads(line) for line in data_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["metadata"]["supervision_arm"] == "answer-only"
        assert rows[0]["metadata"]["trace_format"] == "natural"
    assert not any("symbolic" in name for name in receipt["file_sha256"])
    for name, expected in receipt["file_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    for script in (output / "slurm").glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_frozen_evaluation_rejects_duplicate_answer_rows(tmp_path: Path) -> None:
    release = _release(tmp_path)
    path = release / "bench-core.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.append(rows[0])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["bench-core"].update(
        rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest))

    try:
        prepare_evaluation(release, tmp_path / "prepared", repo_root=tmp_path)
    except ValueError as exc:
        assert "exactly one answer-only row" in str(exc)
    else:
        raise AssertionError("duplicate answer rows were accepted")
