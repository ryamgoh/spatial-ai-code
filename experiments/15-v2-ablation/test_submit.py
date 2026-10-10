"""Local contracts for the V2 Slurm DAG submitter."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

MODULE = Path(__file__).with_name("submit.py")
SPEC = importlib.util.spec_from_file_location("v2_ablation_submit", MODULE)
assert SPEC and SPEC.loader
SUBMIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUBMIT
SPEC.loader.exec_module(SUBMIT)

SUMMARY_MODULE = Path(__file__).with_name("scripts") / "summarize.py"
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "v2_ablation_summary", SUMMARY_MODULE
)
assert SUMMARY_SPEC and SUMMARY_SPEC.loader
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
sys.modules[SUMMARY_SPEC.name] = SUMMARY
SUMMARY_SPEC.loader.exec_module(SUMMARY)

REPO = Path(__file__).parents[2]


def options(tmp_path, run_id: str, **overrides):
    return SUBMIT.SubmissionOptions(
        run_id=run_id,
        run_root=tmp_path,
        **overrides,
    )


def test_checked_in_run_spec_renders_two_arm_configs(tmp_path) -> None:
    spec = SUBMIT.load_run_spec(Path(__file__).with_name("run.yaml"), REPO)

    prepared = SUBMIT.prepare_run(spec, REPO, "test-run", run_root=tmp_path)

    assert [arm.name for arm in prepared.arms] == [
        "single-natural",
        "single-symbolic",
        "answer-only",
        "corrupted-symbolic",
        "untuned-natural",
        "untuned-symbolic",
        "untuned-answer-only",
    ]
    natural = yaml.safe_load(prepared.arms[0].train_config.read_text())
    assert natural["datasets"][0]["path"].endswith(
        "spatial_v2_pilot_views/by_variant/single__natural__checked-trace_train.jsonl"
    )
    assert natural["test_datasets"][0]["path"].endswith(
        "spatial_v2_pilot_views/by_variant/single__natural__checked-trace_dev.jsonl"
    )
    assert natural["output_dir"].endswith("test-run/models/single-natural")
    assert natural["num_epochs"] == 2
    evaluation = yaml.safe_load(prepared.arms[0].eval_config.read_text())
    assert evaluation["model_args"]["choices"] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert evaluation["model_args"]["max_thinking_tokens"] == 4096
    assert len(evaluation["tasks"]) == 1
    assert prepared.arms[0].task_config.exists()


def test_arm_overrides_merge_without_replacing_protected_paths(tmp_path) -> None:
    spec = SUBMIT.load_run_spec(Path(__file__).with_name("run.yaml"), REPO)
    natural = replace(spec.arms[0], train_overrides={"learning_rate": 0.00003})
    spec = replace(spec, arms=(natural, *spec.arms[1:]))

    prepared = SUBMIT.prepare_run(spec, REPO, "override", run_root=tmp_path)

    config = yaml.safe_load(prepared.arms[0].train_config.read_text())
    assert config["learning_rate"] == 0.00003
    assert config["datasets"][0]["path"].endswith(
        "single__natural__checked-trace_train.jsonl"
    )


def test_run_spec_rejects_protected_config_overrides(tmp_path) -> None:
    experiment = Path(__file__).parent
    raw = yaml.safe_load((experiment / "run.yaml").read_text())
    raw["matrix"] = str(experiment / "matrix.yaml")
    raw["train"]["template"] = str(experiment / "templates/train.yaml")
    raw["eval"]["template"] = str(experiment / "templates/eval.yaml")
    raw["eval"]["task_template"] = str(experiment / "templates/task.yaml")
    raw["train"]["overrides"]["datasets"] = []
    path = tmp_path / "bad-run.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="cannot override: datasets"):
        SUBMIT.load_run_spec(path, REPO)


def test_dry_run_builds_graph_without_sbatch_or_run_directory(tmp_path) -> None:
    spec_path = Path(__file__).with_name("run.yaml")

    result = SUBMIT.submit_experiment(
        spec_path,
        REPO,
        options(tmp_path, "dry-run", dry_run=True),
        runner=lambda _command: (_ for _ in ()).throw(AssertionError("called sbatch")),
    )

    assert not (tmp_path / "dry-run").exists()
    assert [job.stage for job in result.jobs] == [
        "generate",
        "train",
        "eval",
        "train",
        "eval",
        "train",
        "eval",
        "train",
        "eval",
        "eval",
        "eval",
        "eval",
        "summarize",
    ]
    assert any("/dry-run/matrix.yaml" in part for part in result.jobs[0].command)
    assert "--dependency=afterok:<generate>" in result.jobs[1].command
    assert "--dependency=afterok:<train:single-natural>" in result.jobs[2].command


def test_skip_generation_requires_existing_selected_views(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="requires existing validated views"):
        SUBMIT.submit_experiment(
            Path(__file__).with_name("run.yaml"),
            REPO,
            options(tmp_path, "missing-data", skip_generation=True),
            runner=lambda _command: "unused",
        )
    assert not (tmp_path / "missing-data").exists()


def test_submission_wires_independent_arm_dependencies(tmp_path) -> None:
    commands: list[list[str]] = []
    ids = iter(str(value) for value in range(8100, 8120))

    def runner(command: list[str]) -> str:
        commands.append(command)
        return next(ids)

    receipt = SUBMIT.submit_experiment(
        Path(__file__).with_name("run.yaml"),
        REPO,
        options(tmp_path, "submitted"),
        runner=runner,
    )

    assert receipt.generate_job == "8100"
    assert receipt.arms["single-natural"] == {"train": "8101", "eval": "8102"}
    assert receipt.arms["single-symbolic"] == {"train": "8103", "eval": "8104"}
    assert receipt.summary_job == "8112"
    assert "--dependency=afterok:8100" in commands[1]
    assert "--dependency=afterok:8101" in commands[2]
    assert "--dependency=afterok:8100" in commands[3]
    assert "--dependency=afterok:8103" in commands[4]
    assert any(value.startswith("--dependency=afterany:") for value in commands[-1])
    stored = json.loads((tmp_path / "submitted" / "jobs.json").read_text())
    assert stored["summary"] == "8112"


def test_partial_submission_failure_preserves_job_ids(tmp_path) -> None:
    calls = 0

    def runner(_command: list[str]) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "9100"
        raise OSError("scheduler unavailable")

    with pytest.raises(RuntimeError, match="scancel 9100"):
        SUBMIT.submit_experiment(
            Path(__file__).with_name("run.yaml"),
            REPO,
            options(tmp_path, "partial"),
            runner=runner,
        )

    stored = json.loads((tmp_path / "partial" / "jobs.json").read_text())
    assert stored == {
        "schema": "spatial-v2-slurm-jobs",
        "run_id": "partial",
        "arms": {},
        "generate": "9100",
        "summary": None,
    }


def test_resume_reuses_configs_and_preserves_previous_receipt(tmp_path) -> None:
    spec = Path(__file__).with_name("run.yaml")
    first_ids = iter(str(value) for value in range(9200, 9220))
    SUBMIT.submit_experiment(
        spec,
        REPO,
        options(tmp_path, "retry"),
        runner=lambda _command: next(first_ids),
    )
    second_ids = iter(str(value) for value in range(9300, 9320))

    result = SUBMIT.submit_experiment(
        spec,
        REPO,
        options(tmp_path, "retry", resume=True),
        runner=lambda _command: next(second_ids),
    )

    assert result.generate_job == "9300"
    history = [
        json.loads(line)
        for line in (tmp_path / "retry" / "jobs-history.jsonl").read_text().splitlines()
    ]
    assert history[-1]["generate"] == "9200"


def test_real_submission_requires_slurm_login_node(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(SUBMIT.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="Slurm login server"):
        SUBMIT.submit_experiment(
            Path(__file__).with_name("run.yaml"),
            REPO,
            options(tmp_path, "no-slurm"),
        )
    assert not (tmp_path / "no-slurm").exists()


def test_generic_slurm_jobs_have_valid_bash_syntax() -> None:
    jobs = sorted(Path(__file__).with_name("jobs").glob("*.sh"))
    assert [path.name for path in jobs] == [
        "eval.sh",
        "generate.sh",
        "summarize.sh",
        "train.sh",
    ]
    common = Path(__file__).with_name("jobs") / "common.bash"
    subprocess.run(["bash", "-n", *map(str, jobs), str(common)], check=True)


def test_summary_reports_partial_arm_completion(tmp_path) -> None:
    (tmp_path / "jobs.json").write_text(
        json.dumps(
            {
                "arms": {
                    "natural": {"train": "1", "eval": "2"},
                    "symbolic": {"train": "3", "eval": "4"},
                }
            }
        )
    )
    natural_model = tmp_path / "models/natural"
    natural_result = tmp_path / "results/natural"
    natural_model.mkdir(parents=True)
    natural_result.mkdir(parents=True)
    (natural_model / "COMPLETED").touch()
    (natural_result / "COMPLETED").touch()
    (natural_result / "results.json").write_text("{}")

    summary = SUMMARY.summarize(tmp_path)

    assert summary["arms"]["natural"]["eval_complete"] is True
    assert summary["arms"]["symbolic"]["train_complete"] is False
    assert (tmp_path / "SUMMARY.md").exists()


def test_generation_views_feed_dev_selection_and_final_test(tmp_path):
    from spatial.v2.matrix import generate_matrix
    from spatial.v2.tests.test_matrix import FramingTokenizer

    spec = SUBMIT.load_run_spec(Path(__file__).with_name("run.yaml"), REPO)
    matrix = yaml.safe_load(spec.matrix.read_text())
    matrix["cells"] = matrix["cells"][:1]
    matrix["cells"][0]["count"] = 1
    matrix_path = tmp_path / "small.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix))
    spec = replace(spec, matrix=matrix_path, data_output=tmp_path / "pilot.jsonl")
    generate_matrix(spec.matrix, spec.data_output, tokenizer=FramingTokenizer())
    prepared = SUBMIT.prepare_run(spec, REPO, "smoke", run_root=tmp_path)
    SUBMIT._validate_existing_views(prepared.arms, spec.data_output, spec.matrix)
    for arm in prepared.arms:
        train = yaml.safe_load(arm.train_config.read_text())
        evaluation = yaml.safe_load(arm.eval_config.read_text())
        assert train["test_datasets"][0]["path"].endswith("_dev.jsonl")
        assert "_test.jsonl" in arm.task_config.read_text()
        assert "utils.process_results_v2" in arm.task_config.read_text()
        assert "filter_list" not in arm.task_config.read_text()
        assert "max_gen_toks: 4096" in arm.task_config.read_text()
        assert evaluation["apply_chat_template"] is False
        assert evaluation["model_args"]["add_special_tokens"] is False
        assert bool(evaluation["model_args"]["lora_path"]) == arm.train
    matrix["seed"] += 1
    matrix_path.write_text(yaml.safe_dump(matrix))
    with pytest.raises(ValueError, match="matrix configuration differs"):
        SUBMIT._validate_existing_views(prepared.arms, spec.data_output, spec.matrix)


def test_run_rejects_context_contract_drift(tmp_path):
    spec = SUBMIT.load_run_spec(Path(__file__).with_name("run.yaml"), REPO)
    spec = replace(spec, train_overrides={"sequence_len": 2048})
    with pytest.raises(ValueError, match="training sequence/template"):
        SUBMIT._validate_training_contract(spec)
