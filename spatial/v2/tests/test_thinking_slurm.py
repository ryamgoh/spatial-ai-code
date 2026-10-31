"""Test job submission without Slurm or a GPU."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXP = Path("experiments/spatialentail-composition-v1")
SCRIPTS = ROOT / EXP / "slurm"


@pytest.fixture
def workspace(tmp_path):
    experiment = tmp_path / EXP
    experiment.mkdir(parents=True)
    shutil.copytree(SCRIPTS, experiment / "slurm")
    (experiment / "active-model.json").write_text("{}")
    (tmp_path / "spatial/v2").mkdir(parents=True)
    (tmp_path / "spatial/v2/entailment_thinking_training.py").touch()
    (experiment / "unique-only-selection-v1").mkdir()
    for name in (
        "generation-exclusions-v1.json",
        "entailment-config-frozen-v1.json",
        "CONTROLLED-TRAINING-BUILD-v2.manifest.json",
        "unique-only-selection-v1/selection.json",
    ):
        (experiment / name).write_text("{}\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    sbatch = binary / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        'count=$(wc -l < "$SBATCH_LOG"); count=$((count))\n'
        'printf "%s\\n" "$*" >> "$SBATCH_LOG"\n'
        'if [[ "${FAIL_AT:-}" == "$count" ]]; then exit 1; fi\n'
        'printf "%s;cluster\\n" "$((101 + count))"\n'
    )
    sbatch.chmod(0o755)
    log = tmp_path / "sbatch.log"
    log.touch()
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "SBATCH_LOG": str(log),
    }
    return tmp_path, env, log


def run_submit(workspace):
    root, env, _ = workspace
    return subprocess.run(
        ["bash", str(EXP / "slurm/submit.sh")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_three_jobs_share_success_dependency_and_fresh_output(workspace):
    result = run_submit(workspace)
    assert result.returncode == 0, result.stderr
    root, _, log = workspace
    calls = log.read_text().splitlines()
    assert len(calls) == 3
    assert "--dependency" not in calls[0]
    assert calls[0].endswith("/prepare.sh")
    assert calls[1].endswith("/train-natural.sh")
    assert calls[2].endswith("/train-symbolic.sh")
    assert all("--dependency=afterok:101" in c for c in calls[1:])
    assert all("--kill-on-invalid-dep=yes" in c for c in calls[1:])
    records = list((root / EXP / "runs").glob("*/jobs.txt"))
    assert len(records) == 1
    assert "natural=102" in records[0].read_text()
    assert "symbolic=103" in records[0].read_text()
    assert not (records[0].parent / "prepared").exists()
    assert not (root / EXP / "unique-training-candidate-v3").exists()


def test_missing_tracked_input_submits_nothing(workspace):
    root, _, log = workspace
    (root / EXP / "generation-exclusions-v1.json").unlink()
    result = run_submit(workspace)
    assert result.returncode != 0
    assert "Missing tracked generation input" in result.stderr
    assert log.read_text() == ""


@pytest.mark.parametrize("fail_at,expected_calls", [("0", 1), ("1", 2), ("2", 3)])
def test_submission_failure_stops_and_reports_record(
    workspace, fail_at, expected_calls
):
    _, env, log = workspace
    env["FAIL_AT"] = fail_at
    result = run_submit(workspace)
    assert result.returncode != 0
    assert len(log.read_text().splitlines()) == expected_calls
    assert "Submission record:" in result.stderr


def test_partitions_gpu_types_and_shell_syntax():
    for path in SCRIPTS.iterdir():
        if path.suffix in {".sh", ".bash"}:
            subprocess.run(["bash", "-n", str(path)], check=True)
    cpu = (SCRIPTS / "prepare.sh").read_text()
    assert "#SBATCH --partition=normal" in cpu
    assert "--gres" not in cpu
    assert "sync --frozen" in cpu
    assert "generate_and_prepare.py" in cpu
    assert "#SBATCH --mail-type=END,FAIL,TIME_LIMIT\n" in cpu
    assert "#SBATCH --mail-user=e0958116@u.nus.edu" in cpu
    for style in ("natural", "symbolic"):
        script = (SCRIPTS / f"train-{style}.sh").read_text()
        assert "#SBATCH --partition=gpu-long" in script
        assert "#SBATCH --gres=gpu:h100-47:1" in script
        assert f"train-common.bash {style}" in script
        assert "#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90" in script
        assert "#SBATCH --mail-user=e0958116@u.nus.edu" in script
    common = (SCRIPTS / "train-common.bash").read_text()
    assert common.index("verify --output") < common.index("finetune/finetune.py")
    assert "--no-sync" in common
    assert common.index("load_wandb_credentials") < common.index("srun uv")
    assert "export WANDB_REQUIRED=1" in common
    assert "export WANDB_MODE=online" in common
    assert "export WANDB_RESUME=allow" in common


@pytest.mark.parametrize("credentials", ["missing", "environment", "file"])
def test_training_requires_randomized_v2_credentials(tmp_path, credentials):
    helper = Path("experiments/15-v2-ablation/jobs/common.bash")
    (tmp_path / helper).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / helper, tmp_path / helper)
    (tmp_path / "slurm/lib").mkdir(parents=True)
    (tmp_path / "slurm/lib/pin-srun-cpus.sh").write_text("# no allocation in test\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    mock = binary / "srun"
    mock.write_text(
        "#!/bin/bash\n"
        '[[ "$WANDB_MODE" == online && "$WANDB_REQUIRED" == 1 ]] || exit 9\n'
        '[[ "$WANDB_JOB_TYPE" == natural && -n "$WANDB_API_KEY" ]] || exit 9\n'
        'printf "%s\\n" "$*" >> "$SRUN_LOG"\n'
    )
    mock.chmod(0o755)
    log = tmp_path / "srun.log"
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("WANDB_")
    }
    env.update(
        SLURM_SUBMIT_DIR=str(tmp_path),
        PREPARED="run/prepared",
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        PATH=f"{binary}:{env['PATH']}",
        SRUN_LOG=str(log),
        WANDB_DISABLED="true",
        WANDB_MODE="offline",
    )
    if credentials == "environment":
        env["WANDB_API_KEY"] = "test-only-not-a-real-key"
    elif credentials == "file":
        config = tmp_path / "config/spatialentail"
        config.mkdir(parents=True)
        (config / "wandb.env").write_text("WANDB_API_KEY=test-only-not-a-real-key\n")
    result = subprocess.run(
        ["bash", str(SCRIPTS / "train-common.bash"), "natural"],
        env=env, capture_output=True, text=True, check=False,
    )
    if credentials == "missing":
        assert result.returncode != 0
        assert "Missing WANDB_API_KEY" in result.stderr
        assert not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert len(log.read_text().splitlines()) == 3
