"""Exercise launcher arguments without allocating CPUs or running admissions."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "experiments/spatialentail-composition-v1/run-cpu.slurm.sh"


@pytest.mark.parametrize("mode", ["generate", "benchmark"])
def test_launcher_supports_older_uv(mode, tmp_path):
    srun = tmp_path / "srun"
    srun.write_text(
        """#!/usr/bin/env bash
for arg in "$@"; do
    if [[ "$arg" == "--system-certs" ]]; then
        echo "error: unexpected argument '--system-certs' found" >&2
        exit 2
    fi
done
printf '%s\\n' "$@"
"""
    )
    srun.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SLURM_SUBMIT_DIR": str(ROOT),
        "SLURM_JOB_ID": "launcher-test",
        "SLURM_SRUN_CPUS_PINNED": "",
        "SLURM_TRES_PER_TASK": "cpu=128",
        "TOKENIZER": "/unused/tokenizer",
        "CONFIG": "/unused/config",
        "OUTPUT": "/unused/output",
        "MODE": mode,
        "WORKERS": "128",
        "RESUME": "0",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
            lscpu() { :; }
            export -f lscpu
            bash "$1"
            """,
            "launcher-test",
            str(LAUNCHER),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "\nuv\n--native-tls\nrun\n" in result.stdout
    assert "--cpus-per-task=128" in result.stdout
    module = "spatial.v2.composition_execution"
    if mode == "benchmark":
        module += "_benchmark"
    assert f"\n{module}\n" in result.stdout
