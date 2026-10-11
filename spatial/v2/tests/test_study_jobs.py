import json
import subprocess
from pathlib import Path

import pytest

from spatial.v2.study.build import build_release
from spatial.v2.study.jobs import prepare_jobs
from spatial.v2.tests.test_context_budget import FixtureTokenizer
from spatial.v2.tests.test_study_release import config


def test_shared_launcher_loads_private_wandb_environment(tmp_path):
    credentials = tmp_path / ".config/spatialentail/wandb.env"
    credentials.parent.mkdir(parents=True)
    credentials.write_text("WANDB_API_KEY=test-only-key\n")
    common = (
        Path(__file__).parents[3]
        / "experiments/15-v2-ablation/jobs/common.bash"
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; load_wandb_credentials; printf %s "$WANDB_API_KEY"',
            "bash",
            str(common),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert result.stdout == "test-only-key"


def test_jobs_refuse_engineering_release(tmp_path):
    release = tmp_path / "release"
    build_release(config(tmp_path), release, tokenizer=FixtureTokenizer())
    with pytest.raises(ValueError, match="sealed"):
        prepare_jobs(release, tmp_path / "jobs", repo_root=tmp_path)


def test_jobs_refuse_status_only_seal(tmp_path):
    release = tmp_path / "release"
    build_release(config(tmp_path), release, tokenizer=FixtureTokenizer())
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "sealed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="audit"):
        prepare_jobs(release, tmp_path / "jobs", repo_root=tmp_path)
