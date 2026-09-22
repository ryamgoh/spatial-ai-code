from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "experiments/14-v13-grpo-feasibility/slurm/eval-retention-h200.sh"


def test_retention_launcher_is_one_gpu_and_evaluation_only() -> None:
    text = LAUNCHER.read_text()

    assert "#SBATCH --gres=gpu:h200-141:1" in text
    assert "eval-grpo-v13.yaml" in text
    assert "eval-grpo-breakpoint.yaml" in text
    assert "finetune.py" not in text
    assert "vllm-serve" not in text
    assert "make_rl_pool.py" not in text
