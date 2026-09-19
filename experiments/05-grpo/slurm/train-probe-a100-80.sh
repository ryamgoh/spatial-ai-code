#!/bin/bash
#SBATCH --job-name=spatialgrpo-probe
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=experiments/05-grpo/logs/%x-%j.out
#SBATCH --error=experiments/05-grpo/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# #SBATCH --cpus-per-task and SLURM_CPUS_PER_TASK must never differ (Slurm 23+).
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

cd finetune
srun uv sync
if [[ -s ../spatial_grpo_data.jsonl ]]; then
  srun uv run python ../spatial/generate_grpo.py --annotate --out ../spatial_grpo_data.jsonl
else
  srun uv run python ../spatial/generate_grpo.py --n 4000 --out ../spatial_grpo_data.jsonl
fi
srun uv run python finetune.py ../experiments/05-grpo/train-grpo-8b-probe.yaml
