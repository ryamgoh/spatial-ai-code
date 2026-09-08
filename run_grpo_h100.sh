#!/bin/bash
#SBATCH --job-name=spatialgrpo
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# #SBATCH --cpus-per-task and SLURM_CPUS_PER_TASK must never differ (Slurm 23+).
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm_pin_srun_cpus.sh"

cd finetune
srun uv sync
if [[ -s ../spatial_grpo_data.jsonl ]]; then
  srun uv run python generate_grpo.py --annotate --out ../spatial_grpo_data.jsonl
else
  srun uv run python generate_grpo.py --n 4000 --out ../spatial_grpo_data.jsonl
fi
srun uv run python finetune.py ../experiments/05-grpo/train-grpo-8b.yaml
