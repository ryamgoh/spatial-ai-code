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

cd finetune
srun uv sync
if [[ -s ../spatial_grpo_data.jsonl ]]; then
  srun uv run python generate_grpo.py --annotate --out ../spatial_grpo_data.jsonl
else
  srun uv run python generate_grpo.py --n 4000 --out ../spatial_grpo_data.jsonl
fi
srun uv run python finetune.py ../experiments/05-grpo/train-grpo-8b.yaml
