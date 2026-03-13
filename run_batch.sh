#!/bin/bash
#SBATCH --job-name=spatialeval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
 
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTORCH_ALLOC_CONF=expandable_segments:True

export AXOLOTL_NO_TELEMETRY=1
export AXOLOTL_DO_NOT_TRACK=1
cd finetune
srun uv sync
srun --cpu-bind=cores uv run python finetune.py qwen3-7b-spatial-reasoning
cd ../eval
# cd eval
srun uv sync
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ../configs/evals/eval_qwen3-7b-spatial-reasoning.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/Qwen_14B_two_pass.yaml
srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/Qwen_7B_Finetuned_two_pass.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/Qwen_7B_two_pass.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/Gemma_12B_two_pass.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ../configs/evals/eval_deepseek-r1-distill-qwen-1.5b_two_pass.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/Qwen_3.5_27B_thinking_two_pass.yaml
