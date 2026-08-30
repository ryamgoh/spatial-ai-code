#!/bin/bash
#SBATCH --job-name=spatialeval_long
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100-80:1 
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
 
# FINETUNING
# export OMP_NUM_THREADS=16
# export MKL_NUM_THREADS=16
# export PYTORCH_ALLOC_CONF=expandable_segments:True
#
# export AXOLOTL_NO_TELEMETRY=1
# export AXOLOTL_DO_NOT_TRACK=1
# cd finetune
# srun uv sync
# srun --cpu-bind=cores uv run python finetune.py ../experiments/03-sft-vs-baseline/train-sft-8b.yaml
# cd ../eval

# EVAL
cd eval
srun uv sync
# configs now live in ../experiments/<experiment>/ (paths below are from this dir)
srun --cpu-bind=cores uv run python eval_new.py --config ../experiments/01-few-shot-prompting/fewshot.yaml
# srun --cpu-bind=cores uv run python eval_new.py --config ../experiments/00-baseline-model/eval-baseline.yaml
# srun --cpu-bind=cores uv run python eval_new.py --config ../experiments/03-sft-vs-baseline/eval-sft-finetuned.yaml
# (removed from repo) Qwen_7B_*/Gemma_12B two-pass configs -> ../experiments/archive/legacy-two-pass/
# (removed from repo) Deepseek nonshot/oneshot/threeshot variants -> ../experiments/01-few-shot-prompting/
# (removed from repo) Qwen_3.5_27B_thinking_two_pass, ../configs/evals/* — never present in this repo
