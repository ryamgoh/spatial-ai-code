#!/bin/bash
# Exp 8 — eval-only for the finished 0.8B-20k adapter.
# Train already saved at models/qwen3.5-0.8b-sft-20000/. Does not start 2B.
#
# One MIG is enough (0.8B QLoRA + two-pass eval). Requesting h100-47:4
# would hand vLLM four devices; this job uses :1.
#
#   sbatch run_batch_08_eval_0.8b_h100_47.sh
#SBATCH --job-name=spatial8-eval-08b
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

SKIP_TRAIN=1
ONLY=0.8b-20k
export SKIP_TRAIN ONLY
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/run_batch_08_scaling_h100.sh"
