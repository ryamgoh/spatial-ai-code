#!/bin/bash
# Exp 8 — 2B-20k on a second H100 96GB, separate folders from the 47 job.
#
# 47 job writes:  models/qwen3.5-2b-sft-20000/     results/scaling/2b-20k/
# This job writes: models/qwen3.5-2b-sft-20000-96b/ results/scaling/2b-20k-96b/
#
# Submit now. scancel the 47 job when it logs "SFT QLoRA 2b-20k".
# Leftover 47 files in the original 2b-20k dir are unused; summarize
# prefers 2b-20k-96b.
#SBATCH --job-name=spatial8-96b
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ONLY="${ONLY:-2b-20k-96b}"
export ONLY
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/run_batch_08_scaling_h100.sh"
