#!/bin/bash
# Exp 8 — 0.8B / 2B parameter-scaling arm on an H100 47GB MIG slice.
# Cells: 0.8b-20k, 2b-20k (2.2). 4B-20k is trained on the 96GB job.
#
# Submit with the 96 job:
#   sbatch experiments/08-dual-scaling/slurm/scaling-h100-96.sh
#   sbatch experiments/08-dual-scaling/slurm/scaling-h100-47.sh
#SBATCH --job-name=spatial8-47
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=experiments/08-dual-scaling/logs/%x-%j.out
#SBATCH --error=experiments/08-dual-scaling/logs/%x-%j.err

ONLY="${ONLY:-0.8b-20k,2b-20k}"
export ONLY
# Body + cell table live in the sequential launcher; #SBATCH there is ignored.
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/experiments/08-dual-scaling/slurm/scaling-all-h100-47.sh"
