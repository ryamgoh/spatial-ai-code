#!/bin/bash
# Exp 8 — 4B data-scaling arm on a full H100 96GB.
# Cells: 4b-1.5k, 4b-5k, 4b-20k (2.1, and 2.2's 4B point).
#
# Submit with the 47 job:
#   sbatch run_batch_08_scaling_h100_96.sh
#   sbatch run_batch_08_scaling_h100_47.sh
#SBATCH --job-name=spatial8-96
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ONLY="${ONLY:-4b-1.5k,4b-5k,4b-20k}"
export ONLY
# Body + cell table live in the sequential launcher; #SBATCH there is ignored.
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/run_batch_08_scaling_h100.sh"
