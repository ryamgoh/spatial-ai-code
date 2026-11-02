#!/usr/bin/env bash
#SBATCH --job-name=spatialentail-final-sft
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu

set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
RUN=${1:?Pass the explicit style/seed run directory}
ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
source "$ROOT/slurm/lib/pin-srun-cpus.sh"
source "$ROOT/experiments/15-v2-ablation/jobs/common.bash"
load_wandb_credentials
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=online WANDB_RESUME=never
export WANDB_JOB_TYPE=$(basename "$RUN")
unset WANDB_DISABLED
export HF_DATASETS_CACHE="$RUN/cache/datasets"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export AXOLOTL_DO_NOT_TRACK=1 AXOLOTL_NO_TELEMETRY=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
srun nvidia-smi
exec srun uv run --project "$ROOT" --frozen --no-sync python \
    "$ROOT/experiments/spatialentail-final-v1/sft.py" train --run "$RUN"
