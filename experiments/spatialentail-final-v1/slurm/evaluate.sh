#!/usr/bin/env bash
#SBATCH --job-name=spatialentail-final-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu

set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
RUN=${1:?Pass the explicit style/seed run directory}
MODE=${2:-selected}
SHOTS=${3:-0}
TRAINING_REVISION=${4:-}
args=(--run "$RUN" --shots "$SHOTS")
case "$MODE" in
    selected) ;;
    untuned) args+=(--untuned) ;;
    *) echo "Unknown evaluation mode: $MODE" >&2; exit 2 ;;
esac
[[ -z "$TRAINING_REVISION" ]] || args+=(--training-revision "$TRAINING_REVISION")
# The submitter sets --chdir; Slurm relocates this script into its spool directory.
ROOT=$(pwd -P)
source "$ROOT/slurm/lib/pin-srun-cpus.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
srun nvidia-smi
exec srun uv run --project "$ROOT" --frozen --no-sync python \
    "$ROOT/experiments/spatialentail-final-v1/sft.py" evaluate "${args[@]}"
