#!/usr/bin/env bash
#SBATCH --job-name=spatialentail-final-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200-141:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu

set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
RUN=${1:?Pass the explicit style/seed run directory}
BENCHMARK=${2:?Pass sft-bench or external-test}
MODE=${3:-selected}
case "$MODE" in
    selected) extra=() ;;
    untuned) extra=(--untuned) ;;
    *) echo "Unknown evaluation mode: $MODE" >&2; exit 2 ;;
esac
ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
source "$ROOT/slurm/lib/pin-srun-cpus.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
srun nvidia-smi
exec srun uv run --project "$ROOT" --frozen --no-sync python \
    "$ROOT/experiments/spatialentail-final-v1/sft.py" evaluate \
    --run "$RUN" --benchmark "$BENCHMARK" "${extra[@]}"
