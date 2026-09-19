#!/bin/bash
# Generate the frozen V13 diagnostic suite and evaluate the untuned 4B model
# plus the frozen Exp 12 1.5k and 6k SFT adapters. No training is performed.
#
#   sbatch experiments/13-iterative-hardening/slurm/eval-v12-on-v13-h200.sh
#   ONLY=v12-4b-1.5k sbatch experiments/13-iterative-hardening/slurm/eval-v12-on-v13-h200.sh
#   FORCE=1 sbatch experiments/13-iterative-hardening/slurm/eval-v12-on-v13-h200.sh
#   FORCE_DATA=1 sbatch experiments/13-iterative-hardening/slurm/eval-v12-on-v13-h200.sh
#   STAGES=1 ONLY=v12-4b-1.5k sbatch experiments/13-iterative-hardening/slurm/eval-v12-on-v13-h200.sh
#SBATCH --job-name=spatial13-diagnostic
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# The gpu partition has a three-hour wall-time limit.
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=experiments/13-iterative-hardening/logs/%x-%j.out
#SBATCH --error=experiments/13-iterative-hardening/logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

EXP=experiments/13-iterative-hardening
DATA=data/spatial_v13_diagnostic_test.jsonl
RESULTS=$EXP/results

# tag  eval config  optional adapter
CELLS=(
  "baseline      eval-baseline-4b-v13-diagnostic.yaml -"
  "v12-4b-1.5k  eval-v12-4b-1500-v13-diagnostic.yaml experiments/12-v6-mix-reweight/models/qwen3.5-4b-sft-v12-1500"
  "v12-4b-6k    eval-v12-4b-6000-v13-diagnostic.yaml experiments/12-v6-mix-reweight/models/qwen3.5-4b-sft-v12-6000"
)

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

want_tag() {
  local tag=$1
  [[ -z "${ONLY:-}" ]] && return 0
  local IFS=, candidate
  for candidate in $ONLY; do
    [[ "$candidate" == "$tag" ]] && return 0
  done
  return 1
}

mkdir -p "$EXP/logs" "$RESULTS" data
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0"
  export CUDA_VISIBLE_DEVICES=0
fi
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
echo "ONLY=${ONLY-} FORCE=${FORCE-0} FORCE_DATA=${FORCE_DATA-0} STAGES=${STAGES-2}"
nvidia-smi -L || true

cd eval
srun uv sync || exit 1
cd "$SLURM_SUBMIT_DIR"

if [[ "${FORCE_DATA:-0}" != "1" && -s "$DATA" ]]; then
  uv run --no-project python "$EXP/scripts/validate_diagnostic_data.py" "$DATA" || FORCE_DATA=1
fi
if [[ "${FORCE_DATA:-0}" == "1" || ! -s "$DATA" ]]; then
  echo "=== Generate frozen V13 diagnostic suite (2,256 rows, seed 1313) ==="
  srun --cpu-bind=cores uv run python spatial/generate_diagnostic_v13.py \
    --out data/spatial_v13_diagnostic.jsonl \
    --seed 1313 || exit 1
fi
uv run --no-project python "$EXP/scripts/validate_diagnostic_data.py" "$DATA" || exit 1

selected=0
for row in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  tag=$1 config=$2 adapter=$3
  want_tag "$tag" || continue
  selected=1
  if [[ "$adapter" != "-" ]] && ! has_adapter "$adapter"; then
    echo "Missing frozen adapter for $tag at $adapter; aborting"
    exit 1
  fi
  out=$RESULTS/$tag
  if [[ "${STAGES:-2}" != "2" ]]; then
    out=$RESULTS/$tag-stage${STAGES}
  fi
  if [[ "${FORCE:-0}" != "1" && -f "$out/results.json" ]]; then
    echo "=== skip $tag (already have results.json; set FORCE=1 to rerun) ==="
    continue
  fi
  echo "=== evaluate $tag on V13 diagnostic ==="
  mkdir -p "$out"
  cd eval
  srun --cpu-bind=cores uv run python eval_new.py \
    --config "../$EXP/$config" \
    --stages "${STAGES:-2}" \
    --output-dir "../$out" || exit 1
  cd "$SLURM_SUBMIT_DIR"
done

if [[ "$selected" -eq 0 ]]; then
  echo "No evaluation cells selected (ONLY=${ONLY-})"
  exit 1
fi

if [[ "${STAGES:-2}" == "2" ]]; then
  uv run --no-project python "$EXP/scripts/summarize.py" || exit 1
  echo "Summary: $RESULTS/SUMMARY.md"
  echo "All cells: $RESULTS/BUCKETS.csv"
else
  echo "Stage-${STAGES} samples are under $RESULTS/*-stage${STAGES}/"
fi
