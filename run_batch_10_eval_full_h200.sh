#!/bin/bash
# Exp 10 — optional H200 eval if the 96 jobs skipped eval (SKIP_EVAL=1).
# Default path: eval runs on the train GPU after SFT.
#
#   sbatch run_batch_10_eval_full_h200.sh
#   ONLY=4b-5k sbatch run_batch_10_eval_full_h200.sh
#SBATCH --job-name=spatial10-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/10-option-e-full
FULL=$SLURM_SUBMIT_DIR/data/spatialeval_corr_full.jsonl
EVAL_OUT=$SLURM_SUBMIT_DIR/$EXP/results/full

CELLS=(
  "4b-1.5k  models/qwen3.5-4b-sft-full1500   eval-sft-4b-1500.yaml"
  "4b-5k    models/qwen3.5-4b-sft-full5000   eval-sft-4b-5000.yaml"
  "4b-20k   models/qwen3.5-4b-sft-full20000  eval-sft-4b-20000.yaml"
)

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

want_tag() {
  local tag=$1
  if [[ -z "${ONLY:-}" ]]; then
    return 0
  fi
  local IFS=,
  local t
  for t in $ONLY; do
    [[ "$t" == "$tag" ]] && return 0
  done
  return 1
}

if [[ ! -s "$FULL" ]]; then
  echo "Missing $FULL — run make_corr_full.py (or an SFT launcher) first."
  exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true

cd eval
srun uv sync

ran=0
for row in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  tag=$1 adapter=$2 ev=$3
  want_tag "$tag" || continue
  dest=$SLURM_SUBMIT_DIR/$EXP/$adapter
  out=$EVAL_OUT/$tag
  if ! has_adapter "$dest"; then
    echo "WARN: no adapter at $dest — skip eval $tag"
    continue
  fi
  if [[ -f "$out/results.json" ]]; then
    echo "=== skip eval $tag (already have results.json) ==="
    continue
  fi
  echo "=== eval $tag ==="
  mkdir -p "$out"
  srun --cpu-bind=cores uv run python eval_new.py \
    --config "../$EXP/$ev" \
    --output-dir "$out" || {
    echo "Exp 10 eval $tag FAILED"
    exit 1
  }
  echo "  eval $tag OK -> $out/"
  ran=1
done

if [[ "$ran" -eq 0 ]]; then
  echo "Nothing to eval (missing adapters or results already present)."
fi
echo "    cd eval && uv run --no-project python ../experiments/10-option-e-full/scripts/summarize.py"
