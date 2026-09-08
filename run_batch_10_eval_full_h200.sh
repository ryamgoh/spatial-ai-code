#!/bin/bash
# Exp 10 — eval-only on TQA-Corr-Full (adapters already trained).
#
# Partitions (wall-clock max):
#   gpu       — 3 hours  (H200 lives here; this script's default)
#   gpu-long  — 3 days   (H100-47 / anything that needs more than 3h)
# Eight cells will likely not finish in 3h. Resubmit the same job: it skips
# any tag that already has results.json. For one shot, use gpu-long below.
#
# Default skips 4b-20k (still training). Same script evals it later.
#
#   sbatch run_batch_10_eval_full_h200.sh
#   ONLY=4b-20k sbatch run_batch_10_eval_full_h200.sh          # after 4B-20k finishes
#   SKIP_TAGS= sbatch run_batch_10_eval_full_h200.sh           # all 9 cells
#   ONLY=4b-1.5k,4b-5k sbatch run_batch_10_eval_full_h200.sh
#   FORCE=1 sbatch run_batch_10_eval_full_h200.sh              # redo even if results.json exists
#
# H100-47 MIG on gpu-long (max 3 days) instead of H200 / gpu (max 3h):
#   sbatch --partition=gpu-long --time=3-00:00:00 --gres=gpu:h100-47:1 \
#     run_batch_10_eval_full_h200.sh
#SBATCH --job-name=spatial10-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# #SBATCH --cpus-per-task and SLURM_CPUS_PER_TASK must never differ (Slurm 23+).
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm_pin_srun_cpus.sh"

EXP=experiments/10-option-e-full
FULL=$SLURM_SUBMIT_DIR/data/spatialeval_corr_full.jsonl
EVAL_OUT=$SLURM_SUBMIT_DIR/$EXP/results/full

# Default: skip the cell that is still training. ONLY= overrides this.
# Unset SKIP_TAGS (SKIP_TAGS=) to include 4b-20k in a full sweep.
SKIP_TAGS="${SKIP_TAGS-4b-20k}"

CELLS=(
  "0.8b-1.5k  models/qwen3.5-0.8b-sft-full1500    eval-sft-0.8b-1500.yaml"
  "0.8b-5k    models/qwen3.5-0.8b-sft-full5000    eval-sft-0.8b-5000.yaml"
  "0.8b-20k   models/qwen3.5-0.8b-sft-full20000   eval-sft-0.8b-20000.yaml"
  "2b-1.5k    models/qwen3.5-2b-sft-full1500      eval-sft-2b-1500.yaml"
  "2b-5k      models/qwen3.5-2b-sft-full5000      eval-sft-2b-5000.yaml"
  "2b-20k     models/qwen3.5-2b-sft-full20000     eval-sft-2b-20000.yaml"
  "4b-1.5k    models/qwen3.5-4b-sft-full1500      eval-sft-4b-1500.yaml"
  "4b-5k      models/qwen3.5-4b-sft-full5000      eval-sft-4b-5000.yaml"
  "4b-20k     models/qwen3.5-4b-sft-full20000     eval-sft-4b-20000.yaml"
)

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

in_csv() {
  local needle=$1
  local csv=$2
  local IFS=,
  local t
  for t in $csv; do
    [[ "$t" == "$needle" ]] && return 0
  done
  return 1
}

want_tag() {
  local tag=$1
  if [[ -n "${ONLY:-}" ]]; then
    in_csv "$tag" "$ONLY"
    return $?
  fi
  if [[ -n "${SKIP_TAGS:-}" ]] && in_csv "$tag" "$SKIP_TAGS"; then
    return 1
  fi
  return 0
}

if [[ ! -s "$FULL" ]]; then
  echo "Missing $FULL — run make_corr_full.py (or an SFT launcher) first."
  exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
# vLLM/axolotl do int(CUDA_VISIBLE_DEVICES); SLURM may pass a MIG UUID.
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0"
  export CUDA_VISIBLE_DEVICES=0
fi
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
echo "ONLY=${ONLY-} SKIP_TAGS=${SKIP_TAGS-} FORCE=${FORCE-0}"
if [[ -z "${ONLY:-}" && -n "${SKIP_TAGS:-}" ]]; then
  echo "Default skip: $SKIP_TAGS  (ONLY=4b-20k later, or SKIP_TAGS= for all 9)"
fi
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
  if [[ "${FORCE:-0}" != "1" && -f "$out/results.json" ]]; then
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
  echo "Nothing to eval (filtered out, missing adapters, or results already present)."
fi
if [[ -z "${ONLY:-}" ]] && in_csv "4b-20k" "${SKIP_TAGS:-}"; then
  echo "4b-20k skipped. After that adapter is written:"
  echo "    ONLY=4b-20k sbatch run_batch_10_eval_full_h200.sh"
fi
echo "    cd eval && uv run --no-project python ../experiments/10-option-e-full/scripts/summarize.py"
