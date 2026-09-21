#!/bin/bash
# Matched trace ablation: train only the delta-state arm; frozen 8K is full-state.
#SBATCH --job-name=spatial13-delta
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=experiments/13-iterative-hardening/logs/%x-%j.out
#SBATCH --error=experiments/13-iterative-hardening/logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

EXP=experiments/13-iterative-hardening
ADAPTER=$EXP/models/qwen3.5-4b-v13-sft-8000-delta
SOURCE_TRAIN=data/spatial_v13_sft_8000_train.jsonl
SOURCE_VAL=data/spatial_v13_sft_8000_val.jsonl
DELTA_TRAIN=data/spatial_v13_trace_ablation_delta_train.jsonl
DELTA_VAL=data/spatial_v13_trace_ablation_delta_val.jsonl
TRACE_MANIFEST=data/spatial_v13_trace_ablation_manifest.json

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
mkdir -p "$EXP/logs" "$EXP/results" data
nvidia-smi -L || true

cd finetune
uv sync || exit 1
cd "$SLURM_SUBMIT_DIR"

if [[ ! -s "$SOURCE_TRAIN" || ! -s "$SOURCE_VAL" ]]; then
  echo "Missing frozen V13.1 8K source data; run/sync run-sft-8000-h200.sh first."
  exit 1
fi
uv run --no-project python "$EXP/scripts/make_sft_8000_data.py" --validate-only || exit 1
if [[ "${FORCE_DATA:-0}" != "1" && -s "$DELTA_TRAIN" && -s "$DELTA_VAL" && -s "$TRACE_MANIFEST" ]]; then
  uv run --no-project python "$EXP/scripts/make_trace_ablation_data.py" --validate-only || FORCE_DATA=1
fi
if [[ "${FORCE_DATA:-0}" == "1" || ! -s "$DELTA_TRAIN" || ! -s "$DELTA_VAL" || ! -s "$TRACE_MANIFEST" ]]; then
  echo "=== Derive matched full-state/delta-state data from frozen 8K worlds ==="
  uv run --no-project python "$EXP/scripts/make_trace_ablation_data.py" || exit 1
fi
uv run --no-project python "$EXP/scripts/make_trace_ablation_data.py" --validate-only || exit 1

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if has_adapter "$ADAPTER"; then
    echo "=== skip delta-state training (adapter complete) ==="
  else
    cd finetune
    train_args=(python finetune.py ../"$EXP"/train-sft-4b-8000-delta.yaml)
    if compgen -G "../$ADAPTER/checkpoint-*" >/dev/null; then
      train_args+=(--resume)
    fi
    uv run "${train_args[@]}" || exit 1
    cd "$SLURM_SUBMIT_DIR"
  fi
fi
if ! has_adapter "$ADAPTER"; then
  echo "Missing delta-state adapter: $ADAPTER"
  exit 1
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  cd eval
  uv sync || exit 1
  for row in \
    "trace-delta-v13-stage1 eval-trace-delta-v13.yaml" \
    "trace-delta-breakpoint-stage1 eval-trace-delta-breakpoint.yaml"; do
    set -- $row
    tag=$1 config=$2 out=../$EXP/results/$1
    if [[ "${FORCE_EVAL:-0}" != "1" && -f "$out/results.json" ]]; then
      continue
    fi
    uv run python eval_new.py \
      --config ../$EXP/"$config" --stages 1 --output-dir "$out" || exit 1
  done
  cd "$SLURM_SUBMIT_DIR"
fi

uv run --no-project python "$EXP/scripts/summarize_trace_ablation.py" || true
echo "Trace ablation: $EXP/results/TRACE-ABLATION-SUMMARY.md"
