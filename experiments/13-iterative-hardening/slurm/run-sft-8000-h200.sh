#!/bin/bash
# Native V13.1 nested 8K run, fresh Qwen3.5-4B, one full H200.
# 8,192-token traces use micro-batch 2 on the 141 GB H200; checkpoint/resubmit
# support remains available on the gpu partition's 03:00:00 limit.
#SBATCH --job-name=spatial13-sft8k
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
ADAPTER=$EXP/models/qwen3.5-4b-v13-sft-8000
DIAGNOSTIC=data/spatial_v13_diagnostic_test.jsonl
BREAKPOINT=data/spatial_v13_breakpoint_test.jsonl
BREAKPOINT_MANIFEST=data/spatial_v13_breakpoint_manifest.json
V1_TRAIN=data/spatial_v13_probe_train.jsonl
V1_VAL=data/spatial_v13_probe_val.jsonl
V2_TRAIN=data/spatial_v13_probe_v2_train.jsonl
V2_VAL=data/spatial_v13_probe_v2_val.jsonl
SFT15_TRAIN=data/spatial_v13_sft_1500_train.jsonl
SFT15_VAL=data/spatial_v13_sft_1500_val.jsonl
SFT15_MANIFEST=data/spatial_v13_sft_1500_manifest.json
SFT6_TRAIN=data/spatial_v13_sft_6000_train.jsonl
SFT6_VAL=data/spatial_v13_sft_6000_val.jsonl
SFT6_MANIFEST=data/spatial_v13_sft_6000_manifest.json
TRAIN=data/spatial_v13_sft_8000_train.jsonl
VAL=data/spatial_v13_sft_8000_val.jsonl
MANIFEST=data/spatial_v13_sft_8000_manifest.json

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

if [[ -s "$DIAGNOSTIC" ]]; then
  uv run --no-project python "$EXP/scripts/validate_diagnostic_data.py" "$DIAGNOSTIC" || FORCE_DIAGNOSTIC=1
else
  FORCE_DIAGNOSTIC=1
fi
if [[ "${FORCE_DIAGNOSTIC:-0}" == "1" ]]; then
  uv run python spatial/generate_diagnostic_v13.py \
    --out data/spatial_v13_diagnostic.jsonl --seed 1313 || exit 1
fi
uv run --no-project python "$EXP/scripts/validate_diagnostic_data.py" "$DIAGNOSTIC" || exit 1
if [[ -s "$V1_TRAIN" && -s "$V1_VAL" ]]; then
  uv run python "$EXP/scripts/make_probe_data.py" --validate-only || FORCE_V1=1
else
  FORCE_V1=1
fi
if [[ "${FORCE_V1:-0}" == "1" ]]; then
  uv run python "$EXP/scripts/make_probe_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_probe_data.py" --validate-only || exit 1
if [[ -s "$V2_TRAIN" && -s "$V2_VAL" ]]; then
  uv run python "$EXP/scripts/make_probe_v2_data.py" --validate-only || FORCE_V2=1
else
  FORCE_V2=1
fi
if [[ "${FORCE_V2:-0}" == "1" ]]; then
  uv run python "$EXP/scripts/make_probe_v2_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_probe_v2_data.py" --validate-only || exit 1
if [[ -s "$SFT15_TRAIN" && -s "$SFT15_VAL" && -s "$SFT15_MANIFEST" ]]; then
  uv run python "$EXP/scripts/make_sft_1500_data.py" --validate-only || FORCE_SFT15=1
else
  FORCE_SFT15=1
fi
if [[ "${FORCE_SFT15:-0}" == "1" ]]; then
  uv run python "$EXP/scripts/make_sft_1500_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_sft_1500_data.py" --validate-only || exit 1
if [[ -s "$SFT6_TRAIN" && -s "$SFT6_VAL" && -s "$SFT6_MANIFEST" ]]; then
  uv run python "$EXP/scripts/make_sft_6000_data.py" --validate-only || FORCE_SFT6=1
else
  FORCE_SFT6=1
fi
if [[ "${FORCE_SFT6:-0}" == "1" ]]; then
  uv run python "$EXP/scripts/make_sft_6000_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_sft_6000_data.py" --validate-only || exit 1
if [[ -s "$BREAKPOINT" && -s "$BREAKPOINT_MANIFEST" ]]; then
  uv run --no-project python "$EXP/scripts/make_breakpoint_data.py" --validate-only || FORCE_BREAKPOINT=1
else
  FORCE_BREAKPOINT=1
fi
if [[ "${FORCE_BREAKPOINT:-0}" == "1" ]]; then
  uv run --no-project python "$EXP/scripts/make_breakpoint_data.py" || exit 1
fi
uv run --no-project python "$EXP/scripts/make_breakpoint_data.py" --validate-only || exit 1

if [[ "${FORCE_DATA:-0}" != "1" && -s "$TRAIN" && -s "$VAL" && -s "$MANIFEST" ]]; then
  uv run --no-project python "$EXP/scripts/make_sft_8000_data.py" --validate-only || FORCE_DATA=1
fi
if [[ "${FORCE_DATA:-0}" == "1" || ! -s "$TRAIN" || ! -s "$VAL" || ! -s "$MANIFEST" ]]; then
  echo "=== Generate exact-nested native V13.1 8K data ==="
  uv run --no-project python "$EXP/scripts/make_sft_8000_data.py" || exit 1
fi
uv run --no-project python "$EXP/scripts/make_sft_8000_data.py" --validate-only || exit 1

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if has_adapter "$ADAPTER"; then
    echo "=== skip 8K training (adapter complete) ==="
  else
    cd finetune
    train_args=(python finetune.py ../"$EXP"/train-sft-4b-8000.yaml)
    if compgen -G "../$ADAPTER/checkpoint-*" >/dev/null; then
      echo "=== resume 8K training from latest checkpoint ==="
      train_args+=(--resume)
    else
      echo "=== train fresh native V13.1 8K adapter ==="
    fi
    uv run "${train_args[@]}" || exit 1
    cd "$SLURM_SUBMIT_DIR"
  fi
fi
if ! has_adapter "$ADAPTER"; then
  echo "Missing 8K adapter: $ADAPTER"
  exit 1
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  cd eval
  uv sync || exit 1
  out=../$EXP/results/sft-8000-v13-stage1
  if [[ "${FORCE_EVAL:-0}" == "1" || ! -f "$out/results.json" ]]; then
    uv run python eval_new.py \
      --config ../$EXP/eval-sft-4b-8000-v13.yaml \
      --stages 1 --output-dir "$out" || exit 1
  fi
  out=../$EXP/results/breakpoint-sft-8k-stage1
  if [[ "${FORCE_EVAL:-0}" == "1" || ! -f "$out/results.json" ]]; then
    uv run python eval_new.py \
      --config ../$EXP/eval-breakpoint-sft-8000.yaml \
      --stages 1 --output-dir "$out" || exit 1
  fi
  cd "$SLURM_SUBMIT_DIR"
fi

uv run --no-project python "$EXP/scripts/summarize_sft_8000.py" || true
echo "8K summary: $EXP/results/SFT-8000-SUMMARY.md"
echo "Paired changes: $EXP/results/SFT-8000-PAIRED-CHANGES.jsonl"
