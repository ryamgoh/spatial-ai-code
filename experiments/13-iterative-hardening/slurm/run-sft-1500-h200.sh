#!/bin/bash
# Native V13 1.5K scale run, fresh Qwen3.5-4B, one full H200.
# If evaluation cannot finish inside 03:00:00, rerun with SKIP_TRAIN=1.
#SBATCH --job-name=spatial13-sft15
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
ADAPTER=$EXP/models/qwen3.5-4b-v13-sft-1500
DIAGNOSTIC=data/spatial_v13_diagnostic_test.jsonl
V1_TRAIN=data/spatial_v13_probe_train.jsonl
V1_VAL=data/spatial_v13_probe_val.jsonl
V2_TRAIN=data/spatial_v13_probe_v2_train.jsonl
V2_VAL=data/spatial_v13_probe_v2_val.jsonl
TRAIN=data/spatial_v13_sft_1500_train.jsonl
VAL=data/spatial_v13_sft_1500_val.jsonl
MANIFEST=data/spatial_v13_sft_1500_manifest.json
V12_TEST=data/spatial_sft_v12_2000_test.jsonl

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

if [[ ! -s "$DIAGNOSTIC" ]]; then
  uv run python spatial/generate_diagnostic_v13.py --out data/spatial_v13_diagnostic.jsonl --seed 1313 || exit 1
fi
uv run --no-project python "$EXP/scripts/validate_diagnostic_data.py" "$DIAGNOSTIC" || exit 1

# Recreate probe datasets because their worlds are held out from 1.5K.
if [[ ! -s "$V1_TRAIN" || ! -s "$V1_VAL" ]]; then
  uv run python "$EXP/scripts/make_probe_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_probe_data.py" --validate-only || exit 1
if [[ ! -s "$V2_TRAIN" || ! -s "$V2_VAL" ]]; then
  uv run python "$EXP/scripts/make_probe_v2_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_probe_v2_data.py" --validate-only || exit 1

if [[ "${FORCE_DATA:-0}" != "1" && -s "$TRAIN" && -s "$VAL" && -s "$MANIFEST" ]]; then
  uv run python "$EXP/scripts/make_sft_1500_data.py" --validate-only || FORCE_DATA=1
fi
if [[ "${FORCE_DATA:-0}" == "1" || ! -s "$TRAIN" || ! -s "$VAL" || ! -s "$MANIFEST" ]]; then
  echo "=== Generate frozen native V13 1.5K data ==="
  uv run python "$EXP/scripts/make_sft_1500_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_sft_1500_data.py" --validate-only || exit 1

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if has_adapter "$ADAPTER"; then
    echo "=== skip 1.5K training (adapter complete) ==="
  else
    cd finetune
    uv run axolotl train ../experiments/13-iterative-hardening/train-sft-4b-1500.yaml --launcher python || exit 1
    cd "$SLURM_SUBMIT_DIR"
  fi
fi
if ! has_adapter "$ADAPTER"; then
  echo "Missing 1.5K adapter: $ADAPTER"
  exit 1
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  cd eval
  uv sync || exit 1
  for stages in 1 2; do
    out=../$EXP/results/sft-1500-v13-stage$stages
    if [[ "${FORCE_EVAL:-0}" == "1" || ! -f "$out/results.json" ]]; then
      uv run python eval_new.py \
        --config ../$EXP/eval-sft-4b-1500-v13.yaml \
        --stages "$stages" \
        --output-dir "$out" || exit 1
    fi
  done
  if [[ -s "../$V12_TEST" ]]; then
    out=../$EXP/results/sft-1500-v12-stage1
    if [[ "${FORCE_EVAL:-0}" == "1" || ! -f "$out/results.json" ]]; then
      uv run python eval_new.py \
        --config ../$EXP/eval-sft-4b-1500-v12.yaml \
        --stages 1 \
        --output-dir "$out" || exit 1
    fi
  fi
  cd "$SLURM_SUBMIT_DIR"
fi

uv run --no-project python "$EXP/scripts/summarize_sft_1500.py" || true
echo "1.5K summary: $EXP/results/SFT-1500-SUMMARY.md"
