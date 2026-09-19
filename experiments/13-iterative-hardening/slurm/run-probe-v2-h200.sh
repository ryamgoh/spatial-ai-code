#!/bin/bash
# Probe v2: consistency-focused 400-row curriculum, fresh Qwen3.5-4B.
# Short run on one full H200; gpu partition max is 03:00:00.
#SBATCH --job-name=spatial13-probe-v2
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
ADAPTER=$EXP/models/qwen3.5-4b-v13-probe-v2-400
DIAGNOSTIC=data/spatial_v13_diagnostic_test.jsonl
V1_TRAIN=data/spatial_v13_probe_train.jsonl
V1_VAL=data/spatial_v13_probe_val.jsonl
V2_TRAIN=data/spatial_v13_probe_v2_train.jsonl
V2_VAL=data/spatial_v13_probe_v2_val.jsonl
V2_MANIFEST=data/spatial_v13_probe_v2_manifest.json
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

# V1 data is held out from v2. Recreate it if the server has only the code.
if [[ ! -s "$V1_TRAIN" || ! -s "$V1_VAL" ]]; then
  uv run python "$EXP/scripts/make_probe_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_probe_data.py" --validate-only || exit 1

if [[ "${FORCE_DATA:-0}" != "1" && -s "$V2_TRAIN" && -s "$V2_VAL" && -s "$V2_MANIFEST" ]]; then
  uv run python "$EXP/scripts/make_probe_v2_data.py" --validate-only || FORCE_DATA=1
fi
if [[ "${FORCE_DATA:-0}" == "1" || ! -s "$V2_TRAIN" || ! -s "$V2_VAL" || ! -s "$V2_MANIFEST" ]]; then
  echo "=== Generate Probe v2 data ==="
  uv run python "$EXP/scripts/make_probe_v2_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_probe_v2_data.py" --validate-only || exit 1

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if has_adapter "$ADAPTER"; then
    echo "=== skip Probe v2 training (adapter complete) ==="
  else
    cd finetune
    uv run axolotl train ../experiments/13-iterative-hardening/train-sft-4b-probe-v2-400.yaml --launcher python || exit 1
    cd "$SLURM_SUBMIT_DIR"
  fi
fi
if ! has_adapter "$ADAPTER"; then
  echo "Missing Probe v2 adapter: $ADAPTER"
  exit 1
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  cd eval
  uv sync || exit 1
  for stages in 1 2; do
    out=../$EXP/results/probe-v2-v13-stage$stages
    if [[ "${FORCE_EVAL:-0}" == "1" || ! -f "$out/results.json" ]]; then
      uv run python eval_new.py \
        --config ../$EXP/eval-probe-v2-v13.yaml \
        --stages "$stages" \
        --output-dir "$out" || exit 1
    fi
  done
  if [[ -s "../$V12_TEST" ]]; then
    out=../$EXP/results/probe-v2-v12-stage1
    if [[ "${FORCE_EVAL:-0}" == "1" || ! -f "$out/results.json" ]]; then
      uv run python eval_new.py \
        --config ../$EXP/eval-probe-v2-v12-retention.yaml \
        --stages 1 \
        --output-dir "$out" || exit 1
    fi
  fi
  cd "$SLURM_SUBMIT_DIR"
fi

uv run --no-project python "$EXP/scripts/summarize_probe_v2.py" || true
echo "Probe v2 summary: $EXP/results/PROBE-V2-SUMMARY.md"
