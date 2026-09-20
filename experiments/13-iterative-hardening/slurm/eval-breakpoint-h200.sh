#!/bin/bash
# Evaluate the frozen V13.1 structural breakpoint suite. No training.
# Default: native V13 6K only. Select comparisons with ONLY=base,sft-1.5k,sft-6k.
#SBATCH --job-name=spatial13-break
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
DATA=data/spatial_v13_breakpoint_test.jsonl
MANIFEST=data/spatial_v13_breakpoint_manifest.json
RESULTS=$EXP/results
DIAGNOSTIC=data/spatial_v13_diagnostic_test.jsonl
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

CELLS=(
  "base      eval-breakpoint-base.yaml     -"
  "sft-1.5k  eval-breakpoint-sft-1500.yaml $EXP/models/qwen3.5-4b-v13-sft-1500"
  "sft-6k    eval-breakpoint-sft-6000.yaml $EXP/models/qwen3.5-4b-v13-sft-6000"
)

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

want_tag() {
  local tag=$1
  local IFS=, candidate
  for candidate in ${ONLY:-sft-6k}; do
    [[ "$candidate" == "$tag" ]] && return 0
  done
  return 1
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
mkdir -p "$EXP/logs" "$RESULTS" data
nvidia-smi -L || true

cd eval
srun uv sync || exit 1
cd "$SLURM_SUBMIT_DIR"

if [[ ! -s "$DIAGNOSTIC" ]]; then
  uv run python spatial/generate_diagnostic_v13.py \
    --out data/spatial_v13_diagnostic.jsonl --seed 1313 || exit 1
fi
uv run --no-project python "$EXP/scripts/validate_diagnostic_data.py" "$DIAGNOSTIC" || exit 1
if [[ ! -s "$V1_TRAIN" || ! -s "$V1_VAL" ]]; then
  uv run python "$EXP/scripts/make_probe_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_probe_data.py" --validate-only || exit 1
if [[ ! -s "$V2_TRAIN" || ! -s "$V2_VAL" ]]; then
  uv run python "$EXP/scripts/make_probe_v2_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_probe_v2_data.py" --validate-only || exit 1
if [[ ! -s "$SFT15_TRAIN" || ! -s "$SFT15_VAL" || ! -s "$SFT15_MANIFEST" ]]; then
  uv run python "$EXP/scripts/make_sft_1500_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_sft_1500_data.py" --validate-only || exit 1
if [[ ! -s "$SFT6_TRAIN" || ! -s "$SFT6_VAL" || ! -s "$SFT6_MANIFEST" ]]; then
  uv run python "$EXP/scripts/make_sft_6000_data.py" || exit 1
fi
uv run python "$EXP/scripts/make_sft_6000_data.py" --validate-only || exit 1

if [[ "${FORCE_DATA:-0}" != "1" && -s "$DATA" && -s "$MANIFEST" ]]; then
  uv run --no-project python "$EXP/scripts/make_breakpoint_data.py" --validate-only || FORCE_DATA=1
fi
if [[ "${FORCE_DATA:-0}" == "1" || ! -s "$DATA" || ! -s "$MANIFEST" ]]; then
  echo "=== Generate frozen V13.1 breakpoint suite (1,224 rows, seed 13700) ==="
  uv run --no-project python "$EXP/scripts/make_breakpoint_data.py" || exit 1
fi
uv run --no-project python "$EXP/scripts/make_breakpoint_data.py" --validate-only || exit 1

selected=0
for row in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  tag=$1 config=$2 adapter=$3
  want_tag "$tag" || continue
  selected=1
  if [[ "$adapter" != "-" ]] && ! has_adapter "$adapter"; then
    echo "Missing adapter for $tag at $adapter"
    exit 1
  fi
  out=$RESULTS/breakpoint-$tag-stage1
  if [[ "${FORCE_EVAL:-0}" != "1" && -f "$out/results.json" ]]; then
    echo "=== skip $tag (already complete) ==="
    continue
  fi
  echo "=== evaluate $tag on V13.1 breakpoint suite ==="
  cd eval
  uv run python eval_new.py \
    --config ../$EXP/"$config" \
    --stages 1 \
    --output-dir ../"$out" || exit 1
  cd "$SLURM_SUBMIT_DIR"
done

if [[ "$selected" -eq 0 ]]; then
  echo "No model selected (ONLY=${ONLY-sft-6k})"
  exit 1
fi

uv run --no-project python "$EXP/scripts/summarize_breakpoint.py" || true
echo "Summary: $RESULTS/BREAKPOINT-SUMMARY.md"
echo "All cells: $RESULTS/BREAKPOINT-CELLS.csv"
