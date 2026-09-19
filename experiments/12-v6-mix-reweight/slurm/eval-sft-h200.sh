#!/bin/bash
# Exp 12 — eval-only: 2k TEST + SpatialMap v6.
# Optional SCORE_CKPTS=1 scores checkpoints on the 1k VAL (overfit pick).
#
#   sbatch experiments/12-v6-mix-reweight/slurm/eval-sft-h200.sh
#   ONLY=4b-1.5k sbatch experiments/12-v6-mix-reweight/slurm/eval-sft-h200.sh
#   SCORE_CKPTS=1 sbatch experiments/12-v6-mix-reweight/slurm/eval-sft-h200.sh
#   FORCE=1 sbatch experiments/12-v6-mix-reweight/slurm/eval-sft-h200.sh
#SBATCH --job-name=spatial12-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=experiments/12-v6-mix-reweight/logs/%x-%j.out
#SBATCH --error=experiments/12-v6-mix-reweight/logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

EXP=experiments/12-v6-mix-reweight
EVAL_OUT=$SLURM_SUBMIT_DIR/$EXP/results
SYNTH_TEST=$SLURM_SUBMIT_DIR/data/spatial_sft_v12_2000_test.jsonl
VAL=$SLURM_SUBMIT_DIR/data/spatial_sft_v12_1000_val.jsonl
SMAP=$SLURM_SUBMIT_DIR/data/spatialeval_v6_corr.jsonl

CELLS=(
  "4b-1.5k  models/qwen3.5-4b-sft-v12-1500  eval-sft-4b-1500.yaml  eval-sft-4b-1500-val.yaml"
  "4b-6k    models/qwen3.5-4b-sft-v12-6000  eval-sft-4b-6000.yaml  eval-sft-4b-6000-val.yaml"
  "4b-18k   models/qwen3.5-4b-sft-v12-18000 eval-sft-4b-18000.yaml eval-sft-4b-18000-val.yaml"
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
  return 0
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0"
  export CUDA_VISIBLE_DEVICES=0
fi
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
echo "ONLY=${ONLY-} FORCE=${FORCE-0} SCORE_CKPTS=${SCORE_CKPTS-0}"
nvidia-smi -L || true

if [[ ! -s "$SMAP" ]]; then
  echo "=== Build SpatialMap v6 corr ==="
  cd finetune
  uv run --no-project --with typer python \
    ../experiments/11-v6-synthetic/scripts/make_spatialmap_v6.py || exit 1
  cd "$SLURM_SUBMIT_DIR"
fi
if [[ ! -s "$VAL" && -s "$SYNTH_TEST" ]]; then
  echo "=== Slice val/test from pool if needed ==="
  uv run --no-project python experiments/12-v6-mix-reweight/scripts/make_v12_data.py || true
fi
if [[ ! -s "$SMAP" ]]; then
  echo "Missing $SMAP"
  exit 1
fi
if [[ ! -s "$SYNTH_TEST" ]]; then
  echo "Missing $SYNTH_TEST (Exp 12 matched 2k holdout)"
  exit 1
fi

cd eval
srun uv sync

ran=0
for row in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  tag=$1 adapter=$2 ev=$3 hard_ev=$4
  want_tag "$tag" || continue
  dest=$SLURM_SUBMIT_DIR/$EXP/$adapter
  out=$EVAL_OUT/$tag
  if ! has_adapter "$dest"; then
    echo "WARN: no adapter at $dest — skip eval $tag"
    continue
  fi
  if [[ "${FORCE:-0}" != "1" && -f "$out/results.json" ]]; then
    echo "=== skip eval $tag (already have results.json) ==="
  else
    echo "=== eval $tag (matched 2k + SpatialMap v6) ==="
    mkdir -p "$out"
    srun --cpu-bind=cores uv run python eval_new.py \
      --config "../$EXP/$ev" \
      --output-dir "$out" || {
      echo "Exp 12 eval $tag FAILED"
      exit 1
    }
    echo "  eval $tag OK -> $out/"
    ran=1
  fi

  if [[ "${SCORE_CKPTS:-0}" == "1" ]]; then
    if [[ ! -s "$VAL" ]]; then
      echo "WARN: no $VAL — skip checkpoint scoring"
      continue
    fi
    echo "=== score checkpoints for $tag on 1k VAL ==="
    shopt -s nullglob
    ckpts=("$dest" "$dest"/checkpoint-*)
    shopt -u nullglob
    for ckpt in "${ckpts[@]}"; do
      has_adapter "$ckpt" || continue
      name=$(basename "$ckpt")
      if [[ "$ckpt" == "$dest" ]]; then
        name=final
      fi
      ckpt_out=$out/ckpts/$name
      if [[ "${FORCE:-0}" != "1" && -f "$ckpt_out/results.json" ]]; then
        echo "=== skip ckpt $name (already have results.json) ==="
        continue
      fi
      tmp=$out/ckpts/${name}.yaml
      uv run --no-project python \
        "$SLURM_SUBMIT_DIR/$EXP/scripts/rewrite_eval_lora.py" \
        "$SLURM_SUBMIT_DIR/$EXP/$hard_ev" "$tmp" "$ckpt" || exit 1
      mkdir -p "$ckpt_out"
      echo "=== val eval $tag/$name ==="
      srun --cpu-bind=cores uv run python eval_new.py \
        --config "$tmp" \
        --output-dir "$ckpt_out" || {
        echo "Exp 12 val eval $tag/$name FAILED"
        exit 1
      }
      ran=1
    done
    cd "$SLURM_SUBMIT_DIR"
    uv run --no-project python \
      experiments/12-v6-mix-reweight/scripts/score_checkpoints.py "$tag" || true
    cd eval
  fi
done

if [[ "$ran" -eq 0 ]]; then
  echo "Nothing to eval (filtered out, missing adapters, or results already present)."
fi
cd "$SLURM_SUBMIT_DIR"
uv run --no-project python experiments/12-v6-mix-reweight/scripts/summarize.py || true
echo "    uv run --no-project python experiments/12-v6-mix-reweight/scripts/summarize.py"
