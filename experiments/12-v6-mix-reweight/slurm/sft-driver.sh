#!/bin/bash
# Exp 12 — v6 mix-reweight SFT body (1 GPU, --launcher python).
# Sourced by the h100-47 wrapper. Do not sbatch this file.
#
#   0. generate_all_v6 21k seed 52 (v12 prompt)
#   1. 2k TEST + 1k VAL + nested 1.5k ⊂ 6k ⊂ 18k train (disjoint, matched mix)
#   2. axolotl train (NLL on 200-row slice of val)
#   3. eval_new.py on 2k TEST + SpatialMap; ckpt pick on 1k VAL
#
# Overrides: SKIP_TRAIN=1 SKIP_EVAL=1 ONLY=4b-1.5k
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

EXP=experiments/12-v6-mix-reweight
PROMPT=$EXP/system_prompt.txt
POOL=data/spatial_sft_v12_21000_pool.jsonl
VAL=data/spatial_sft_v12_1000_val.jsonl
NLL=data/spatial_sft_v12_val_nll.jsonl
TRAIN15=data/spatial_sft_v12_1500_train.jsonl
SYNTH_TEST=data/spatial_sft_v12_2000_test.jsonl
SMAP=data/spatialeval_v6_corr.jsonl
A_POOL=$SLURM_SUBMIT_DIR/$POOL
A_VAL=$SLURM_SUBMIT_DIR/$VAL
A_NLL=$SLURM_SUBMIT_DIR/$NLL
A_TRAIN15=$SLURM_SUBMIT_DIR/$TRAIN15
A_SYNTH=$SLURM_SUBMIT_DIR/$SYNTH_TEST
A_SMAP=$SLURM_SUBMIT_DIR/$SMAP
EVAL_OUT=$EXP/results
A_EVAL_OUT=$SLURM_SUBMIT_DIR/$EVAL_OUT
mkdir -p "$A_EVAL_OUT"

# tag  train_yaml  adapter_rel  eval_yaml  hard_yaml
CELLS=(
  "4b-1.5k  train-sft-4b-1500.yaml  models/qwen3.5-4b-sft-v12-1500  eval-sft-4b-1500.yaml  eval-sft-4b-1500-val.yaml"
  "4b-6k    train-sft-4b-6000.yaml  models/qwen3.5-4b-sft-v12-6000  eval-sft-4b-6000.yaml  eval-sft-4b-6000-val.yaml"
  "4b-18k   train-sft-4b-18000.yaml models/qwen3.5-4b-sft-v12-18000 eval-sft-4b-18000.yaml eval-sft-4b-18000-val.yaml"
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

selected=()
for row in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  tag=$1 cfg=$2 adapter=$3 ev=$4 hard=$5
  if want_tag "$tag"; then
    selected+=("$tag $cfg $adapter $ev $hard")
  fi
done

if [[ ${#selected[@]} -eq 0 ]]; then
  echo "No cells selected (ONLY=${ONLY-})"
  exit 1
fi
echo "Selected cells (H100-47:1, 1-GPU python, cpus-per-task pinned):"
printf '  %s\n' "${selected[@]}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1

echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0"
  export CUDA_VISIBLE_DEVICES=0
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  cd finetune
  export PYTHONPATH="$SLURM_SUBMIT_DIR/finetune${PYTHONPATH:+:$PYTHONPATH}"
  uv sync || exit 1

  mkdir -p "$SLURM_SUBMIT_DIR/data"
  exec 9>"$SLURM_SUBMIT_DIR/data/.spatial_sft_v12.lock"
  flock 9

  if [[ -s "$A_SMAP" ]]; then
    echo "=== SpatialMap v6 already at $SMAP ==="
  else
    echo "=== Build SpatialMap v6 corr ==="
    uv run --no-project --with typer python \
      ../experiments/11-v6-synthetic/scripts/make_spatialmap_v6.py || exit 1
  fi

  if [[ -s "$A_POOL" ]]; then
    echo "=== [0] v12 21k pool already at $POOL ==="
  else
    echo "=== [0] Generate 21000 v12 traces (seed 52, hierarchical uniform) ==="
    uv run --no-project --with typer python generate_all_v6.py \
      --out ../data/spatial_sft_v12_21000_pool.jsonl \
      --test-split 0 \
      --seed 52 \
      --system-prompt-file "../$PROMPT" \
      --shuffle-special \
      --shuffle-none-phrase \
      --num-type0-1-answer 1167 \
      --num-type0-2-answer 1167 \
      --num-type0-undetermined 1166 \
      --num-type0-cycle 1166 \
      --num-type0-incomplete-pair 1166 \
      --num-type0-none 1167 \
      --num-type1-1-answer 1401 \
      --num-type1-2-answer 1401 \
      --num-type1-3-answer 1401 \
      --num-type1-4-answer 1399 \
      --num-type1-none 1399 \
      --num-type2 3500 \
      --num-type2-none 3500 || {
      echo "v12 pool generation FAILED — aborting."
      exit 1
    }
    if [[ -s "$SLURM_SUBMIT_DIR/data/spatial_sft_v12_21000_pool_train.jsonl" ]]; then
      mv "$SLURM_SUBMIT_DIR/data/spatial_sft_v12_21000_pool_train.jsonl" "$A_POOL"
    elif [[ -s "$SLURM_SUBMIT_DIR/data/spatial_sft_v12_21000_pool.jsonl" ]]; then
      : # already named
    fi
  fi
  if [[ ! -s "$A_POOL" ]]; then
    echo "Missing $A_POOL after generate; aborting"
    exit 1
  fi

  echo "=== [0] 2k TEST + 1k VAL + nested 1.5k ⊂ 6k ⊂ 18k train ==="
  uv run --no-project python ../experiments/12-v6-mix-reweight/scripts/make_v12_data.py || {
    echo "make_v12_data.py FAILED — aborting."
    exit 1
  }
  flock -u 9

  if [[ ! -s "$A_TRAIN15" ]]; then
    echo "Missing $A_TRAIN15"
    exit 1
  fi
  if [[ ! -s "$A_SYNTH" || ! -s "$A_VAL" || ! -s "$A_NLL" ]]; then
    echo "Missing test/val/nll after slicer"
    exit 1
  fi

  train_one() {
    local tag=$1 cfg=$2 dest=$3
    if has_adapter "$dest"; then
      echo "=== skip SFT $tag (adapter complete) ==="
      return 0
    fi
    echo "=== SFT QLoRA $tag ==="
    uv run axolotl train "$cfg" --launcher python || {
      echo "SFT $tag FAILED — aborting."
      exit 1
    }
    if ! has_adapter "$dest"; then
      echo "SFT $tag did not produce a complete adapter at $dest; aborting"
      exit 1
    fi
  }

  for row in "${selected[@]}"; do
    # shellcheck disable=SC2086
    set -- $row
    tag=$1 cfg=$2 adapter=$3
    train_one "$tag" "../experiments/12-v6-mix-reweight/$cfg" "$SLURM_SUBMIT_DIR/$EXP/$adapter"
  done
  cd ..
else
  echo "SKIP_TRAIN=1 — using existing adapters"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  if [[ ! -s "$A_SYNTH" ]]; then
    echo "Missing $A_SYNTH — Exp 12 synth eval needs data/spatial_sft_v12_2000_test.jsonl"
    exit 1
  fi
  if [[ ! -s "$A_SMAP" ]]; then
    echo "Missing $A_SMAP — cannot eval SpatialMap v6"
    exit 1
  fi
  cd eval
  uv sync || exit 1

  run_eval() {
    local tag=$1 cfg=$2 adapter=$3
    if [[ -n "$adapter" ]] && ! has_adapter "$adapter"; then
      echo "WARN: no adapter at $adapter — skip eval $tag"
      return 0
    fi
    if [[ -f "$A_EVAL_OUT/$tag/results.json" ]]; then
      echo "=== skip eval $tag (already have results.json) ==="
      return 0
    fi
    echo "=== eval $tag ($cfg) ==="
    mkdir -p "$A_EVAL_OUT/$tag"
    local status=0
    uv run python eval_new.py \
      --config "$cfg" \
      --output-dir "$A_EVAL_OUT/$tag" || status=$?
    if [[ $status -ne 0 ]]; then
      echo "  eval $tag FAILED (status $status)"
      return "$status"
    fi
    echo "  eval $tag OK -> $A_EVAL_OUT/$tag/"
  }

  for row in "${selected[@]}"; do
    # shellcheck disable=SC2086
    set -- $row
    tag=$1 cfg=$2 adapter=$3 ev=$4
    run_eval "$tag" "../experiments/12-v6-mix-reweight/$ev" "$SLURM_SUBMIT_DIR/$EXP/$adapter"
  done
  cd ..
  uv run --no-project python experiments/12-v6-mix-reweight/scripts/summarize.py || true
fi

echo "=== Exp 12 cells done. After the 47 job finishes:"
echo "    sbatch experiments/12-v6-mix-reweight/slurm/eval-sft-h200.sh"
echo "    SCORE_CKPTS=1 sbatch experiments/12-v6-mix-reweight/slurm/eval-sft-h200.sh"
echo "    uv run --no-project python experiments/12-v6-mix-reweight/scripts/summarize.py"
