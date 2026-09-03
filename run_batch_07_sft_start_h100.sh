#!/bin/bash
# Exp 7 — Architectural Starting State for SFT.
# Same 5k single-answer traces, same 4b QLoRA recipe; IV = Base vs Instruct.
#
#   0. generate_all.py  5000 single-answer  -> data/spatial_sft_single_5000_train.jsonl
#   1. axolotl train    Qwen3.5-4B          -> models/qwen3.5-4b-instruct-sft
#   2. axolotl train    Qwen3.5-4B-Base     -> models/qwen3.5-4b-base-sft
#   3. eval_new.py      both on TQA-Corr-Single
#   4. summarize.py
#
# Submit:  sbatch run_batch_07_sft_start_h100.sh
# Overrides: SKIP_TRAIN=1 SKIP_EVAL=1 SKIP_BASE=1 SKIP_INSTRUCT=1
#SBATCH --job-name=spatial7-sft
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/07-sft-starting-state
SFT_DATA=data/spatial_sft_single_5000_train.jsonl
A_SFT_DATA=$SLURM_SUBMIT_DIR/$SFT_DATA
BASE_ADAPTER=$EXP/models/qwen3.5-4b-base-sft
INSTRUCT_ADAPTER=$EXP/models/qwen3.5-4b-instruct-sft
A_BASE_ADAPTER=$SLURM_SUBMIT_DIR/$BASE_ADAPTER
A_INSTRUCT_ADAPTER=$SLURM_SUBMIT_DIR/$INSTRUCT_ADAPTER
EVAL_OUT=$EXP/results/starting-state
A_EVAL_OUT=$SLURM_SUBMIT_DIR/$EVAL_OUT
mkdir -p "$A_EVAL_OUT"

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1

echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  cd finetune
  export PYTHONPATH="$SLURM_SUBMIT_DIR/finetune${PYTHONPATH:+:$PYTHONPATH}"
  srun uv sync

  if [[ -s "$A_SFT_DATA" ]]; then
    echo "=== [0/4] SFT data already at $SFT_DATA ==="
  else
    echo "=== [0/4] Generate 5000 single-answer traces (seed 42) ==="
    srun uv run python generate_all.py \
      --out ../data/spatial_sft_single_5000.jsonl \
      --test-split 0 \
      --seed 42 \
      --num-type0-1-answer 1599 \
      --num-type0-2-answer 0 \
      --num-type0-4-answer 0 \
      --num-type1-1-answer 1455 \
      --num-type1-2-answer 0 \
      --num-type2 1946 || {
      echo "SFT data generation FAILED — aborting."
      exit 1
    }
  fi
  if [[ ! -s "$A_SFT_DATA" ]]; then
    echo "Missing $A_SFT_DATA after generate; aborting"
    exit 1
  fi

  train_one() {
    local tag=$1 cfg=$2 dest=$3
    if has_adapter "$dest"; then
      echo "=== skip SFT $tag (adapter complete) ==="
      return 0
    fi
    echo "=== SFT QLoRA $tag ==="
    srun --cpu-bind=cores uv run axolotl train "$cfg" --launcher python || {
      echo "SFT $tag FAILED — aborting."
      exit 1
    }
    if ! has_adapter "$dest"; then
      echo "SFT $tag did not produce a complete adapter at $dest; aborting"
      exit 1
    fi
  }

  if [[ "${SKIP_INSTRUCT:-0}" != "1" ]]; then
    train_one instruct ../experiments/07-sft-starting-state/train-sft-instruct.yaml "$A_INSTRUCT_ADAPTER"
  fi
  if [[ "${SKIP_BASE:-0}" != "1" ]]; then
    train_one base ../experiments/07-sft-starting-state/train-sft-base.yaml "$A_BASE_ADAPTER"
  fi
  cd ..
else
  echo "SKIP_TRAIN=1 — using existing adapters"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  cd eval
  srun uv sync

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
    local status=0
    srun --cpu-bind=cores uv run python eval_new.py --config "$cfg" || status=$?
    local latest
    latest=$(ls -dt "$SLURM_SUBMIT_DIR/$EXP"/results/2*/ 2>/dev/null | head -1 || true)
    mkdir -p "$A_EVAL_OUT/$tag"
    if [[ -n "${latest:-}" && -f "${latest}results.json" ]]; then
      mv "${latest}"* "$A_EVAL_OUT/$tag/"
      rmdir "${latest}" 2>/dev/null || true
      echo "  -> $A_EVAL_OUT/$tag/"
    fi
    if [[ $status -ne 0 ]]; then
      echo "  eval $tag FAILED (status $status)"
      return "$status"
    fi
    echo "  eval $tag OK"
  }

  if [[ "${SKIP_INSTRUCT:-0}" != "1" ]]; then
    run_eval instruct-sft ../experiments/07-sft-starting-state/eval-sft-instruct.yaml "$A_INSTRUCT_ADAPTER"
  fi
  if [[ "${SKIP_BASE:-0}" != "1" ]]; then
    run_eval base-sft ../experiments/07-sft-starting-state/eval-sft-base.yaml "$A_BASE_ADAPTER"
  fi

  srun uv run --no-project python ../experiments/07-sft-starting-state/scripts/summarize.py \
    || echo "summarize failed — check $EVAL_OUT manually"
fi

echo "=== Exp 7 starting-state done. Summary: $A_EVAL_OUT/SUMMARY.md ==="
