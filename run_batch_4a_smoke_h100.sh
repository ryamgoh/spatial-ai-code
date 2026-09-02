#!/bin/bash
# Exp 4a — Qwen3.5-4B full-pipeline smoke on 2x H100-47 (MIG 3g.47gb).
#
# Pipeline (all artifacts land under experiments/04a-smoke/):
#   1. SFT QLoRA on Qwen3.5-4B          -> models/qwen3.5-4b-sft
#   2. Merge SFT LoRA into bf16         -> models/qwen3.5-4b-sft-merged
#   3. Generate ~300 GRPO prompts       -> ../spatial_grpo_data_smoke4a.jsonl
#   4. GRPO 20 steps (2-GPU vLLM+train) -> models/qwen3.5-4b-grpo-smoke
#   5. Eval base / SFT / GRPO on the 16-row stratified TQA-CORR set
#      -> results/smoke16/{base,sft,grpo}/   +   results/smoke16/SUMMARY.md
#
# Overrides:
#   SKIP_TRAIN=1   skip SFT + merge + GRPO (re-run evals only)
#   SKIP_EVAL=1    skip the eval phase (train only)
#
# Submit:  sbatch run_batch_4a_smoke_h100.sh
#SBATCH --job-name=spatial4a-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100-47:2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/04a-smoke
SFT_ADAPTER=$EXP/models/qwen3.5-4b-sft
MERGED=$EXP/models/qwen3.5-4b-sft-merged
GRPO_ADAPTER=$EXP/models/qwen3.5-4b-grpo-smoke
GRPO_DATA=spatial_grpo_data_smoke4a.jsonl
SMOKE_OUT=$EXP/results/smoke16
# Absolute forms of the same paths, for file checks/relocations done after
# `cd finetune` / `cd eval` (yaml paths stay relative — resolved by each
# tool from its own CWD, which is what the configs assume).
A_SFT_ADAPTER=$SLURM_SUBMIT_DIR/$SFT_ADAPTER
A_MERGED=$SLURM_SUBMIT_DIR/$MERGED
A_GRPO_ADAPTER=$SLURM_SUBMIT_DIR/$GRPO_ADAPTER
A_GRPO_DATA=$SLURM_SUBMIT_DIR/$GRPO_DATA
A_SMOKE_OUT=$SLURM_SUBMIT_DIR/$SMOKE_OUT

# An adapter dir is only "complete" when it has both the config AND the
# weights: axolotl pre-saves adapter_config.json/tokenizer at the START of
# training, so its presence alone does not mean training finished.
has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

# ── GPU sanity (h100-47 is MIG 3g.47gb; pin 0/1 not MIG-UUIDs) ──────────
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true
n_vis=0
if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
  IFS=',' read -ra _devs <<< "$CUDA_VISIBLE_DEVICES"
  n_vis=${#_devs[@]}
fi
n_mig=$(nvidia-smi -L 2>/dev/null | grep -c 'MIG-' || true)
n_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "device count: CUDA_VISIBLE_DEVICES=${n_vis} MIG=${n_mig} GPU=${n_gpu}"
# nvidia-smi -L often lists only the parent GPU (GPU=1, MIG=0) even when
# SLURM gave two MIG UUIDs in CUDA_VISIBLE_DEVICES.
if [[ "${n_vis}" -lt 2 && "${n_mig}" -lt 2 && "${n_gpu}" -lt 2 ]]; then
  echo "Need 2 GPUs or MIG slices. Check: scontrol show job ${SLURM_JOB_ID-} | grep -E 'GRES|TRES'"
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0,1 (vLLM cannot parse UUIDs)"
  export CUDA_VISIBLE_DEVICES=0,1
fi

# ── 1-2-3-4: TRAIN (SFT -> merge -> GRPO data -> GRPO) ───────────────────
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  cd finetune
  srun uv sync --extra vllm

  if has_adapter "$A_SFT_ADAPTER"; then
    echo "=== [1/5] SFT adapter already complete, skipping ==="
  else
    if [[ -f "$A_SFT_ADAPTER/adapter_config.json" ]]; then
      echo "Found an incomplete SFT adapter (config, no weights) at $A_SFT_ADAPTER; retraining"
    fi
    echo "=== [1/5] SFT QLoRA on Qwen3.5-4B ==="
    srun --cpu-bind=cores uv run python finetune.py ../experiments/04a-smoke/train-sft-4b.yaml || {
      echo "SFT training FAILED — aborting. Check the log for the traceback."
      exit 1
    }
  fi
  if ! has_adapter "$A_SFT_ADAPTER"; then
    echo "SFT did not produce a complete adapter (no adapter_model.* at $A_SFT_ADAPTER); aborting"
    exit 1
  fi

  if [[ -f "$A_MERGED/config.json" ]]; then
    echo "=== [2/5] Merged SFT weights already present, skipping ==="
  else
    if [[ -d "$A_MERGED" ]]; then
      echo "Removing partial merge output at $A_MERGED from a failed earlier run"
      find "$A_MERGED" -depth -delete 2>/dev/null || true
    fi
    echo "=== [2/5] Merge SFT LoRA into bf16 (adapter left untouched) ==="
    CUDA_VISIBLE_DEVICES="" srun uv run python merge_sft.py \
      --base Qwen/Qwen3.5-4B \
      --adapter "$A_SFT_ADAPTER" \
      --out "$A_MERGED" || {
      echo "Merge FAILED — aborting. Delete $A_MERGED before re-running."
      exit 1
    }
  fi

  echo "=== [3/5] Generate GRPO prompt data (~300 rows, seeded) ==="
  if [[ -s "$A_GRPO_DATA" ]]; then
    echo "GRPO data already at $A_GRPO_DATA, regenerating would be identical (seeded); skipping"
  else
    srun uv run python generate_grpo.py --n 300 --seed 42 --out "../$GRPO_DATA"
  fi
  srun uv run python generate_grpo.py --annotate --out "../$GRPO_DATA"

  echo "=== [4/5] GRPO 20 steps (GPU0 vLLM serve, GPU1 train) ==="
  CFG=../experiments/04a-smoke/train-grpo-4b-smoke.yaml
  CUDA_VISIBLE_DEVICES=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    uv run axolotl vllm-serve "$CFG" &
  VLLM_PID=$!
  trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

  ok=0
  for _ in $(seq 1 90); do
    if curl -sfL "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1 \
      || curl -sfL "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      ok=1
      break
    fi
    # Fail fast if the server process already died (e.g. bad base_model).
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vllm-serve exited during startup — check the log, aborting"
      exit 1
    fi
    sleep 10
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "vLLM server did not become ready"
    exit 1
  fi

  RESUME=()
  if compgen -G "$A_GRPO_ADAPTER/checkpoint-*" > /dev/null; then
    echo "Resuming from existing GRPO checkpoints"
    RESUME=(--resume)
  fi

  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK || true
  CUDA_VISIBLE_DEVICES=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    uv run python finetune.py ../experiments/04a-smoke/train-grpo-4b-smoke.yaml "${RESUME[@]}" || {
    echo "GRPO training FAILED — aborting."
    exit 1
  }
  kill "$VLLM_PID" 2>/dev/null || true
  cd ..
else
  echo "SKIP_TRAIN=1 — using existing adapters/models"
fi

# ── 5: EVALS (base / SFT / GRPO, 16-row stratified TQA-CORR) ─────────────
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  cd eval
  srun uv sync

  # Resolve GRPO adapter: final dir, else newest checkpoint. (absolute)
  grpo_lora=$A_GRPO_ADAPTER
  if ! has_adapter "$grpo_lora"; then
    latest=$(ls -d "$grpo_lora"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [[ -n "${latest:-}" ]] && has_adapter "$latest"; then
      echo "Final GRPO adapter missing; using $latest"
      grpo_lora=$latest
    else
      echo "WARN: no complete GRPO adapter at $grpo_lora — grpo eval skipped, continuing with base+sft"
      grpo_lora=""
    fi
  fi

  run_eval() {
    local tag=$1 cfg=$2 lora=$3
    local cfg_to_use=$cfg
    # If a checkpoint (not the final dir) was resolved, rewrite lora_path.
    # lora_path is relative to eval/ (the CWD), same as in the yaml.
    if [[ -n "$lora" && "$lora" != "$A_GRPO_ADAPTER" ]]; then
      cfg_to_use=$(mktemp "../$EXP/eval-grpo-4b.XXXX.yaml")
      sed "s|lora_path:.*|lora_path: ../${lora#$SLURM_SUBMIT_DIR/}|" "$cfg" > "$cfg_to_use"
      trap "rm -f '$cfg_to_use'" EXIT
    fi
    echo "=== [5/5] Eval $tag ($cfg_to_use) ==="
    local status=0
    CUDA_VISIBLE_DEVICES=0 srun --cpu-bind=cores uv run python eval_new.py --config "$cfg_to_use" || status=$?
    # Move the newest timestamped results dir into smoke16/<tag>/.
    # eval_new.py writes to <config dir>/results/<timestamp>/
    local latest
    latest=$(ls -dt "$SLURM_SUBMIT_DIR/$EXP"/results/2*/ 2>/dev/null | head -1 || true)
    mkdir -p "$A_SMOKE_OUT/$tag"
    if [[ -n "${latest:-}" && -f "${latest}results.json" ]]; then
      mv "${latest}"* "$A_SMOKE_OUT/$tag/"
      rmdir "${latest}" 2>/dev/null || true
      echo "  -> $A_SMOKE_OUT/$tag/"
    fi
    [[ $status -ne 0 ]] && echo "  eval $tag FAILED (status $status)" || echo "  eval $tag OK"
  }

  run_eval base  ../experiments/04a-smoke/eval-base-4b.yaml  ""
  run_eval sft   ../experiments/04a-smoke/eval-sft-4b.yaml   ""
  if [[ -n "${grpo_lora:-}" ]]; then
    run_eval grpo  ../experiments/04a-smoke/eval-grpo-4b.yaml "$grpo_lora"
  fi

  # Per-config results are now split under $SMOKE_OUT/; build the summary.
  srun uv run --no-project python ../experiments/04a-smoke/scripts/summarize.py \
    || echo "summarize failed — check $SMOKE_OUT manually"
fi

echo "=== Exp 4a smoke done. Summary: $A_SMOKE_OUT/SUMMARY.md ==="
