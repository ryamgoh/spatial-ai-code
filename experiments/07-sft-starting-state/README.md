# Experiment 7 — Architectural Starting State for SFT

**Source.** Post-thesis Phase 1. Same train/eval stack as Exp 4b
(`axolotl train --launcher python`, `eval_new.py`).

**Question.** Does fine-tuning a raw base model beat fine-tuning an
instruction-tuned model on SpatialMap-TQA-Corr-Single?

**Hypothesis.** SFT from `Qwen3.5-4B-Base` ends higher on task accuracy.
The instruct checkpoint (`Qwen3.5-4B`) carries alignment tax / catastrophic
forgetting and is more resistant to this specialised format.

## Independent / dependent variables

| | |
|---|---|
| IV | Starting checkpoint (`Qwen/Qwen3.5-4B-Base` vs `Qwen/Qwen3.5-4B`) |
| DV | Spatial strict/loose acc; formatting adherence; loss curve (steps, train/eval loss) |
| Constants | Eval = TQA-Corr-Single (1,038); train = 5,000 single-answer AxisDecomposition traces; Exp 4b QLoRA hyperparameters |

Zero-shot Non-shot-2 numbers live in Exp 6, not here. This run’s eval prompt
is the **SFT system prompt** baked into `generate_all.py` traces (same as
Exp 3/4b `eval-sft`), so train and test match.

## Training data (5,000, single-letter gold only)

Not the 1,038 eval rows. Synthetic traces from `finetune/generate_all.py`,
type mix matched to TQA-Corr-Single (count 404 / dir 332 / which 302 of 1,038):

| type | 1-ans n | share |
|---|---|---|
| type0 dir | 1,599 | 332/1038 |
| type1 which | 1,455 | 302/1038 |
| type2 count | 1,946 | 404/1038 |
| multi-answer | **0** | — |
| **total train** | **5,000** | seed 42, `--test-split 0` |

Writes `data/spatial_sft_single_5000_train.jsonl`. Axolotl then holds out
5% for `eval_loss` (same as 4b). Launcher generates the file if missing.

## Systems

| id | system | train | eval |
|---|---|---|---|
| B-SFT | Qwen3.5-4B-Base + QLoRA | `train-sft-base.yaml` | `eval-sft-base.yaml` |
| I-SFT | Qwen3.5-4B + QLoRA | `train-sft-instruct.yaml` | `eval-sft-instruct.yaml` |

Hyperparameters copied from `../04b-cli/train-sft-4b.yaml` (QLoRA r=64,
lr 1e-4, 2 epochs, seq 4096, microbatch 1, grad acc 8, early stopping on
`eval_loss`). Only `base_model` and `output_dir` differ.

## Run (1× H100-47)

```bash
sbatch run_batch_07_sft_start_h100.sh
```

Idempotent: existing adapter / 5k jsonl / `results.json` are skipped.
Overrides: `SKIP_TRAIN=1`, `SKIP_EVAL=1`, `SKIP_BASE=1`, `SKIP_INSTRUCT=1`.

Evals land in `results/starting-state/{base-sft,instruct-sft}/`.
`scripts/summarize.py` writes `SUMMARY.md` (Single acc, format flags,
trainer loss curves).

## Artifacts

| artifact | location |
|---|---|
| 5k single-answer SFT jsonl | `data/spatial_sft_single_5000_train.jsonl` |
| Base SFT adapter | `models/qwen3.5-4b-base-sft/` |
| Instruct SFT adapter | `models/qwen3.5-4b-instruct-sft/` |
| Eval | `results/starting-state/{base-sft,instruct-sft}/` |
| Summary | `results/starting-state/SUMMARY.md` |

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
