# Experiment 7b — Untuned 4B Base vs Instruct on TQA-Corr-Single (SFT prompt)

**Source.** Pairing baseline for Exp 7/8. No training. Same two
checkpoints as Exp 7 (`Qwen3.5-4B-Base` and `Qwen3.5-4B`), but **zero-shot**
on Single with the **SFT system prompt** (not Non-shot-2).

**Question.** What do un-tuned Base and Instruct score on
SpatialMap-TQA-Corr-Single when the eval prompt matches Exp 7/8 SFT?

**Why this exists.** Exp 6 already has zero-shot Single **sliced from
Corr 1,329** under **Non-shot-2**: Instruct 46.5% / Base 20.4%. Exp 7/8
SFT evals use the `generate_all.py` system prompt. Exp 7b is the paired
untuned Δ for those SFT numbers.

| | Exp 6 | Exp 7b | Exp 8 4B-5k |
|---|---|---|---|
| Checkpoint | Base + Instruct | Base + Instruct | Instruct + 5k SFT |
| Prompt | Non-shot-2 | SFT system prompt | SFT system prompt |
| Eval | 1329, Single **slice** | Single **task** (1,038) | Single 1,038 |
| Strict Single | 20.4% / 46.5% | **this run** | 98.5% |

## Run (H200, `gpu`, 3h)

Two 4B × 1,038 evals will not both finish in 3h on one card. Submit
**two jobs** (you have four H200s):

```bash
sbatch --export=ALL,TAGS=instruct run_batch_07b_eval_single_h200.sh
sbatch --export=ALL,TAGS=base     run_batch_07b_eval_single_h200.sh
```

Writes `results/zero-shot-single/{instruct,base}/`. After both finish:

```bash
cd eval && uv run --no-project python ../experiments/07b-zero-shot-single/scripts/summarize.py
```

## Corr 1,329 (Single + Multi slices)

Same SFT prompt, no LoRA. One 1,329 run per checkpoint; summarize slices
multi (291) the way Exp 6 did.

```bash
sbatch --export=ALL,TAGS=instruct-corr run_batch_07b_eval_single_h200.sh
sbatch --export=ALL,TAGS=base-corr     run_batch_07b_eval_single_h200.sh
```

## Stages 1 vs 2 (does the constrained re-ask matter?)

Default eval is `--stages 2`. GRPO scores a single `Answer:` in the
completion. Re-run Single (or Corr) with `--stages 1` into `*-s1/` dirs
(does not overwrite the tables above):

```bash
sbatch --export=ALL,TAGS=instruct,STAGES=1 run_batch_07b_eval_single_h200.sh
sbatch --export=ALL,TAGS=base,STAGES=1     run_batch_07b_eval_single_h200.sh
```

The SFT check that actually matters for GRPO is 4B-5k, not 7b zs:

```bash
sbatch run_batch_08_eval_4b5k_s1_h200.sh
```

```bash
cd eval && uv run --no-project python ../experiments/07b-zero-shot-single/scripts/summarize_corr.py
```

## Artifacts

| artifact | location |
|---|---|
| Single eval | `results/zero-shot-single/{instruct,base}/` |
| Corr eval | `results/zero-shot-corr/{instruct,base}/` |
| Summaries | `results/zero-shot-single/SUMMARY.md`, `results/zero-shot-corr/SUMMARY.md` |

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
