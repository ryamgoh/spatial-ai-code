# Experiment 9 — Mixed-cardinality SFT (Single was too easy)

**Scrapped 2026-09-07.** Never run. Mixed 1/2/4-ans SFT on Corr was
superseded by Exp 10 (option E / TQA-Corr-Full), which already includes
dir-2 and which-4 plus gold-E. Launchers live in this folder.

**Source.** Follow-up to Exp 8 / 8b. Same QLoRA recipe as Exp 8 4B
(`axolotl train --launcher python`, `eval_new.py`).

**Question.** Does SFT that includes 2- and 4-letter gold raise multi
strict above 0% on TQA-Corr without wiping Single (~98%)?

**Why.** Exp 8 4B saturates Single (95–98%). Exp 8b: that same 4B-5k
adapter on full Corr is **76.9% / 98.5% / 0.0%** (all / single / multi).
Zero multi is a one-letter **policy** (0% over-select, `strict == loose`),
not proof that multi is spatially hard. Exp 8 never showed a 2- or
4-letter `Answer:` in training.

**Not GRPO.** This is SFT on mixed gold. GRPO on 4B is still on the table
if mixed SFT lifts multi but leaves a hole; 2B GRPO is the capacity
chapter (Exp 8 59% Single).

## Design

| | |
|---|---|
| IV | Train mix: Corr type×cardinality (1-ans **and** dir-2 / which-4) vs Exp 8b single-only 5k |
| DV | Strict/loose on Corr 1,329; Single vs Multi slices; type × cardinality; over-select / all_four |
| Constants | `Qwen/Qwen3.5-4B` Instruct; 5,000 traces; 2 epochs; no early stop; SFT system prompt; two-pass eval |

**Hypothesis.** Mixed 5k SFT gets multi ≫ 0% (especially which-4) and
keeps Single ≳ 90%. Dir 2-ans may stay harder (Exp 6 floor 0% even
zero-shot). `all_four` should stay near 0% — exact set, not letter dump.

Control = Exp 8b `4b-5k-corr` (same 4B, single-only 5k, same eval).

One training arm (keep the grid small): **scratch** Instruct QLoRA on the
mixed 5k, not continue-from-Exp-8. Continuation is a later add-on if
scratch kills Single.

## Training data (5,000, mix of TQA-Corr)

TQA-Corr 1,329: count 404 / dir-1 332 / dir-2 93 / which-1 302 / which-4 198.

| type | gold letters | n in 5k |
|---|---|---|
| type2 count | 1 | 1,520 |
| type0 dir | 1 | 1,249 |
| type0 dir | 2 | 350 |
| type1 which | 1 | 1,136 |
| type1 which | 4 | 745 |
| type0 dir 4-ans / type1 2-ans | — | **0** (do not occur in Corr) |
| **total** | | **5,000** |

`generate_all.py --test-split 0 --seed 42` →
`data/spatial_sft_corr_mix_5000_train.jsonl`.

Which-4 needs `--num-type1-4-answer` (generator already allowed 4 correct
options; the CLI flag is new).

## Recipe

Copied from Exp 8 `train-sft-4b-5000.yaml` (QLoRA r=64, lr 1e-4, 2 epochs,
seq 4096, microbatch 1, grad acc 8, no early stop). Only `output_dir` and
dataset path change.

## Eval

Task `spatial_eval_gen_cleaned_1329`. Same SFT system prompt as Exp 8/8b.
Summarize like 8b: all / single / multi + type × cardinality + flags.

## Run

Train on H100-96 (`gpu-long`); eval on H200 (`gpu`, 3h) or 96 fallback.

```bash
sbatch run_batch_09_multi_sft_h100_96.sh    # generate 5k + SFT
sbatch run_batch_09_eval_corr_h200.sh       # after adapter exists
```

```bash
cd eval && uv run --no-project python ../experiments/09-multi-sft/scripts/summarize.py
```

## Artifacts

| artifact | location |
|---|---|
| 5k Corr-mix jsonl | `data/spatial_sft_corr_mix_5000_train.jsonl` |
| Adapter | `models/qwen3.5-4b-sft-mix5k/` |
| Eval | `results/corr/4b-mix5k/` |
| Summary | `results/corr/SUMMARY.md` |

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
