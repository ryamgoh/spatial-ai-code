# Experiment 8b — 4B-5k Single-SFT transferred to TQA-Corr (1,329)

**Source.** Follow-up to Exp 8. No new training. The Exp 8 `4b-5k`
adapter (Qwen3.5-4B Instruct, 5,000 single-answer traces, 2 epochs)
is evaluated on **full SpatialMap-TQA-Corr**, not Single.

**Question.** Does Single-only SFT transfer to multi-answer gold
(`Answer: A, B` / four-letter `which`), or does strict acc collapse
off-slice?

**Why this exists.** Exp 8 4B is at ~95–98% on TQA-Corr-Single. GRPO
on that checkpoint on Single is a null. The remaining hole, if any, is
the 291 multi rows SFT never saw. This eval is the gate for a 4B GRPO
chapter.

**Results.** Single 98.5% (paired with Exp 8). Multi **0.0%** (291/291).
Corr 76.9% is only the Single mass. Write-up: [`RESULTS.md`](RESULTS.md).

**Not SFT on Corr.** The adapter is unchanged. This is transfer.

| | |
|---|---|
| IV | Eval set: Corr 1,329 (vs Exp 8 Single 1,038) |
| DV | Strict/loose all, Single slice, Multi slice; type × cardinality; over-select |
| Constants | `models/qwen3.5-4b-sft-5000` from Exp 8; SFT system prompt; two-pass eval |

Compare to Exp 6 Instruct + Non-shot-2 on the same 1,329 (strict all 41.2%,
Single 46.5%, Multi 22.3%). Prompt is **not** Non-shot-2 here — it is the
SFT prompt, so the paired zero-shot is Exp 7b, not Exp 6.

## Run

H200 (`gpu`, 3h) first; if it times out, the 96GB long job:

```bash
sbatch run_batch_08b_eval_corr_h200.sh
sbatch run_batch_08b_eval_corr_h100_96.sh   # fallback
```

Writes `results/transfer/4b-5k-corr/`. After it finishes:

```bash
cd eval && uv run --no-project python ../experiments/08b-sft-transfer-corr/scripts/summarize.py
```

## Artifacts

| artifact | location |
|---|---|
| Adapter (from Exp 8) | `../08-dual-scaling/models/qwen3.5-4b-sft-5000/` |
| Eval | `results/transfer/4b-5k-corr/` |
| Summary | `results/transfer/SUMMARY.md` |

## Run log

| date | results dir | notes |
|---|---|---|
| 2026-09 | `results/transfer/4b-5k-corr/` | 76.9% / 98.5% / 0.0% all/single/multi. See `RESULTS.md`. |
