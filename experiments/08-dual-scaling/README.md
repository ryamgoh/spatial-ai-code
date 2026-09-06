# Experiment 8 — Dual Scaling Laws (data volume × parameter scale)

**Source.** Post-thesis Phase 2. Same train/eval stack as Exp 7
(`axolotl train --launcher python`, `eval_new.py`). Locked to the
Exp 6/7 conclusions: **Instruct, not Base**; **TQA-Corr-Single, not
full Corr**.

**Question.** How do SFT dataset size and Qwen3.5 parameter count
jointly affect SpatialMap-TQA-Corr-Single accuracy?

Two sub-experiments share the 4B × 20k cell.

**Results.** 2.1 rejected (4B saturates at 1.5k–5k; 20k is not best).
2.2: 4B ≫ 0.8B > 2B at 20k; not monotone in size. Full write-up:
[`RESULTS.md`](RESULTS.md).

## What Exp 6/7 change here

| finding | consequence |
|---|---|
| Instruct beats Base on strict (83.5% vs 64.7% after 5k SFT); Base still 100% over-selects | Every arm is `Qwen/Qwen3.5-{0.8,2,4}B` (post-trained), never `*-Base` |
| Multi-answer is type-confounded and rewards letter-dumping | Train traces are **single-letter gold only**; eval is TQA-Corr-Single (1,038) |
| Exp 7 early-stopped at ~100–130 steps (~1k examples seen) | **No early stop.** 2 full epochs so 20k is actually 20k, not 5k with a longer file |

## Independent / dependent variables

### 2.1 Data scaling (fixed 4B Instruct)

| | |
|---|---|
| IV | Train n ∈ {1,500, 5,000, 20,000} (nested, same type mix) |
| DV | Strict/loose acc on TQA-Corr-Single; error = 1 − strict; over-select rate |
| Constants | `Qwen/Qwen3.5-4B`; QLoRA recipe below; Single-only traces; SFT system prompt |

**Hypothesis.** Strict error falls log-linearly with n; 20k is the lowest error.

### 2.2 Parameter scaling (fixed 20k)

| | |
|---|---|
| IV | Instruct checkpoint ∈ {0.8B, 2B, 4B} |
| DV | Same as 2.1 |
| Constants | 20,000 nested traces; same QLoRA recipe (r=64 is *not* scaled with width) |

**Hypothesis.** Parameter count still separates the three. Extra data may
close the gap on easy items; 0.8B hits a capacity ceiling on the
axis-closure / option-mapping the traces teach.

## Grid (5 SFT runs, 4B-20k shared)

| tag | model | n | sub-exp |
|---|---|---|---|
| `4b-1.5k` | `Qwen/Qwen3.5-4B` | 1,500 | 2.1 |
| `4b-5k` | `Qwen/Qwen3.5-4B` | 5,000 | 2.1 |
| `4b-20k` | `Qwen/Qwen3.5-4B` | 20,000 | 2.1 ∩ 2.2 |
| `2b-20k` | `Qwen/Qwen3.5-2B` | 20,000 | 2.2 |
| `0.8b-20k` | `Qwen/Qwen3.5-0.8B` | 20,000 | 2.2 |

Not a replication of Exp 7’s 4B-instruct-5k (83.5%): that run early-stopped
and used a separately generated 5k file. This 5k is a **stratified nested
subset** of the 20k pool, trained for 2 epochs.

## Training data (single-letter, type mix of TQA-Corr-Single)

TQA-Corr-Single mix is count 404 / dir 332 / which 302 of 1,038.
Generate 20k once (`generate_all.py`, seed 42, `--test-split 0`), then
`scripts/make_scale_data.py` writes nested 5k and 1.5k (1.5k ⊂ 5k ⊂ 20k).

| n | type0 dir | type1 which | type2 count |
|---|---|---|---|
| 1,500 | 480 | 436 | 584 |
| 5,000 | 1,599 | 1,455 | 1,946 |
| 20,000 | 6,397 | 5,819 | 7,784 |

Axolotl then holds out 5% of whichever file for `eval_loss` (same as 4b/7).

## Recipe (QLoRA, copied from Exp 7 except early stop)

r=64, α=128, lr 1e-4, 2 epochs, seq 4096, microbatch 1, grad acc 8,
4-bit, same LoRA targets including Qwen3.5 `linear_attn.*`.
`load_best_model_at_end` still picks the lowest `eval_loss` checkpoint;
there is **no** `early_stopping_patience`, so both epochs run.

`eval_steps` / `save_steps` scale with n (≈4 evals/epoch) so 20k does
not spend half the job on val.

## Eval

Same as Exp 7 instruct: two-pass vLLM, SFT system prompt from
`generate_all.py` (not Non-shot-2), task `spatial_eval_gen_cleaned_single`.
Stage 2 still does **not** load the LoRA (existing harness). That was
harmless for 4B-Instruct in Exp 7 (0% over-select); 0.8B/2B get the
same protocol so the IV is the checkpoint, not a new decoder.

## Run (1× H100-96 + 1× H100-47)

Same QLoRA recipe on both cards (microbatch 1, grad acc 8). 96GB is
only more VRAM / less paging, not a different training condition.

```bash
sbatch run_batch_08_scaling_h100_96.sh     # 4B × {1.5k, 5k, 20k}
sbatch run_batch_08_scaling_h100_47.sh     # 0.8B-20k (+ 2B-20k if not split off)
sbatch run_batch_08_scaling_h100_96_b.sh   # 2B-20k on a second 96, other folders
```

| job | GRES | cells | adapter dir |
|---|---|---|---|
| `spatial8-96` | `h100-96:1` | `4b-1.5k`, `4b-5k`, `4b-20k` | `models/qwen3.5-4b-sft-*` |
| `spatial8-47` | `h100-47:1` | `0.8b-20k`, `2b-20k` | `…-0.8b-sft-20000`, `…-2b-sft-20000` |
| `spatial8-96b` | `h100-96:1` | `2b-20k-96b` | `models/qwen3.5-2b-sft-20000-96b` |

`96b` does **not** share a folder with 47. If 47 later starts 2B, `scancel` it;
junk in `qwen3.5-2b-sft-20000/` is unused. Summarize prefers `2b-20k-96b`.

Submit both at once. Data gen is `flock`'d: whichever job starts first
writes the 20k pool + nested 1.5k/5k; the other waits and reuses.
Evals write to `results/scaling/<tag>/` (no shared timestamp dir).

If `h100-96` is only on partition `gpu`, resubmit the 96 job with
`-p gpu` (keep the 2-day limit if the partition allows it).

Sequential fallback (all five cells on one 47GB slice):

```bash
sbatch run_batch_08_scaling_h100.sh
```

Idempotent: existing 20k jsonl / nested slices / adapter / `results.json`
are skipped. Overrides:

- `SKIP_TRAIN=1` / `SKIP_EVAL=1`
- `SKIP_2_1=1` — skip `4b-1.5k` and `4b-5k` (still trains `4b-20k` if selected)
- `SKIP_2_2=1` — skip `2b-20k` and `0.8b-20k`
- `ONLY=4b-5k,2b-20k` — run that subset (comma-separated tags)

0.8B train finished, eval not run (prefer H200, 3h `gpu` cap):

```bash
sbatch run_batch_08_eval_0.8b_h200.sh
```

One `h200-141`, not `:4`. Writes `results/scaling/0.8b-20k/`. Fallback MIG:

```bash
sbatch run_batch_08_eval_0.8b_h100_47.sh
```

Do **not** wait for a job to write `SUMMARY.md`. After **both** GPU jobs
have `results.json` under `results/scaling/<tag>/`:

```bash
cd eval && uv run --no-project python ../experiments/08-dual-scaling/scripts/summarize.py
```

Wall-clock ≈ max(4B 1.5k+5k+20k on 96, 0.8B-20k+2B-20k on 47). 4B-20k
is still the long pole. Budget **2 days** `gpu-long` on each.

## Artifacts

| artifact | location |
|---|---|
| 20k pool | `data/spatial_sft_single_scale_20000_train.jsonl` |
| nested 5k / 1.5k | `data/spatial_sft_single_scale_{5000,1500}_train.jsonl` |
| adapters | `models/qwen3.5-{4b-sft-1500,4b-sft-5000,4b-sft-20000,2b-sft-20000,0.8b-sft-20000}/` |
| Eval | `results/scaling/<tag>/` |
| Summary | `results/scaling/SUMMARY.md` |

## Run log

| date | results dir | notes |
|---|---|---|
| 2026-09-05 | `results/scaling/` | All five cells. 2B is `2b-20k-96b` (4,750 steps). 0.8B eval on H200. See `RESULTS.md`. |
