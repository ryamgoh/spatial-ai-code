# Exp 8 results — Dual scaling on SpatialMap-TQA-Corr-Single

SFT on nested single-answer AxisDecomposition traces. Eval is
SpatialMap-TQA-Corr-Single (n = 1,038), two-pass vLLM, SFT system prompt
(not Non-shot-2). All starts are Instruct (`Qwen/Qwen3.5-{0.8,2,4}B`),
never `*-Base`. Protocol: `README.md`. Raw tables:
`results/scaling/SUMMARY.md`.

**Headline.** 4B Instruct saturates Single after 2 epochs of 1.5k traces
(95.5%). More unique data does not help. 20k traces do not let 0.8B or 2B
match 4B-1.5k. 2B finished the same 4,750-step budget as 4B-20k and 0.8B
and is the **worst** of the three (59.2%). Parameter count is not monotone.

---

## Verdicts

### 2.1 Data scaling (fixed 4B Instruct)

**Hypothesis.** Strict error falls log-linearly with n; 20k is lowest error.

**Rejected.** Error is not log-linear. Lowest error is **5k**, not 20k.

| config | n train | steps | strict = loose | error | misses / 1,038 |
|---|---|---|---|---|---|
| 4b-1.5k | 1,500 | 357 | **95.5%** | 4.5% | ~47 |
| **4b-5k** | 5,000 | 1,188 | **98.5%** | **1.5%** | **~16** |
| 4b-20k | 20,000 | 4,750 | 97.5% | 2.5% | ~26 |

4b-20k − 4b-1.5k: **+2.0 pp**. 5k → 20k: **−1.0 pp**.

Δerror / Δln n is −0.025 then **+0.007**. The second segment has the
wrong sign for a log-linear error law.

Format flags are **0%** on every 4B cell (`over_select` included).
`strict == loose` everywhere: remaining errors are wrong letters, not
cardinality.

1.5k already has the procedure. Extra unique traces past 5k do not buy
another algorithm on this eval.

### 2.2 Parameter scaling (fixed 20k, 2 epochs)

**Hypothesis.** Width still separates the three; 0.8B hits a capacity
ceiling that 20k cannot close vs 4B.

**Half right.** 4B is far above both small models. 0.8B is **not** the
floor — **2B is**.

| config | params | steps | first eval_loss | last eval_loss | strict = loose |
|---|---|---|---|---|---|
| 0.8b-20k | 0.8B | 4,750 | 0.454 | 0.0003 | **76.8%** |
| 2b-20k-96b | 2B | **4,750** | 0.392 | 0.0002 | **59.2%** |
| 4b-20k | 4B | 4,750 | 0.294 | 0.0001 | **97.5%** |

4B-20k − 0.8B-20k: **+20.7 pp**. 4B-1.5k (95.5%) still beats 0.8B-20k
and 2B-20k. Sample efficiency of 4B, not a leaked test set (generator
civic names vs SpatialMap shop names; 0 shared map cores on the local
SFT mix).

2B is a completed job: `global_step` 4,750, train loss 0, same 20k
file as 0.8B/4B-20k. The cancelled 47 `2b-20k` dir has no trainer
state; summarize used `2b-20k-96b`.

Format on small models is almost clean (0.7–0.8% token loops, **0%**
over-select). This is not Exp 7 Base-SFT (100% over-select).

---

## Against Exp 7 (same 4B Instruct, 5k-scale traces)

| | Exp 7 instruct-sft | Exp 8 4b-5k |
|---|---|---|
| n train | 5,000 (own seed-42 file) | 5,000 nested from the 20k pool |
| YAML epochs | 2 | 2 |
| Early stop | yes (patience 1) | **no** |
| Steps | **130** | **1,188** |
| last eval_loss | 0.0092 | 0.0004 |
| Single strict | **83.5%** | **98.5%** |
| over_select | 0% | 0% |

Same recipe except early stop and a different 5k draw. The +15 pp is
~9× more optimizer steps, not a new method. Exp 7 `eval_loss` already
looked converged (~0.009) at step 130; that NLL is a bad proxy for
Single acc. Exp 8 2.2 shows the same disconnect: all three 20k models
end at eval_loss ~1e-4, acc is 59 / 77 / 98.

Exp 7 Base-SFT (64.7% strict / 92.8% loose, 100% over-select, 100 steps)
is why Exp 8 never starts from `*-Base`.

---

## Loss curves (what `eval_loss` actually is)

Axolotl holds out **5% of the synthetic jsonl** for `eval_loss`. That is
not TQA. `--test-split 0` on `generate_all.py`; TQA-Corr-Single is
final eval only.

`load_best_model_at_end` + no early stop: on every finished cell, **best
eval_loss = last step**. The reported adapter is the fully trained one.

| config | first eval_loss | last train / eval_loss | best (step) |
|---|---|---|---|
| 4b-1.5k | 0.294 | 0.0019 / 0.0017 | 0.0017 (357) |
| 4b-5k | 0.285 | 0.0001 / 0.0004 | 0.0004 (1188) |
| 4b-20k | 0.294 | 0.0000 / 0.0001 | 0.0001 (4750) |
| 2b-20k-96b | 0.392 | 0.0000 / 0.0002 | 0.0002 (4750) |
| 0.8b-20k | 0.454 | 0.0000 / 0.0003 | 0.0003 (4750) |

They all clone traces. 4B **transfers** the procedure onto SpatialMap
names. 0.8B less so. 2B least, despite slightly better NLL than 0.8B.

---

## What 4B-1.5k beating 20k-small does *not* mean

- **Not item leak.** Train names are 20 civic landmarks (`Library`,
  `Hospital`, …). Eval is SpatialMap shop names (`Narwhal's Novelties`,
  …). Exact name match 0/20. Local SFT jsonl map-core overlap with eval:
  0. Exp 8’s 20k file is the same generator.
- **Not a 4B scoring bug.** Same Single task, same two-pass harness.
  A broken metric would not give 2B 59% and 4B 98% on the same 1,038
  rows.
- **Not incomplete 2B training.** 4,750 steps = 0.95 × 20,000 / 8 × 2.

It **does** mean: once a 4B Instruct model can run AxisGraph, 1.5k
copies of the template are enough. More unique 0.8B/2B traces do not
add a new circuit.

---

## Caveats (do not bury these)

**1. Stage 2 is a constrained letter readout with LoRA off.**
`eval_new.py` generates thinking **with** LoRA, then fills `Answer: `
with `lora_request=None`, regex `[A-D](,[A-D])*`, T=0, max 16 tokens.
The metric uses the last `Answer:` line, which is that readout.

On these runs `strict == loose` and `over_select` is 0%, so stage 2
almost always emitted **one** letter. If the CoT already commits
(`Answer: C`), greedy copy ≈ “parse the trace” and the size ranking is
mostly CoT quality. If the CoT never maps to a letter, stage 2 is the
**untuned** Instruct model of that size. 4B Instruct is a better mapper
than 2B/0.8B (Exp 6 Single 46.5% with no SFT).

A `--stages 1` rescore (letter from the LoRA CoT only) is the check
that 2.2 is honest. Not done at write-up.

**2. LoRA r=64 is not a constant treatment across width.** Same rank is
a much larger fraction of 0.8B than of 2B or 4B. That can produce
**0.8B > 2B** without 0.8B being a better spatial reasoner. 2.2 is
therefore **not** a clean parameter sweep.

**3. 0.8B eval used `batch_size: 128` on H200; 4B/2B used 4.** Unlikely
to invert 77 vs 59 vs 98; it is still not the same eval.

**4. Stage-1 sampling is T=0.6.** One sample per item. Cannot turn 59%
into 95%.

**5. Nested 1.5k ⊂ 5k ⊂ 20k, type mix matched to Single** (dir / which /
count). Axolotl’s 5% val split is a **new random holdout per file**,
not nested.

**6. Eval is Single only.** No 2-answer `dir`, no 4-answer `which`.
Ceiling on Single is not a ceiling on TQA-Corr 1,329. That is Exp 8b.

---

## What to write / not write

**Write**

- Instruct 4B + 2 full epochs of oracle Single traces saturates
  TQA-Corr-Single from 1.5k (95.5%); 5k is 98.5%; 20k does not win.
- Exp 7’s 83.5% was undertrained 5k (130 steps), not a different method.
- Trace `eval_loss` is a bad proxy for SpatialMap acc.
- 20k does not let 0.8B or 2B match 4B-1.5k.
- 2B-20k is a finished run at 59.2%, format-clean.

**Do not write**

- “Accuracy scales log-linearly with SFT n.”
- “Accuracy increases with parameter count at 20k.”
- “0.8B is the capacity ceiling.”
- GRPO-on-4B-Single as the next main result (no reward variance).

---

## Next (already scaffolded)

| exp | question |
|---|---|
| **8b** | 4B-5k Single-SFT → **Corr 1,329** (transfer; gate for 4B GRPO on multi) |
| **7b** | Untuned 4B Base + Instruct on Single, **SFT prompt** (paired zero-shot vs 95–98%) |
| GRPO | **2B-20k** (59%, dense outcome errors). Not 4B-Single. 4B GRPO only if 8b multi is weak |

---

## Numbers (copy)

Eval n = 1,038 unless noted. Strict = loose on every Exp 8 cell.

```
4B-1.5k    95.5%   steps 357
4B-5k      98.5%   steps 1188
4B-20k     97.5%   steps 4750
0.8B-20k   76.8%   steps 4750
2B-20k     59.2%   steps 4750   (adapter qwen3.5-2b-sft-20000-96b)

Exp 7 instruct-sft  83.5%  steps 130   (early stop)
Exp 7 base-sft      64.7% strict / 92.8% loose   (100% over-select)
Exp 6 Instruct zs   46.5% Single (Non-shot-2, not this prompt)
```
