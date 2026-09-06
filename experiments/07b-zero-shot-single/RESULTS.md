# Exp 7b results — untuned 4B, SFT prompt, stages 1 vs 2

No LoRA. Same `generate_all.py` system prompt as Exp 7/8. Not Non-shot-2.

**Headline.** The SFT prompt already gets untuned 4B into the high 70s on
Single **if stage 2 is on**. Stage 1 (letter from the CoT only) **collapses
Instruct to ~4%**. Two-pass eval is carrying zero-shot. That is **not**
evidence that SFT+GRPO still need stage 2.

---

## Single task (n = 1,038), stages 2

| config | strict | loose | over_select |
|---|---|---|---|
| base | 76.9% | 96.0% | 37.0% |
| instruct | 78.2% | 92.6% | 15.7% |

Vs Exp 6 Non-shot-2 Single (same weights): Base 20.4% → 76.9%, Instruct
46.5% → 78.2%. Prompt, not SFT.

Vs Exp 8 4B-5k SFT (same prompt, 2 epochs): 78.2% → **98.5%**. That +20 pp
is SFT. Exp 7 Instruct-SFT at 130 steps (83.5%) barely beat this zs.

---

## Corr 1,329 — type × cardinality, stages 2 vs 1

`instruct` / `base` = `--stages 2` (constrained A–D re-ask, T=0).
`*-s1` = `--stages 1` (regex on the thinking string). Same SFT prompt.

| config | count 1 (404) | dir 1 (332) | which 1 (302) | dir 2 (93) | which 4 (198) |
|---|---|---|---|---|---|
| base | 54.0% | 97.3% | 89.4% | 3.2% | **62.6%** |
| base-s1 | 29.0% | 59.3% | 36.1% | 0.0% | 0.0% |
| instruct | 57.9% | 96.7% | 92.7% | 0.0% | 18.2% |
| instruct-s1 | **0.7%** | **6.0%** | **5.0%** | 0.0% | 0.0% |

Implied Single (weighted from the three 1-ans types):

| | stages 2 | stages 1 |
|---|---|---|
| base | ~78% | ~41% |
| instruct | ~80% | **~4%** |

Implied multi (dir-2 + which-4):

| | stages 2 | stages 1 |
|---|---|---|
| base | ~44% | 0% |
| instruct | ~12% | 0% |

---

## What this means

**Stage 2 is a zs crutch, especially for Instruct.** Untuned Instruct
does not write a parseable `Answer: A` in the CoT (`s1` ~4%). The
constrained re-ask then reads the CoT and emits a letter. Dropping
stage 2 on **zero-shot Instruct** would fake a broken model.

**Base s1 > Instruct s1** (41% vs 4%): Base dumps letters into the
thinking more often, so the regex can catch them. Instruct is chatty
without the `Answer:` line. Same pattern as Exp 6/7 over-select: Base
covers gold (loose 96%) and over-selects (37%); Instruct is tighter.

**which-4 at stages 2:** Base 62.6% vs Instruct 18.2%. Base’s letter
dumping hits `A,B,C,D` more often. Instruct’s one-letter prior hurts
which-4 even zero-shot. Dir-2 stays ~0 for everyone (exact 2-subset).

**Do not use 7b s1 to decide GRPO eval.** GRPO and SFT traces already
end with `Answer: A`. Exp 8 4B-5k s2 has 0% `format_fail`. The check
that matters is `sbatch run_batch_08_eval_4b5k_s1_h200.sh` (SFT
`--stages 1`). If that stays ~98%, GRPO eval can be one pass. If 7b
Instruct s1 being 4% talks you into keeping two-pass for SFT, you are
optimizing for a model that no longer exists.

---

## Copy

```
7b Single s2   base 76.9% / instruct 78.2%   (SFT prompt, no LoRA)
7b Instruct Corr s1 ~4% Single; s2 ~80% Single
7b which-4 s2  base 62.6% / instruct 18.2%
7b dir-2 ~0% all configs
```
