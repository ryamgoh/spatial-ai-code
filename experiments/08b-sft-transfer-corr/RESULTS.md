# Exp 8b results — 4B-5k Single-SFT transferred to TQA-Corr

No new training. Adapter is Exp 8 `4b-5k` (Qwen3.5-4B Instruct, 5,000
single-answer traces, 2 epochs). Eval is SpatialMap-TQA-Corr (n = 1,329),
same SFT system prompt and two-pass harness as Exp 8 Single.

**Headline.** Single is unchanged at **98.5%**. Multi is **0.0%** (291/291
wrong). Full Corr is **76.9%**, which is exactly the Single mass:
\(1038 \times 0.985 / 1329 = 76.9\%\). The model never emits a 2- or
4-letter set.

---

## Verdict

**Question.** Does Single-only SFT transfer to multi-answer gold?

**No.** It saturates Single and **zeros** multi. This is the gate for
4B GRPO: the hole is **set cardinality**, not axis-closure on one-letter
items.

| slice | n | strict | loose |
|---|---|---|---|
| all (Corr) | 1,329 | **76.9%** | **76.9%** |
| single | 1,038 | **98.5%** | 98.5% |
| multi | 291 | **0.0%** | 0.0% |

Paired with Exp 8 `4b-5k` on the Single **task** (also 98.5%). Same 1,038
rows, same adapter: in-slice number is stable under the Corr file.

`strict == loose` on **all three slices**. `over_select` is **0%** on
multi. Loose requires gold ⊆ prediction. A 2- or 4-letter gold is never
a subset of a 1-letter prediction, so loose dies with strict. The model
is **under-selecting** (almost certainly always one letter), not dumping
`A,B,C,D`.

### Type × cardinality (strict)

| type | single n / acc | multi n / acc |
|---|---|---|
| count | 404 / **99.0%** | — (no multi count in Corr) |
| dir | 332 / **96.7%** | **93 / 0.0%** |
| which | 302 / **99.7%** | **198 / 0.0%** |

Both multi families are at floor: dir 2-ans and which 4-ans. SFT never
saw `Answer: A, B` or four gold letters.

Format flags are 0% on every slice (no loops, no missing `Answer:`, no
all-four). This is a **policy** of one letter, not generation collapse.

---

## Against Exp 6 zero-shot (same 1,329, different prompt)

Exp 6 Instruct + Non-shot-2 (not paired):

| | Exp 6 Instruct zs | Exp 8b 4B-5k SFT |
|---|---|---|
| Prompt | Non-shot-2 | SFT system prompt |
| strict all | 41.2% | 76.9% |
| strict single | 46.5% | **98.5%** |
| strict multi | **22.3%** | **0.0%** |
| which-4-ans | 32.8% | **0.0%** |
| dir-2-ans | 0.0% | 0.0% |
| over_select all | 33.1% | 0.0% |

Single-SFT **beats** zero-shot on Single by a huge margin and **loses**
on multi vs the same Instruct checkpoint that still over-selected
sometimes. Teaching `Answer: A` to 98.5% wiped the residual multi-letter
behaviour Exp 6 still had on `which` 4-ans (32.8% → 0%).

Exp 6 Base got which-4-ans 73.7% by dumping letters (96.8% over-select).
That is not a skill to restore; GRPO should teach **exact set match**,
not letter-dumping.

---

## What this means for GRPO / FYP

| idea | now |
|---|---|
| GRPO 4B on **Single** | Still a null. 98.5%, 0% format errors, no advantage. |
| GRPO 4B on **mixed cardinality** | **Justified.** 0/291 multi, format already one-letter. Outcome reward on `A,C` / `A,B,C,D` is the signal. Watch Single so it does not fall off 98.5%. |
| GRPO **2B** | Still valid for the 59% Single capacity hole (Exp 8). Different question. |
| “SFT on Corr then done” | Would also teach 2/4-ans, but that is a new SFT, not this adapter. |

Eval metric for a 4B GRPO chapter: strict on 1,329, plus Single vs multi
sliced like this table. A win is multi ≫ 0% without dumping `all_four`.

---

## Copy

```
4b-5k-corr   n=1329/1038/291
strict all / single / multi = 76.9% / 98.5% / 0.0%
loose all = 76.9%
dir multi 93/0.0%   which multi 198/0.0%
count 404/99.0%  dir single 332/96.7%  which single 302/99.7%
anomalies 0% all slices
```
