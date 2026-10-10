# Split vs distribution (Exp 12 data recipe)

Two different knobs. Mixing them up is how “20k train + 4k eval” looked like leakage,
and how a 94% headline can hide `dir-cycle`.

| Word | Question it answers | Exp 12 setting |
|---|---|---|
| **Split** | *Which rows* are train vs val vs test? | 21k draw → **2k test** + **1k val** + **18k train**. Three-way, disjoint. |
| **Distribution** (mix) | *What fraction* of each type/subtype is in a file? | **Matched:** train, val, and test use the same two-level equal mix. |

SpatialMap v6 is extra: different names, different maps, **unmatched** distribution on purpose (transfer).

Do **not** pick checkpoints on the 2k test. That is the overfit detector’s job for the **val** set; the test is the number you report once.

---

## 1. Split — who is in which file

A **split** is a partition of examples: train / val / test.

Rules we care about:

1. **Disjoint.** An eval item’s user text must not appear in train. If it does, the score can be memorization of that map, not a policy. That *is* leakage / cheating.
2. **Same generator.** Train and the matched 2k are both `generate_all_v6` + `SpatialSolver`, same prompt, same seed-52 pool, then sliced.
3. **Holdout vs independent draw.**
   - **Holdout:** one pool, then cut (Exp 12: 21k → 2k test + 1k val + 18k train). Classic three-way split.
   - **Independent draw:** two generator runs, two seeds (old recipe: seed 42 train 20k, seed 43 extra 4k). Not leakage if the RNGs don’t collide, but the **mix** of the 4k was a second sample of the *old* quotas, not a holdout from the 20k.

The first Exp 11 20k+4k was (2): two draws. We did not call it 16k because we did **not** hold 4k out of the 20k. Exp 12 is (1): test and val are peeled from one pool; the 18k is the leftover.

| File | Role | When you look at it |
|---|---|---|
| 18k train (1.5k ⊂ 6k ⊂ 18k) | SFT | Training only |
| 1k val | **Eval / overfit** | `eval_loss` on a 200-row NLL slice during SFT; **letter-set ckpt pick** on the full 1k |
| 2k test | **Test** | Final headline, once. Never for stopping or picking. |
| SpatialMap v6 | Transfer | After the 2k, not for ckpt pick |

**Nested prefixes** (1.5k ⊂ 6k ⊂ 18k) are a split *inside train* so a size curve uses the same items, not three independent mixes.

---

## 2. Distribution — how much of each bucket

A **distribution** (mix, stratum weights) is P(bucket). Ours is two-level:

1. **Global:** P(dir) = P(which) = P(count) = 1/3.
2. **Local:** equal weight on the subtypes of that type (6 dir, 5 which, 2 count).

That is **hierarchical uniform**, not “13-way equal.” Count still has more *rows per subtype* because it only has two subtypes (each is 1/6 of the file).

**Matched distribution:** the 2k holdout uses the same P(bucket) as the 18k (up to 1–2 leftover which rows). So cycle/incomplete/omit are a real share of the headline, not ~3% noise.

**Unmatched distribution:** eval mix ≠ train mix.

| Unmatched eval | What the number means | Cheat? |
|---|---|---|
| Old seed-43 synth (easy-heavy `count-1`, few cycle) | Transfer to the *previous* mix. Headline dominated by easy buckets. | No, if labeled as old mix. Yes if you quote it as “Exp 12 accuracy.” |
| SpatialMap v6 | Transfer to TQA maps / shop names. | No. That is the point. |
| Train items in the 2k | Leakage. | Yes. |

Reweighting **train** toward hard buckets and then scoring a **matched** hard-ish 2k is not cheating. It is the estimand “accuracy on the mix we trained for.” Reweighting **eval only** (drop cycle from the 2k so the % jumps) would be cheating.

---

## 3. What this is backed by

None of this is HBO and none of it is a spatial-reasoning paper.

| Practice | What it is | Lineage |
|---|---|---|
| Train/test split, disjoint | Don’t score the training rows | Standard supervised learning (e.g. Hastie, Tibshirani, Friedman, *Elements of Statistical Learning*, Ch. 7). |
| Holdout from one pool | One sample, then cut | Same; vs two independent generator seeds. |
| **Stratified** split | Preserve class fractions in train *and* test | Sampling theory (stratified random sample). In ML: keep rare classes in the test set so the metric is not majority-class. sklearn `train_test_split(..., stratify=y)`. |
| Matched P(bucket) | Test estimates risk under the train mixture | i.i.d. / same-distribution assumption. Unmatched test estimates a **different** risk (domain shift). |
| Mixture **weights** (how many of each source) | T5-style data mix; temperature mixing p ∝ n^{1/T} in multilingual SFT | Raffel et al. T5; mT5 / later instruction-mix papers. About *counts*, not file order. |
| Two-level mix | First pick type, then subtype | Hierarchical sampling. HBO (Wang et al., 2025) *learns* those two levels on many tasks; we **set** them (equal). |

**Not** claimed as research-backed here:

- Round-robin JSONL order — **not used** in Exp 12. Files are shuffled; Axolotl `shuffle_merged_datasets: true`. Mixture **counts** are the lever, not cyclic order.
- Error-tilt of Type 0 (more cycle because last 4B failed there) — we **did not** use that for Exp 12; we used equal local weights because this is a new SFT, not a continuation of the old adapter.

---

## 4. Exp 12 files (the instance)

```
21k pool  (seed 52, hierarchical uniform, v12 system prompt)
 ├─ 2k TEST      data/spatial_sft_v12_2000_test.jsonl    ← spatial_eval_v12_synth (report once)
 ├─ 1k VAL       data/spatial_sft_v12_1000_val.jsonl     ← spatial_eval_v12_val (ckpt / overfit)
 │    └─ 200 NLL data/spatial_sft_v12_val_nll.jsonl      ← Axolotl eval_loss during SFT
 └─ 18k leftover
      ├─ 1.5k prefix   spatial_sft_v12_1500_train.jsonl
      ├─ 6k prefix     spatial_sft_v12_6000_train.jsonl
      └─ 18k all       spatial_sft_v12_18000_train.jsonl

SpatialMap v6   = data/spatialeval_v6_corr.jsonl         ← unmatched transfer
```

Slicer: `experiments/12-v6-mix-reweight/scripts/make_v12_data.py`.  
It prints overlap train vs test, train vs val, val vs test — all must be **0**.

Headline: strict letter-set on the **2k test**, plus SpatialMap. Pick the adapter on the **1k val**. Report **13 buckets** on the test. Do not steer on the old seed-43 file.

---

## 5. One-sentence checks

- Same item in train and test, or train and val, or val and test → **split is broken** (leakage).
- Pick ckpt on the 2k test → **test leakage** (mild, but you no longer have an untouched number).
- Same mix, disjoint train / val / test → **matched three-way split** (Exp 12).
- Different mix, disjoint items → **shift eval** (SpatialMap).
- Different mix, quote as the Exp 12 score → **wrong estimand**, not a better model.
