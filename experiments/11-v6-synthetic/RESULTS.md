# Exp 11 results — 4B-1.5k error analysis (v6 synth 4k)

Cell: Qwen3.5-4B Instruct, QLoRA SFT on `spatial_sft_v6_scale_1500_train.jsonl`
(2 epochs, r=64). Eval: `spatial_eval_v6_synth` (generate_all_v6, seed 43,
n = 4,000) and `spatial_eval_v6_spatialmap` (n = 1,500). Protocol:
`eval_new.py` default `--stages 2`, extract-answer regex on the last
`Answer:` line. Source traces: repo-root
`responses_spatial_eval_v6_synth.jsonl` (scp of the 4k synth run).

**Headline.** 4B-1.5k is **93.9% strict / 94.1% loose** on the 4k synth
test and **96.1% strict** on SpatialMap v6. That mean is easy mass.
**245 synth misses; 188 (77%) are three Type-0 policies:** cycle,
incomplete-pair, and dir-2. Omit, undetermined, which-k, and count are
already ≥96%. Stage 2 is not carrying the number. This is not SpatialMap
item leak.

---

## Headline numbers

| task | n | strict | loose | misses |
|---|---:|---:|---:|---:|
| `spatial_eval_v6_synth` | 4,000 | **93.875%** | 94.10% | 245 |
| `spatial_eval_v6_spatialmap` | 1,500 | **96.13%** | 96.40% | ~58 |

`strict ≈ loose` (9 extra loose hits). Format is clean: **0** empty
parses, **0** missing `### Final Deduction`. 3,998/4,000 completions
have two `Answer:` lines (CoT + stage-2 readout). Stage-1 vs stage-2
disagree on **9** items; **8** of those stage-1 was already gold and
stage-2 overwrote it. Metric ≈ the LoRA CoT, not a constrained-decode
crutch.

---

## Acc by gold kind (synth 4k)

Kinds from gold option text + gold CoT (same census as `README.md`
test row). Train 1.5k counts in the last column.

| kind | n eval | n train 1.5k | strict | misses |
|---|---:|---:|---:|---:|
| dir-omit | 134 | 50 | **100.0%** | 0 |
| which-0 | 267 | 100 | **99.3%** | 2 |
| count-omit | 267 | 100 | **99.3%** | 2 |
| which-1 | 400 | 150 | 98.8% | 5 |
| which-4 | 133 | 50 | 98.5% | 2 |
| count-1 | 1,067 | 400 | 98.5% | 16 |
| which-3 | 200 | 75 | 98.0% | 4 |
| dir-undetermined | 200 | 75 | 98.0% | 4 |
| dir-1 | 400 | 150 | 97.8% | 9 |
| which-2 | 333 | 125 | 96.1% | 13 |
| **dir-2** | 333 | 125 | **86.5%** | **45** |
| **dir-incomplete** | 133 | 50 | **55.6%** | **59** |
| **dir-cycle** | 133 | 50 | **36.8%** | **84** |

Easy types (everything except dir-2 / incomplete / cycle): 3,401 / 3,534
= **96.2%**. The three hard types: 411 / 599 = **68.6%**. Averaging them
with 1,067 count-1 items is why 1.5k “looks saturated.”

---

## Where the 245 misses go

| mode | n | share of misses |
|---|---:|---:|
| cycle → picked a direction | 84 | 34% |
| incomplete → picked the listed compound | 59 | 24% |
| dir-2 wrong set (often undetermined / none) | 28 | 11% |
| dir-2 underselect (one of two letters) | 17 | 7% |
| which-2 underselect | 11 | 4% |
| count wrong number | 8 | 3% |
| count picked special | 8 | 3% |
| dir-1 wrong letter | 6 | 2% |
| undetermined → picked a direction | 4 | 2% |
| other which / dir-1 / count-omit | 20 | 8% |

**dir-cycle + dir-incomplete + dir-2 = 188 / 245 = 77%.**

Gold-cardinality on misses is mostly 1-letter (181) because cycle and
incomplete gold is a single special. Residual multi errors are dir-2
and a handful of which-k underselects. This is **not** Exp 8b’s “never
emit two letters” hole: which-2/3/4 are 96–98%.

---

## Failure 1 — cycle (36.8%)

Gold: injected reversing sentence, both `<` edges on an axis, derived
line is `none (both axes unknown / contradiction)`, letter is Cannot be
determined.

The model treats one of the two conflicting edges as true and emits a
compound (often two letters, as if dir-2). Every cycle item queries a
**stated** pair (`stated=True` on 133/133). Only **58 / 133** CoTs
mention “contradiction”; 17 of those 58 still pick a direction.

Example (`doc_id` 138):

- Q: Supermarket relative to Police Station. Gold **A** Cannot be determined.
- Gold derived: `none (both axes unknown / contradiction)`.
- Pred derived: `Northwest (both axes proven)` → **E**.

The 1.5k mix has **50** cycle traces. The eval system prompt never says
what to do when both `A < B` and `B < A` exist.

---

## Failure 2 — incomplete-pair (55.6%)

Not a graph error. Gold and model often derive the **same** axis facts.
The miss is the menu rule.

Example (`doc_id` 0):

- Q: Bakery relative to Supermarket. Options include Northeast and
  Cannot be determined; Southeast is **not** listed.
- Gold and pred both: `East proven, North/South unknown → must be
  Northeast or Southeast.`
- Gold: only one of the two live compounds is listed → **C** Cannot be
  determined (picking Northeast would invent East).
- Pred: marks Northeast **In** → **B**.

When the CoT actually writes “only one of the two remaining compounds
is listed” (74 items), accuracy is **100%**. The 59 misses never fire
that sentence. The 1.5k mix has **50** incomplete traces.

The eval system prompt **pushes the wrong policy**:

> If exactly one axis is derived, the remaining compounds that match it
> are the only possible answers.

That is exactly “pick the listed Northeast.”

---

## Failure 3 — dir-2 (86.5%)

Same derivation as incomplete, opposite menu. Both remaining compounds
**are** listed; gold is two letters. Model applies the incomplete /
none-of-options rule anyway.

Example (`doc_id` 133):

- Gold derived: `West proven … must be Northwest or Southwest` → **B, D**.
- Pred derived: the same line → **A** Cannot be determined.

28 wrong-set + 17 underselect (exactly one letter). This is the
incomplete rule leaking onto the case where both options exist — not
failure to close the axis.

---

## What is already solved at 1.5k

| claim | evidence |
|---|---|
| AxisGraph on civic maps | dir-1 97.8%, which-1 98.8%, count-1 98.5% |
| None-of-options when a theorem is missing | dir-omit **100%**, which-0 / count-omit **99.3%** |
| Both axes unknown (no cycle) | dir-undetermined **98.0%** |
| Multi-letter which | which-2 96.1%, which-3 98.0%, which-4 98.5% |
| Format / `Answer:` | 0 parse fails |
| Stage-2 readout | 8 items where it *hurts* |

Map size is a weak slope, not the hole:

| entities | strict | sentences | strict |
|---|---:|---|---:|
| 5 | 95.9% | 5 | 97.0% |
| 8 | 94.3% | 10 | 94.6% |
| 10 | 91.1% | 15 | 86.1% |

---

## Not item leak

Train is `generate_all_v6` seed 42, 20 civic names. Synth test is the
same generator, seed 43. SpatialMap eval is original shop names + fifth
option + `SpatialSolver` gold.

If the 1,500 SpatialMap rows had leaked into train, SpatialMap would
spike and synth would lag. Observed: SpatialMap **96.1% > synth 93.9%**.
That is “SpatialMap is the easier distribution” (0 cycles in the original
1,500; mostly determined 1-letter items). Synth train/test share a
template; they do not share SpatialMap maps. See Exp 8 RESULTS: 0 exact
name overlap, 0 shared map cores on the civic vs shop split.

The real sharing is: same solver labels train and both evals; same CoT
format; 1.5k mix is a census of the eval buckets (but **50** traces for
the two buckets that dominate error).

---

## Prompt gap

Eval `system_instruction` (and the generator system prompt) states:

- both axes proven → that compound
- one axis proven → the remaining compounds
- neither axis → Cannot be determined
- proven direction not listed → None of the Options

It does **not** state:

1. One axis proven **and both** remaining compounds listed → emit **both**
   letters (dir-2).
2. One axis proven **and only one** of those two listed → Cannot be
   determined; do not pick the listed compound (incomplete).
3. Both `A < B` and `B < A` on an axis → Cannot be determined (cycle).

(1) and (2) are currently the same sentence in the prompt, which is why
incomplete and dir-2 trade errors.

---

## What this cell does *not* justify

- **Dropping to 2B to “make it hard.”** Exp 8: 2B-20k was 59% on Single
  vs 4B-1.5k 95.5%. That is a bad student, not a harder map. Keep 2B as
  an Exp 11 scaling row when those cells finish.
- **GRPO on SpatialMap 96% or on this 94% mean.** Residual is 245 items,
  concentrated in three policies. RL on count-1 / SpatialMap has no
  group variance worth the job.
- **More unique 1.5k-style data of the same mix.** 400 count-1 vs 50
  cycle in train is why the mean looks done.
- **Growing maps as the next headline.** 5→10 entities is ~5 pp. Cycle
  is 37%.

---

## Next (what the 4k file supports)

1. **Patch the system prompt** with the three bullets above. Cheapest
   expected gain: incomplete is a recalled sentence (74/74 when written).
2. **Reweight SFT**, same 4B. Target mix heavy on cycle / incomplete /
   dir-2 (hundreds each, not 50). Stop adding dir-omit / count-1.
3. **Headline eval** = those three buckets, plus `--stages 1`. Report
   `cycle / incomplete / dir-2 / rest`, not 93.9% alone. SpatialMap v6
   stays the easy transfer check.
4. **GRPO after (1)+(2)** if letter set is still wrong with clean format.
   Prompt-only, exact-set reward, mix = cycle + incomplete + dir-2.

Suggested one-cell follow-up: same 4B, patched prompt, SFT mix with
~400 cycle + 400 incomplete + 400 dir-2, then the three-bucket table
on a new seed-43-style hard slice.
