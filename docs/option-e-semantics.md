# What "Option E" Means: Provable vs. Consistent Semantics for SpatialMap-TQA

*Research note. Explains how the corrected benchmark (SpatialMap-TQA-Corr / Corr-Full)
labels items, why a 5th option **E — "None of these is proven"** was introduced, and
exactly what E means for each of the three question types. Includes worked proofs and
concrete counterexamples. All figures computed on the committed data.*

---

## 1. The one-sentence summary

SpatialMap-TQA ships one gold letter per item. Re-deriving every gold with an
**axis-decomposition solver** (`eval/clean_v5.py`) shows ~42% of the shipped gold
conflicts with what the passage *logically forces*. The corrected benchmark scores on
what is **provable** — true in *every* layout consistent with the passage — instead of
what the original author *intended*. Option **E** is the label for "no listed option is
provable." E is *not* one thing: it is **three different situations** that the single
letter happens to cover, and they should be reported as three separate subtypes.

| Subtype | n | What E means here | Status |
|---|---|---|---|
| **dir** (Type 0) | 75 | Both axes undetermined → no listed direction is *forced* | clean "none proven" |
| **count** (Type 2) | 96 | The provable count is not among the 4 numeric options | pure **none-of-the-above** |
| **which** (Type 1) | 198 | No single option is *proven*, though one is merely *consistent* | **contested** — stricter than original |

---

## 2. The framework: axis decomposition

A diagonal direction is two independent one-dimensional facts glued together.
"Northeast" = *North* (a fact on the Y-axis) **and** *East* (a fact on the X-axis).
The framework never reasons about "Northeast" as a thing. It maintains **two separate
ledgers**:

- **X-axis** (west → east): is `A` provably east of `B`?
- **Y-axis** (south → north): is `A` provably north of `B`?

### Rules

**Rule 1 — a sentence is an edge on each axis it touches, and its inverse is free.**

```
"A is to the Southeast of B"
  ⇒ A is east of B   (X-edge: A > B)
  ⇒ A is south of B  (Y-edge: B > A)
  and inverses are free:
  ⇒ B is west of A,  B is north of A.
```

**Rule 2 — transitivity (closure).** If A is north of B and B is north of C, then A is
north of C, *even if the passage never states it*. Chaining these is where the proving
happens. After closure, an axis-relation between two objects is one of three states:

- `gt` — `A > B` is **proven** (forced)
- `lt` — `A < B` is **proven**
- `unknown` — neither can be derived

### "Provable" has a precise definition

> An option is **proven** iff it is true in **every** layout consistent with the passage.
> Equivalently, it is *forced* by the transitive closure of the two axis graphs.

This is the single decision that drives the whole corrected benchmark, and it is the
thing to name explicitly in the write-up. It trades **coverage for soundness**: it only
claims what it can guarantee, never what is merely likely.

*Data check:* all 1,500 passages are axis-wise consistent (0 X-cycles, 0 Y-cycles), so
the "consistent world" framing is well-founded on the actual data — there is no passage
that is itself self-contradictory.

---

## 3. How "number of answers" falls out of the axes

Because X and Y are independent, the number of provable letters is a direct function of
how many axes are forced. Verified on the data:

### dir (Type 0) — "In which direction is X relative to Y?"

| Provable axes | Surviving diagonals | Gold | n |
|---|---|---|---|
| 2 of 2 | 1 | single letter | 332 |
| 1 of 2 | 2 | two letters | 93 |
| 0 of 2 | 4 | none proven | 75 → **E** |

This is *the framework itself*, not a coincidence: each proven axis halves the four
diagonals. A two-letter dir gold (e.g. `A,B`) means exactly one axis is forced.

### which (Type 1) — "Which object is in the [Dir] of X?"

An option object is **in** the queried direction only if *every* required axis is forced.
The solver runs a **first pass** (collect options fully proven-in). If **zero** options
pass, it falls back to keeping every option whose name appears in the passage — i.e. all
four — yielding gold `A,B,C,D`. That fallback set is what becomes **which-E (198)**.

> **The 4 in "which-4" is a *fallback label*, not a claim that four objects are in the
> direction.** It means: "no option is provably in the direction, and none can be ruled
> out either."

### count (Type 2) — "How many objects are in the [Dir] of X?"

An object is counted only if it is provably in the direction (**both** axes forced, the
"definite" test). The solver returns the **definite count**. If that number is not among
the four numeric options, gold is **count-E (96)**. This is the literal
**none-of-the-above** case.

---

## 4. Worked examples (all verified against `clean_v5.solve`)

### 4.1 dir → two-letter answer (the "why two directions?" case)

Passage (3 objects, 2 sentences):

```
Charlie is in the map.
Alpha is to the Southeast of Charlie.
Charlie is to the Southwest of Beta.
```

Question: *In which direction is Alpha relative to Beta?* Options: SE / SW / NE / NW.

- **Y-axis:** Alpha south of Charlie, Charlie south of Beta ⇒ **Alpha south of Beta (proven).**
- **X-axis:** Alpha east of Charlie, Beta east of Charlie, but **never compared to each other** ⇒ **unknown.**

Product: `{south} × {east or west}` = **Southeast or Southwest** → gold `A,B`.

Two consistent layouts (both satisfy the passage) confirm the ambiguity is real:

```
GRID 1 (count: Alpha is SW)            GRID 2 (count: Alpha is SE)
     North ↑                                North ↑
   · Beta (10,1)                         · Beta (1,1)
       ⋱                                    ⋱
   · Charlie (0,0)                       · Charlie (0,0)
     ⋱                                    ⋱
   · Alpha (1,-1)                        · Alpha (10,-1)
   → East                                → East
   Alpha is SOUTHWEST of Beta            Alpha is SOUTHEAST of Beta
```

`clean_v5.solve` returns `A,B` for this item. The framework never *chooses* between the
two because the passage doesn't — picking one would fabricate a fact.

---

### 4.2 which → 4-letter fallback → E (the contested case)

Real item `spatialmap.tqa.2000.1` (original gold: **B**).

> *Which object is in the **Southwest** of Ice Queen Ice Cream?*

Southwest requires **south AND west** of the reference. Per option, per axis:

| Option | Object | Y-axis (needs south) | X-axis (needs west) | Verdict |
|---|---|---|---|---|
| A | Coral Crafts | proven **north** → wrong way | west ✓ | **OUT** (contradicted) |
| B | Narwhal's Novelties | unknown | unknown | unproven |
| C | Planetarium Prints | south ✓ | proven **east** → wrong way | **OUT** (contradicted) |
| D | Police Supply Store | south ✓ | unknown | unproven (1 axis missing) |

**Zero options have both axes proven** → first pass fails → fallback → `A,B,C,D` → **E**.

Two of the "proven" facts above are nontrivial — the passage never states them:

> **C is proven EAST of Ice Queen** (which disqualifies it, since west is required):
> Planetarium NE of Police Supply → Planetarium east of Police Supply;
> Coral NW of Police Supply → Police Supply east of Coral;
> Ice Queen SE of Coral → Coral east of Ice Queen.
> ∴ Planetarium ⟶ Police Supply ⟶ Coral ⟶ Ice Queen, all "east of" → **Planetarium east of Ice Queen (proven).**

**Why this is the contested subgroup:** the original gold is **B** (Narwhal's Novelties),
an object the passage says *nothing about*. The original author's intended answer is the
*least-supported* option, not a proven one. Under "provable" semantics, B is no more an
answer than A or C. This is not an error in the re-derivation — it is evidence the
original label was under-determined — but it is exactly where "E = none proven" is doing
**stricter-than-none-of-the-above** work. This is the 198-item subgroup to either
defend ("provable is the right bar for *which* questions") or relabel (restore A–D gold,
dropping E from 369 → 171).

Aggregate over all 198 which-4 items (792 option-slots):

| Option status | n |
|---|---|
| Proven **in** the queried direction | **0** (the trigger) |
| Proven **out** (contradicted) | 490 |
| Merely consistent (unproven, not contradicted) | 302 |

---

### 4.3 count → E (the cleanest case)

Real item `spatialmap.tqa.2241.2` (original gold: **C = 3**).

> *How many objects are in the **Southwest** of Circus Central?*
> Options: **A. 5, B. 0, C. 3, D. 4**

Southwest = south **and** west. Per object vs Circus Central:

| Object | South? | West? | Counted? |
|---|---|---|---|
| Tiger's Tapestries | proven IN | proven IN | **YES — definite** |
| Yak's Yarns | unknown | proven IN | only if south… |
| Unicorn's Umbrellas | unknown | unknown | only if both… |
| Monumental Memories | unknown | unknown | only if both… |
| Whale's Wicker | proven IN | proven **OUT** (east) | **NO — contradicted** |

**Definite count = 1** (only Tiger guaranteed); **possible count = up to 4**. Two
layouts that both satisfy all 9 passage sentences:

```
World 1 (count = 1)                    World 2 (count = 4)
     North ↑                                North ↑
  · Monumental                              · Circus (6,11)
  · Yak                                          ⋱
  · Unicorn                                · Yak (−1,10.5)
       · Circus (2,12)                     · Monumental (4,10.8)
            ⋱                              · Unicorn (1,10.2)
  · Tiger (0,0)                                 ⋱
                                            · Tiger (0,0)
  → East                                   → East
  SW of Circus: just Tiger              SW of Circus: all four
```

The solver returns the definite count = **1**, which is **not** in {5,0,3,4} → **E**.
This is the textbook *none-of-the-above*: the provable answer is a number the options
simply don't include. Note the original gold was 3 — right under a *possible/most-likely*
reading, not under the *definite* reading. So even count-E is where the two semantics
disagree, just less visibly than in which-198.

---

## 5. The distribution difference (the 1500-vs-1500 comparison)

Same 1,500 rows, identical question text. Gold changed on **628 / 1,500 (42%)**:

| Change | n | What happened |
|---|---|---|
| Unchanged | 872 (58%) | original gold kept |
| Single → **E** (empty) | 171 | dir 75 + count 96 (truly under-determined) |
| Single → 2 letters | 93 | two options provable (e.g. `C` → `C,D`) |
| Single → **E** (which-4) | 198 | type-1 fallback, remapped to E |
| Single → different single | 166 | e.g. original `D` now `A` |

Letter distributions:

| | A | B | C | D | E | notes |
|---|---|---|---|---|---|---|
| **Original** | 376 | 378 | 369 | 377 | — | perfectly balanced, all single-answer |
| **Corr-Full** | 316 | 292 | 316 | 300 | 369 | + 93 two-letter golds |

The shift is largest in the **which** type (χ² ≈ 247, Cramér's V ≈ 0.50), driven by the
198 which-4 → E remaps; smallest in **dir** (V ≈ 0.25).

This is what the "compare the distributions" comparison measures: **how much the
re-derived test differs from the original benchmark**, per type, per letter, per
cardinality — and (per-item) whether models match the *old* gold, the *new* gold, or
neither on the 628 relabeled rows.

---

## 6. Train/eval caveats to state honestly

1. **Option-space mismatch (dir).** Original dir questions *always* offer the four
   diagonals {NE, NW, SE, SW} and **never** a cardinal (N/S/E/W). The synthetic SFT data
   shows a cardinal option in ~75% of dir items, but **cardinals are never the gold**.
   So the model is trained in a world where "North" is a plausible option and evaluated
   on a benchmark where it never appears.
2. **dir-E train/eval mismatch.** SFT dir-E items are built with all 4 options
   *impossible*; eval dir-E items have options that are *consistent but unproven* (both
   axes unknown). "E" in training and "E" in test are different situations — worth one
   line in the write-up and a check that it isn't distorting E-accuracy.
3. **The which-198 relabel** is the item most worth an explicit decision with the
   advisor (see §4.2).

---

## 7. What to report / recommend

1. **Decompose E by type** in the 1500-vs-1500 comparison and report E-selection rate +
   accuracy on each of dir-75 / which-198 / count-96. This is the direct, data-backed
   answer to "what does E mean."
2. **Commit to one definition in a 3-sentence note:** *"E = none of A–D is provably
   correct (true in every layout consistent with the passage)."* Name the three subtypes.
3. **Make an explicit call on which-198.** Either justify "provable is the right bar"
   (keep E) or concede "consistent is the right bar" (restore A–D gold, E 369 → 171).
   The count-96 and dir-75 are defensible as *corrections*; which-198 is defensible only
   as a *stricter standard* — that distinction is the crux of the advisor's question.

---

## Appendix: dataset census (from `experiments/tasks/utils.py`)

```
Original SpatialMap-TQA (data/spatialeval_org.jsonl), 1500 rows
  500 dir (.0) / 500 which (.1) / 500 count (.2)
  1 gold letter per item; A 376 / B 378 / C 369 / D 377

SpatialMap-TQA-Corr (data/spatialeval_cleaned.jsonl), 1500 rows
  empty oracle            171   dir 75 + count 96
  Single (1 letter)      1038   count 404 + dir 332 + which 302
  Multi (2+ letters)      291   dir-2 93 + which-4 198

SpatialMap-TQA-Corr-Full (data/spatialeval_corr_full.jsonl), 1500 rows
  + "E. None of these is proven" on every item
  gold E (369): dir 75 + count 96 + which-4 198
  gold A–D (1131): original letters; E is a distractor
```
