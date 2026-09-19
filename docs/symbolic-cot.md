# Symbolic chain-of-thought for SpatialMap

Design note for the SFT traces in `spatial/generate_all_v6.py`. Oracle for the same calculus: `eval/clean_v6.py`. The previous generator (`spatial/generate_all.py`) is frozen.

Catalog of every question bucket, with examples from the 6k set: `docs/question-types.md`.

The traces are **not** free-form “think step by step.” They are an algorithm written in tokens: two posets (X and Y), closed under transitivity, then a lookup that marks each MCQ option In or Out.

---

## 1. What this CoT is

A completion is:

```
<think>
  Initialization
  Step 1 … Step N   (extract facts, rewrite the two axis states)
  Final Deduction   (read the states, score A–E)
</think>
Answer: B
```

That is chain-of-thought in the original sense (Wei et al.): intermediate steps that **compute** the answer. The intermediate language is **symbolic**, not prose.

`<` is the only relation. On **X**, `A < B` means A is west of B. On **Y**, `A < B` means A is south of B. Flip it and you get `>` (east / north). We print `<` so a chain reads left-to-right as west→east and south→north.

---

## 2. Why symbols, not English

Cardinal direction on a map is the **product of two linear orders**. Proving “Church is Northeast of Pharmacy” is exactly:

1. derive `Pharmacy < Church` on X (east),
2. derive `Pharmacy < Church` on Y (north),
3. conjoin.

A Transformer that must emit the running order (`Pharmacy < Zoo < Coffee Shop < Church`) is using the page as working memory. That is the Feng et al. picture: some problems need serialized state; CoT is that scratchpad.

For **spatial** maps this is not a taste call. Chain-of-Symbol prompting (Hu et al., COLM 2024) compared NL CoT to condensed relation symbols on spatial tasks and beat NL CoT on every setting they tried, with fewer tokens. Related: SymbCoT (FOL + CoT), Symbolic-Aided CoT (rule match → new premise → **update KB**), OpenAI process supervision (reward each step). CoT gains in the “To CoT or not to CoT” study are concentrated on **math/symbolic execution** — this task is that class.

So: symbolic chaining is a legitimate CoT, and for this domain it is the more natural one.

---

## 3. Trace structure

### 3.1 Initialization

List every name that will appear. Both states start unordered:

```
**Entities Detected**: Bank, Church, Coffee Shop, …
**Initial X-State**: Bank, Church, Coffee Shop, …
**Initial Y-State**: Bank, Church, Coffee Shop, …
```

### 3.2 Each sentence → two atomic facts

A compound in English always splits into one X edge and one Y edge:

| Phrase | X (west → east) | Y (south → north) |
|---|---|---|
| A is **Northeast** of B | `B < A` | `B < A` |
| A is **Northwest** of B | `A < B` | `B < A` |
| A is **Southeast** of B | `B < A` | `A < B` |
| A is **Southwest** of B | `A < B` | `A < B` |

The step then **reprints the whole axis state** after merging the new edges (not only the new pair).

### 3.3 How `A < B < C` is produced

`AxisGraph.format_state` in `generate_all_v6.py`:

1. Keep all extracted `<` edges.
2. Close under transitivity: `A < B` and `B < C` ⇒ `A < C`.
3. Drop skip-edges in the printout (`A < C` is hidden if `A < B < C` is there).
4. Names with the same neighbors become `{A, B}` (order between them unknown).
5. Disconnected components are comma-separated: `Church < Hospital, Church < Zoo < Museum`.

The **proof** does not parse that pretty string. Gold uses the same closed edge set: `(A, B)` in the X-closure iff `A < B` on X is derived. The printed chain is the human-readable form of that closure. Those two must not drift.

### 3.4 Final deduction

Same script for every question type:

1. Restate **X-State** / **Y-State**.
2. Look up the queried names (or the reference + side).
3. Say what is **derived** vs **unknown**.
4. Mark each option **In** or **Out** with the fact that decides it.
5. Last line: `Answer:` the In letters.

---

## 4. Gold rules (no 4-ans)

Necessity, not “everything consistent is correct.”

### Type 0 — `In which direction is Target relative to Ref?`

Options A–D are directions; **E. Cannot be determined** is always present.

| Axes derived | Derived direction | Gold |
|---|---|---|
| both (e.g. East and North) | one compound (Northeast) | that one letter |
| exactly one (e.g. South) | must be SE **or** SW | **both** remaining compounds |
| neither | none | **E** |

Cardinals (`North`, `East`, …) on Type 0 are distractors. A unique answer is a **two-axis compound**. E is Out if any axis was derived.

One-axis items **always list both live compounds** on A–D. SE without SW would look like unique-direction gold; E would also be wrong (South *was* derived). The generator drops the sample if that happens.

Fully unknown is **not** `A,B,C,D`. None of the four compounds is proven.

### Type 1 — `Which object is in the [Dir] of Ref?`

An option is In iff the entity is **definitely** on every required axis (same test as `get_entities_in_direction`). Unproven ≠ In. There is **no** “keep all passage-known names” fallback (that was original SpatialMap which-4-ans). Mix is 1-ans or 2-ans only.

### Type 2 — `How many objects are in the [Dir] of Ref?`

Count the definite set, then match the integer option. Always one correct number.

Oracle for this calculus: `eval/clean_v6.py`. `clean_v5.py` is the old SpatialMap cleaner (which-4-ans fallback kept).

---

## 5. Worked example (Type 0, 1-ans)

Question: Church relative to Pharmacy. Gold: **B. Northeast**.

```
Consider a map with multiple locations:

The Zoo is to the Northeast of the Pharmacy. The Church is to the Southeast of the Coffee Shop. The Zoo is to the Northeast of the Fire Department. The Church is to the Northeast of the Pharmacy. The Fire Department is to the Southwest of the Coffee Shop. The Zoo is to the Southwest of the Coffee Shop. The Pharmacy is to the Southwest of the Coffee Shop. The Bank is to the Southwest of the Coffee Shop.

Question: In which direction is the Church relative to the Pharmacy? Available options: A. North, B. Northeast, C. Southwest, D. Northwest, E. Cannot be determined

<think>
### Initialization
**Entities Detected**: Bank, Church, Coffee Shop, Fire Department, Pharmacy, Zoo
**Initial X-State**: Bank, Church, Coffee Shop, Fire Department, Pharmacy, Zoo
**Initial Y-State**: Bank, Church, Coffee Shop, Fire Department, Pharmacy, Zoo

### Step 1
**Sentence**: "The Zoo is to the Northeast of the Pharmacy."
**X-Extraction**: Pharmacy < Zoo
**Y-Extraction**: Pharmacy < Zoo
**X-State**: Pharmacy < Zoo
**Y-State**: Pharmacy < Zoo

### Step 2
**Sentence**: "The Church is to the Southeast of the Coffee Shop."
**X-Extraction**: Coffee Shop < Church
**Y-Extraction**: Church < Coffee Shop
**X-State**: Coffee Shop < Church, Pharmacy < Zoo
**Y-State**: Church < Coffee Shop, Pharmacy < Zoo

### Step 3
**Sentence**: "The Zoo is to the Northeast of the Fire Department."
**X-Extraction**: Fire Department < Zoo
**Y-Extraction**: Fire Department < Zoo
**X-State**: Coffee Shop < Church, {Fire Department, Pharmacy} < Zoo
**Y-State**: Church < Coffee Shop, {Fire Department, Pharmacy} < Zoo

### Step 4
**Sentence**: "The Church is to the Northeast of the Pharmacy."
**X-Extraction**: Pharmacy < Church
**Y-Extraction**: Pharmacy < Church
**X-State**: Coffee Shop < Church, Fire Department < Zoo, Pharmacy < Church, Pharmacy < Zoo
**Y-State**: Fire Department < Zoo, Pharmacy < Church < Coffee Shop, Pharmacy < Zoo

### Step 5
**Sentence**: "The Fire Department is to the Southwest of the Coffee Shop."
**X-Extraction**: Fire Department < Coffee Shop
**Y-Extraction**: Fire Department < Coffee Shop
**X-State**: Fire Department < Coffee Shop < Church, Fire Department < Zoo, Pharmacy < Church, Pharmacy < Zoo
**Y-State**: Fire Department < Coffee Shop, Fire Department < Zoo, Pharmacy < Church < Coffee Shop, Pharmacy < Zoo

### Step 6
**Sentence**: "The Zoo is to the Southwest of the Coffee Shop."
**X-Extraction**: Zoo < Coffee Shop
**Y-Extraction**: Zoo < Coffee Shop
**X-State**: {Fire Department, Pharmacy} < Zoo < Coffee Shop < Church
**Y-State**: Fire Department < Zoo < Coffee Shop, Pharmacy < Church < Coffee Shop, Pharmacy < Zoo < Coffee Shop

### Step 7
**Sentence**: "The Pharmacy is to the Southwest of the Coffee Shop."
**X-Extraction**: Pharmacy < Coffee Shop
**Y-Extraction**: Pharmacy < Coffee Shop
**X-State**: {Fire Department, Pharmacy} < Zoo < Coffee Shop < Church
**Y-State**: Fire Department < Zoo < Coffee Shop, Pharmacy < Church < Coffee Shop, Pharmacy < Zoo < Coffee Shop

### Step 8
**Sentence**: "The Bank is to the Southwest of the Coffee Shop."
**X-Extraction**: Bank < Coffee Shop
**Y-Extraction**: Bank < Coffee Shop
**X-State**: Bank < Coffee Shop < Church, {Fire Department, Pharmacy} < Zoo < Coffee Shop < Church
**Y-State**: Bank < Coffee Shop, Fire Department < Zoo < Coffee Shop, Pharmacy < Church < Coffee Shop, Pharmacy < Zoo < Coffee Shop

### Final Deduction
**Target**: Church
**Reference**: Pharmacy
**X-State**: Bank < Coffee Shop < Church, {Fire Department, Pharmacy} < Zoo < Coffee Shop < Church
**Y-State**: Bank < Coffee Shop, Fire Department < Zoo < Coffee Shop, Pharmacy < Church < Coffee Shop, Pharmacy < Zoo < Coffee Shop
**X-axis**: Pharmacy < Church → Church is East of Pharmacy.
**Y-axis**: Pharmacy < Church → Church is North of Pharmacy.
**Derived direction**: Northeast (both axes proven).
**Options**:
- A. North — only one axis; live answers are two-axis compounds (or E). Out.
- B. Northeast — East and North both derived. In.
- C. Southwest — needs West, but X-axis is East. Out.
- D. Northwest — needs West, but X-axis is East. Out.
- E. Cannot be determined — East and North are derived, so the direction is not fully unknown. Out.
</think>
Answer: B
```

Reading the example:

- Step 1 is one edge. Step 2 is a **second** chain (names do not meet). `{Fire Department, Pharmacy}` in step 3 means both are west of Zoo, order between them unknown.
- Step 4 already writes `Pharmacy < Church` on **both** axes — that is the proof of Northeast. Later steps only add a bridge (`Zoo < Coffee Shop`) so the printout becomes one long X-chain.
- Final deduction does not re-parse the sentences. It looks up `Pharmacy < Church` in the closed states, then rejects North (one axis), SW/NW (wrong X), and E (an axis *was* derived).

---

## 6. The other Type 0 endings (same lookup)

**One axis (2-ans).** Y has `Gas Station < Park` (North). X has Park and Gas Station in different chains.

```
**X-axis**: neither Gas Station < Park nor Park < Gas Station → East/West unknown.
**Y-axis**: Gas Station < Park → Park is North of Gas Station.
**Derived direction**: North proven, East/West unknown → must be Northeast or Northwest.
- A. Northeast — North derived; East/West unknown. In.
- B. Northwest — North derived; East/West unknown. In.
- D. Southwest — needs South, but Y is North. Out.
- E. … — North is derived, so not fully unknown. Out.
Answer: A, B
```

**Neither axis (E).** Both names sit under Church, but not ordered vs each other.

```
**X-State**: Church < Hospital, Church < Zoo < Museum
**X-axis**: neither Zoo < Hospital nor Hospital < Zoo → East/West unknown.
**Y-axis**: neither Zoo < Hospital nor Hospital < Zoo → North/South unknown.
**Derived direction**: none (both axes unknown).
- A–D compounds — not proven. Out.
- E. Cannot be determined — neither axis derived. In.
Answer: E
```

**Which** uses the same states the other way: fix the reference, collect who sits on the required side(s), intersect, then In/Out per option (`Zoo < Coffee Shop → North. In.` / `Fire Department < Zoo → South, not North. Out.`). **Count** is the size of that intersection.

---

## 7. Implementation

| Piece | Where |
|---|---|
| Shared gold rules | `spatial/spatial_solver.py` (`SpatialSolver.grade`) — **only** gold |
| World + traces | `spatial/generate_all_v6.py` proposes a map; keeps the sample iff `grade.accept` |
| Type 0 undetermined count | `--num-type0-undetermined` (gold E). No 4-ans flag. |
| SpatialMap JSONL labels | `eval/clean_v6.py` (calls the same solver) |
| Old SpatialMap cleaner | `eval/clean_v5.py` (do not edit for this calculus) |

SFT last line is always `Answer: A` or `Answer: A, B` or `Answer: E`. Letters come **from the In/Out list**, not from a separate cartesian product.

---

## 8. Caveats (so the CoT stays a proof)

- **Printed state = closed edges used for gold.** If `format_state` and the closure disagree, the model learns a caption, not a calculus.
- **Rigid headers teach format.** Vary map size, mix 1-ans / 2-ans / E / which / count, keep option-level In/Out so a wrong chain cannot luck into the letter.
- **Do not mix possibility and necessity.** “Which of these is possible?” is a different question. Under that stem, fully unknown would be A–D again and E would be wrong. This project’s stem is “in which direction,” so unknown is E.

---

## 9. Pointers

- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (2022). CoT = intermediate steps; formal and NL lineages both exist.
- Feng et al., *Let’s Think Algorithmically* (arXiv:2305.15408). CoT as serialized state for problems that need sequential computation.
- Hu et al., *Chain-of-Symbol Prompting for Spatial Reasoning* (COLM 2024, arXiv:2305.10276). Symbols beat NL CoT on spatial maps.
- Xu et al., *Faithful Logical Reasoning via Symbolic Chain-of-Thought* (SymbCoT, arXiv:2405.18357).
- Nguyen et al., *Non-Interactive Symbolic-Aided Chain-of-Thought* (arXiv:2508.12425). KB update after each rule — same shape as a Step N.
- OpenAI, *Improving mathematical reasoning with process supervision* (2023). Reward the steps, not only the answer.
- Sprague et al., *To CoT or not to CoT* (arXiv:2409.12183). CoT helps mainly where the work is symbolic execution.
