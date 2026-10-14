# SpatialEntail-ALL-POSSIBLE-Bench: draft cell registry and feasibility pilot

Status: draft, 2026-10-14. Not frozen, not sealed, and no model has been
evaluated on it. The registry is `spatial/v2/all-possible-cells.yaml`; the
generator is `spatial/v2/possibility.py`.

## Purpose

`SpatialEntail-SINGLE-Bench` was largely solved by the untuned base model on
CORE and COMP once answer selection was fair. Under `SINGLE`, every
underdetermined question collapses to one escape option, `Cannot be
determined`. `ALL_POSSIBLE` removes that escape: the model must return the
complete set of answers that could be true given the stated relations, and
leave out every impossible one. The answer space grows from one letter to any
non-empty subset of the menu.

This bench reuses the structures that made SINGLE's SHIFT section
discriminative (depth, Boolean rules, correlated Count, distractors), but every
cell plants two or more possible answers.

## Contract

- Answer mode `ALL_POSSIBLE`. Every menu shows exactly five options, matching
  SINGLE and the external five-option benchmarks, so option count never reveals
  how many answers are correct and transfer is not confounded by menu shape.
  For `ALL_POSSIBLE` the complete possible set is always among the five shown
  (so the full set is expressible); the gold set size is therefore capped at
  four, leaving at least one distractor. Partial menus, where some possible
  answers are omitted, are reserved for the `SpatialEntail-VISIBLE-POSSIBLE`
  probe.
- Prompt instruction: "Select every answer that could be true given the stated
  relations."
- Primary metric: exact-set accuracy. Diagnostics: per-option precision and
  recall, over- and under-inclusion, and accuracy by gold set size.
- Stage-2 constrained decoding already supports 1–N answer subsets over A–E.

## Cell design

Cell IDs follow the existing grammar, with `ALT` as the semantic status.

| Suffix | Meaning |
|---|---|
| `G2`, `G4` | One Boolean rule plants a gold set of exactly 2 or 4 possible answers |
| `D3`, `D5` | One axis is left unordered along a 3- or 5-link chain |
| `NOISE` | A depth-3 chain plus support-attached removable distractors |
| `JCR` | Planted memberships move together, so intermediate counts are impossible |

Skeletons, each for Direction, Selection, and Count unless noted:

- `MP`: a true trigger implies a disjunction of planted answers.
- `IFF`: the same, through an equivalence.
- `DJE`: a disjunction of planted answers plus one alternative that a premise
  rules out.
- `CSR`: one object pair takes exactly one of k exact directions, and each case
  implies a different planted answer.
- `TRC` depth: a chain fixes one axis strictly and leaves the other open.
  Selection and Count use one open chain and one fixed chain.
- `NOISE` (Direction, Selection): the depth skeleton plus premises that hang
  spare objects off the support. They look relevant but are removable.
- `CNT-CSR-JCR` (Count): the planted group is all in or all out, so the
  possible counts are {0, k} rather than {0, ..., k}.

The draft has 34 cells: 24 Boolean (4 skeletons × 2 gold sizes × 3 query
kinds), 6 depth, 2 noise, and 2 joint-count.

Every menu holds exactly five options, so the gold set size cannot be read off
the menu. Direction shows the possible directions plus other directions as
distractors; Selection and Count pad with distractor candidates.

## Feasibility pilot

50 samples per cell, seed `31000 + cell index`, generated after the inert
tautology fix described in
[`certificate-search-bottleneck.md`](certificate-search-bottleneck.md). Every
sample was solved, matched its declared shape, built a replayable answer
certificate, and passed the prompt round trip.

- Accepted: 1,700 of 1,700 requested; every cell was accepted within its
  attempt budget.
- Every menu held exactly five options, and every gold set was a subset of the
  shown options.
- Unique structure signatures: 50 of 50 in every cell.
- Wall time: 1,443 s on one laptop CPU, about 0.85 s per row.
- Longest Natural trace: 15,810 characters, roughly 4K tokens, well inside the
  12,288-token training budget.

| Cell | Accepted | Gold set size | Menu | Median / max trace chars | Seconds |
|---|---:|---:|---:|---:|---:|
| `DIR-MP-ALT-G2` | 50/50 | 2 | 5 | 10,011 / 12,350 | 14.4 |
| `DIR-MP-ALT-G4` | 50/50 | 4 | 5 | 11,945 / 12,945 | 20.3 |
| `DIR-IFF-ALT-G2` | 50/50 | 2 | 5 | 10,034 / 11,169 | 14.3 |
| `DIR-IFF-ALT-G4` | 50/50 | 4 | 5 | 12,014 / 12,954 | 19.4 |
| `DIR-DJE-ALT-G2` | 50/50 | 2 | 5 | 10,134 / 11,076 | 13.0 |
| `DIR-DJE-ALT-G4` | 50/50 | 4 | 5 | 10,345 / 11,137 | 17.7 |
| `DIR-CSR-ALT-G2` | 50/50 | 2 | 5 | 11,409 / 12,404 | 13.3 |
| `DIR-CSR-ALT-G4` | 50/50 | 4 | 5 | 14,323 / 15,229 | 19.3 |
| `DIR-TRC-ALT-D3` | 50/50 | 3 | 5 | 6,798 / 7,681 | 18.0 |
| `DIR-TRC-ALT-D5` | 50/50 | 3 | 5 | 8,176 / 10,318 | 70.5 |
| `DIR-TRC-ALT-NOISE` | 50/50 | 3 | 5 | 6,657 / 7,599 | 90.6 |
| `SEL-MP-ALT-G2` | 50/50 | 2 | 5 | 4,117 / 4,492 | 35.8 |
| `SEL-MP-ALT-G4` | 50/50 | 4 | 5 | 4,506 / 4,977 | 42.2 |
| `SEL-IFF-ALT-G2` | 50/50 | 2 | 5 | 4,068 / 4,471 | 36.0 |
| `SEL-IFF-ALT-G4` | 50/50 | 4 | 5 | 4,456 / 5,055 | 43.7 |
| `SEL-DJE-ALT-G2` | 50/50 | 2 | 5 | 4,123 / 4,453 | 30.2 |
| `SEL-DJE-ALT-G4` | 50/50 | 4 | 5 | 4,498 / 5,095 | 37.4 |
| `SEL-CSR-ALT-G2` | 50/50 | 2 | 5 | 4,201 / 4,441 | 36.5 |
| `SEL-CSR-ALT-G4` | 50/50 | 4 | 5 | 4,666 / 5,090 | 43.6 |
| `SEL-TRC-ALT-D3` | 50/50 | 2 | 5 | 4,093 / 4,563 | 36.5 |
| `SEL-TRC-ALT-D5` | 50/50 | 2 | 5 | 4,963 / 5,696 | 107.7 |
| `SEL-TRC-ALT-NOISE` | 50/50 | 2 | 5 | 4,073 / 4,446 | 80.0 |
| `CNT-MP-ALT-G2` | 50/50 | 2 | 5 | 5,906 / 6,253 | 45.8 |
| `CNT-MP-ALT-G4` | 50/50 | 4 | 5 | 6,595 / 7,068 | 56.4 |
| `CNT-IFF-ALT-G2` | 50/50 | 2 | 5 | 5,908 / 6,250 | 43.8 |
| `CNT-IFF-ALT-G4` | 50/50 | 4 | 5 | 6,554 / 7,132 | 53.5 |
| `CNT-DJE-ALT-G2` | 50/50 | 2 | 5 | 5,915 / 6,495 | 37.0 |
| `CNT-DJE-ALT-G4` | 50/50 | 4 | 5 | 6,532 / 7,074 | 45.8 |
| `CNT-CSR-ALT-G2` | 50/50 | 2 | 5 | 6,212 / 6,538 | 43.3 |
| `CNT-CSR-ALT-G4` | 50/50 | 4 | 5 | 7,023 / 7,478 | 51.8 |
| `CNT-TRC-ALT-D3` | 50/50 | 2 | 5 | 4,711 / 5,122 | 45.8 |
| `CNT-TRC-ALT-D5` | 50/50 | 2 | 5 | 5,673 / 6,116 | 125.5 |
| `CNT-CSR-JCR-ALT-G2` | 50/50 | 2 | 5 | 7,216 / 7,908 | 25.7 |
| `CNT-CSR-JCR-ALT-G3` | 50/50 | 2 | 5 | 14,104 / 15,810 | 28.5 |

## Known limitations before freezing

1. **Gold set size is fixed per cell.** Each cell plants exactly one set size,
   for example `G4` is always size 4. That keeps cells interpretable, but a
   model could learn the size from structure. The bench should mix gold sizes
   across cells, and diagnostics must report accuracy by set size.
2. **Depth and noise cells plant small sets.** Direction depth cells always
   give 3 possible directions; Selection and Count depth cells always give 2.
   Larger open sets would need more open chains.
3. **Not calibrated on a model yet.** The pilot shows the cells can be
   generated; it does not show they are hard. The next step is an untuned
   Qwen3.5-4B run on an engineering pool of about 25 per cell before choosing
   allocations, as was done for SINGLE.
4. **No disjointness audit yet.** Freezing requires canonical-structure
   disjointness from every `SpatialEntail-SINGLE` split, as in the SINGLE
   release.
5. **Boolean Direction traces are long.** They reach about 10–15K characters
   because every one of the eight directions needs a witness or a refutation,
   even though only five are shown in the menu.

## Next steps

1. Generate an untuned engineering pool, about 25 per cell, and evaluate it with
   stage-2 constrained decoding.
2. Drop or harden any cell the untuned model saturates; keep cells with clear
   headroom.
3. Fix per-cell allocations, run the disjointness audit, and seal
   `SpatialEntail-ALL-POSSIBLE-Bench`.
4. Evaluate untuned and the `SpatialEntail-SINGLE-SFT` adapters on it to measure
   transfer to the stricter contract.
5. Build the paired `SpatialEntail-VISIBLE-POSSIBLE` probe from the same bases
   with partial menus.
