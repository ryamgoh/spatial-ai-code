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

- Answer mode `ALL_POSSIBLE`, full menus only: every possible value is shown,
  so the complete set is always expressible and `Cannot be determined` never
  appears. Partial menus are reserved for the `SpatialEntail-VISIBLE-POSSIBLE`
  probe.
- Prompt instruction: "Select every answer that could be true given the stated
  relations."
- Primary metric: exact-set accuracy. Diagnostics: per-option precision and
  recall, over- and under-inclusion, and accuracy by gold set size.
- Stage-2 constrained decoding already supports 1–N answer subsets.

## Cell design

Cell IDs follow the existing grammar, with `ALT` as the semantic status.

| Suffix | Meaning |
|---|---|
| `W2`, `W4` | One Boolean rule plants exactly 2 or 4 possible answers |
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

The draft has 34 cells: 24 Boolean (4 skeletons × 2 widths × 3 query kinds), 6
depth, 2 noise, and 2 joint-count.

Menus are sized so the gold set size cannot be read off the menu. Direction
shows all eight directions. Selection and Count add at least three distractors.

## Feasibility pilot

50 samples per cell, seed `31000 + cell index`, generated after the inert
tautology fix described in
[`certificate-search-bottleneck.md`](certificate-search-bottleneck.md). Every
sample was solved, matched its declared shape, built a replayable answer
certificate, and passed the prompt round trip.

- Accepted: 1,700 of 1,700 requested; every cell was accepted on its first
  attempt.
- Unique structure signatures: 50 of 50 in every cell.
- Wall time: 1,258 s on one laptop CPU, about 0.74 s per row.
- Longest Natural trace: 15,638 characters, roughly 4K tokens, well inside the
  12,288-token training budget.

| Cell | Accepted | Gold set size | Menu size | Median / max trace chars | Seconds |
|---|---:|---|---|---:|---:|
| `DIR-MP-ALT-W2` | 50/50 | 2 | 8 | 9,900 / 10,875 | 14.0 |
| `DIR-MP-ALT-W4` | 50/50 | 4 | 8 | 11,972 / 13,152 | 19.5 |
| `DIR-IFF-ALT-W2` | 50/50 | 2 | 8 | 10,058 / 10,790 | 13.5 |
| `DIR-IFF-ALT-W4` | 50/50 | 4 | 8 | 11,932 / 13,257 | 19.7 |
| `DIR-DJE-ALT-W2` | 50/50 | 2 | 8 | 10,220 / 10,939 | 12.5 |
| `DIR-DJE-ALT-W4` | 50/50 | 4 | 8 | 10,267 / 11,054 | 17.4 |
| `DIR-CSR-ALT-W2` | 50/50 | 2 | 8 | 11,493 / 12,291 | 13.0 |
| `DIR-CSR-ALT-W4` | 50/50 | 4 | 8 | 14,379 / 15,638 | 18.7 |
| `DIR-TRC-ALT-D3` | 50/50 | 3 | 8 | 6,668 / 7,495 | 17.9 |
| `DIR-TRC-ALT-D5` | 50/50 | 3 | 8 | 8,654 / 10,541 | 41.7 |
| `DIR-TRC-ALT-NOISE` | 50/50 | 3 | 8 | 6,686 / 7,566 | 29.6 |
| `SEL-MP-ALT-W2` | 50/50 | 2 | 5 | 4,117 / 4,492 | 33.8 |
| `SEL-MP-ALT-W4` | 50/50 | 4 | 7 | 4,475 / 4,860 | 39.6 |
| `SEL-IFF-ALT-W2` | 50/50 | 2 | 5 | 4,068 / 4,471 | 33.6 |
| `SEL-IFF-ALT-W4` | 50/50 | 4 | 7 | 4,552 / 5,055 | 41.3 |
| `SEL-DJE-ALT-W2` | 50/50 | 2 | 5 | 4,123 / 4,453 | 29.2 |
| `SEL-DJE-ALT-W4` | 50/50 | 4 | 7 | 4,502 / 4,872 | 35.5 |
| `SEL-CSR-ALT-W2` | 50/50 | 2 | 5 | 4,201 / 4,441 | 34.0 |
| `SEL-CSR-ALT-W4` | 50/50 | 4 | 7 | 4,569 / 5,060 | 41.3 |
| `SEL-TRC-ALT-D3` | 50/50 | 2 | 5 | 4,093 / 4,563 | 35.3 |
| `SEL-TRC-ALT-D5` | 50/50 | 2 | 5 | 4,963 / 5,696 | 101.2 |
| `SEL-TRC-ALT-NOISE` | 50/50 | 2 | 5 | 4,073 / 4,446 | 71.5 |
| `CNT-MP-ALT-W2` | 50/50 | 2 | 5 | 5,906 / 6,253 | 39.5 |
| `CNT-MP-ALT-W4` | 50/50 | 4 | 7 | 6,565 / 7,121 | 48.9 |
| `CNT-IFF-ALT-W2` | 50/50 | 2 | 5 | 5,908 / 6,250 | 38.8 |
| `CNT-IFF-ALT-W4` | 50/50 | 4 | 7 | 6,577 / 7,083 | 47.6 |
| `CNT-DJE-ALT-W2` | 50/50 | 2 | 5 | 5,915 / 6,495 | 34.6 |
| `CNT-DJE-ALT-W4` | 50/50 | 4 | 7 | 6,636 / 7,077 | 42.3 |
| `CNT-CSR-ALT-W2` | 50/50 | 2 | 5 | 6,212 / 6,538 | 40.1 |
| `CNT-CSR-ALT-W4` | 50/50 | 4 | 7 | 7,030 / 7,577 | 48.8 |
| `CNT-TRC-ALT-D3` | 50/50 | 2 | 6 | 4,642 / 4,940 | 41.0 |
| `CNT-TRC-ALT-D5` | 50/50 | 2 | 6 | 5,695 / 6,246 | 112.3 |
| `CNT-CSR-JCR-ALT-W2` | 50/50 | 2 | 5 | 7,216 / 7,908 | 23.6 |
| `CNT-CSR-JCR-ALT-W3` | 50/50 | 2 | 6 | 14,097 / 15,419 | 26.4 |

## Known limitations before freezing

1. **Gold set size is fixed per cell.** Each cell plants exactly one set size,
   for example `W4` is always size 4. That keeps cells interpretable, but a model
   could learn the size from structure. The bench should mix widths across
   cells, and diagnostics must report accuracy by set size.
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
5. **Direction traces are long.** Boolean Direction cells reach about 10–15K
   characters because every one of the eight directions needs a witness or a
   refutation.

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
