# Verified Unique Training Candidate

The approved allocation is now implemented, selected, exported and audited.
Status: **validated, unsealed candidate**. No model training has started.

## Training Allocation

Each style receives the same 6,000 distinct questions, with exactly one menu
per question and style-specific evidence-based explanations. Natural and
Symbolic are two matched training arms, not 12,000 independent questions.

| Query | A-D | E | F | Examples per style |
|---|---:|---:|---:|---:|
| Direction | 1,200 | 600 | 200 | 2,000 |
| Which | 1,200 | 600 | 200 | 2,000 |
| Count | 1,200 | 600 | 200 | 2,000 |
| Total | 3,600 | 1,800 | 600 | 6,000 |

| Query | Multiple entailed E | No entailed value E | Inconsistent E |
|---|---:|---:|---:|
| Direction | 0 | 450 | 150 |
| Which | 300 | 225 | 75 |
| Count | 0 | 450 | 150 |

F always means one sole entailed value absent from A-D, never multiple missing
answers. Each F question has a positive proof in its selected explanation.
For Which, sole entailment also requires checked non-entailment of alternatives.

## Source And Depth

Per query: 1,400 controlled, 300 random-world, 200 random-atoms, 100 targeted.
The 70/15/10/5 source allocation was preserved without relaxation.

| Controlled depth | Per query | All queries |
|---|---:|---:|
| L1 | 112 | 336 |
| L2 | 126 | 378 |
| L3 | 140 | 420 |
| L4 | 154 | 462 |
| L5 | 182 | 546 |
| L6 | 196 | 588 |
| L7 | 224 | 672 |
| L8 | 266 | 798 |
| Total | 1,400 | 4,200 |

The remaining 600 questions per query form the semantic-source cohort, not
an invented L9/L10 or ambiguity-depth bucket. Incidental valid depth measurements
are retained where supported; no training row has measured depth above eight.

Selected-explanation depth coverage: Direction 1,400; Which 1,416; Count 1,400,
all meeting the retained minimum of 1,400. Which includes valid cheapest-pair
E certificates. All controlled Count-value, family, direction-presence,
nonhomogeneous-proof and negative-evidence coverage requirements passed.

The original 4,800 synthetic evaluation bases are unchanged. Dev and in-range
controlled depths remain uniform L1-L8; extrapolation remains equally L9/L10.
The separately audited external Unique benchmark is unchanged.

## Files

- [Natural training examples](natural-training.jsonl): 6,000 chat examples,
  including the assistant's reference explanation.
- [Symbolic training examples](symbolic-training.jsonl): the same 6,000 question
  and menu IDs, with Symbolic reference explanations.
- [References](references.jsonl): one selected question/menu row per base.
- [Evidence](evidence.jsonl): full checked evidence for each selected base.
- [Manifest](manifest.json): policy, counts, payload hashes and unsealed status.

All use short SpatialMap-style questions, ordinary place names, all-scene scope,
untouched premise paragraphs and shared system rules. Tokenization was rechecked
with the pinned Qwen3.5-4B tokenizer. Maximum reference lengths are 2,879 Natural
and 3,175 Symbolic tokens, within the 4,096 supervision cap; Stage-2 reservations
also pass the 8,192 context limit.

## Verification And Provenance

- Two separately approved fixed supplementary batches were exhausted: 2,048
  slots and 1,600 slots. No automatic third batch or refill was used.
- All 15,450 eligible training candidates completed evidence/reference replay
  before joint one-menu selection.
- [Selection receipt](../UNIQUE-TRAINING-SELECTION-REPLAY-v1.json) records the
  6,000 selected base/menu IDs and unchanged evaluation IDs.
- [Final export audit](../UNIQUE-TRAINING-EXPORT-AUDIT-v1.json) passed all exported
  rows, both style files, semantic label cases, F proof depth, exact quotas,
  coverage checks and immutable upstream evidence provenance.

Historical candidate datasets, receipts and evaluation artifacts were not
overwritten. The 258 MB JSONL payloads stay local; manifests and audit reports
remain visible to version control. Ignore rules are not backups.

This completes the allocation/export goal, not experiment execution. Manual
prose review and separate release/sealing/training approval remain. Existing
internal wording such as "Unique requires exactly one" has not been silently
rewritten during this allocation task.
