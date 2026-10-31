# Depth-Two Collision Repair

## Cause

The completed fresh v3 audit found six distinct dev depth-2 Count questions
with entailed total four, against the unchanged requirement of seven.
This was not a wrong Count label or a final deduplication accounting error.
Adding multi-entailment Selection supply made two previously usable Count
scenes collide with training scenes:

| Dev Count Slot | Seed | Training Selection Slot | Seed | Direction |
|---|---:|---|---:|---|
| production-b1036-s1 | 2026111095 | production-b52-s11 | 2026112960 | West |
| production-b1038-s0 | 2026111518 | production-b66-s7 | 2026111360 | North |

Different scaffold seeds can collapse to the same shallow spatial scene.
The cross-split quarantine correctly rejected these scenes. It is preserved.
The two actual seed-pair regression tests failed before the repair and passed
after it.

## Bounded Repair

`all-scene-forest-v4` adds seed-dependent context only at depth two, using
the same rule across splits, directions, query types, and Count targets.
It does not special-case dev or a failing slot.

New context objects are directly constrained opposite the queried direction.
Additional acyclic links vary their relationships without changing the answer
support. These links are explicitly marked for individual and joint
irrelevance verification. The total remains within 31 ordinary place names,
32 premises, and the existing eight original candidate slots. Selection and
Count still range over every nonreference scene object, including new context.

The fixed seeds, source slots, split assignments, core answer supports,
family IDs, numeric quotas, and diagnostic 900-of-3,600 multi-Selection target
are unchanged. Extra context does not create independent families.
Depths other than two retain the exact previous story content and v3 recipe
identifier. The previous anchor identifiers remain unchanged.

## Verification

The targeted audit first reconstructs and validates the completed fresh v3
inventory and reproduces its exact supply failure. It then reconstructs current
scenes and requires every non-depth-two story to equal its previous value.
It recomputes scene isolation over the complete inventory.

Every changed eligible query, and any newly eligible query without prior
verification, receives fresh full admission, token checks, transformations,
and independent evidence replay. Unchanged records are reused only from the
completed fresh all-scene v3 run, with exact task/query/recipe checks and
recorded input hashes. No historical scoped checkpoint is used.

After canonical query deduplication, the audit checks every individual quota
and coverage requirement, then runs combined retention if supply permits.
Selected IDs, not a replacement training dataset, are the diagnostic output.
The v3 failure receipt remains unchanged.

## Completed Result

**Combined diagnostic retention passed.** The completed receipt is
`COMBINED-ALL-SCENE-FEASIBILITY-v4.json`.

| Retained Split | Bases |
|---|---:|
| Train | 6,000 |
| Dev | 1,200 |
| In-range test | 2,400 |
| Extrapolation test | 1,200 |
| Total | 10,800 |

- Exactly 3,600 bases for each query type.
- Exactly 900 Selection bases with 2-4 individually entailed objects.
- All exact base/source/depth/Count quotas and coverage/family requirements
  passed together in the selected subset.
- The previously deficient dev depth-2 Count-four group now supplies 20
  distinct verified candidates; exactly the required seven were selected.
- 1,374 depth-two story slots changed. Every other story was confirmed equal
  to its fresh v3 value, and no other depth's overlap status changed.
- 2,061 affected eligible query jobs were freshly admitted and independently
  replayed; all passed. Maximum new targets were 1,579 Natural and
  1,673 Symbolic tokens.
- 20,146 unchanged controlled records were validated against the reconstructed
  fresh v3 inventory and reused. The 7,632-slot fresh v3 source phase was
  validated separately; historical scoped checkpoints were not used.
- Cross-split quarantine still excludes 957 story slots. The revised eligible
  controlled workload is 22,207 queries, with no added seed or input slot.
- 25,848 distinct verified candidates were available for final selection.
- Full entailment regression suite: 842 passed. Changed-code Ruff passed.
- A separate post-run check validated the selected IDs, exact quotas, 10,800
  unique query keys, and family/story split isolation.

Targeted inventory validation, fresh admission, and retention took 701.2
seconds (approximately 11 minutes 41 seconds). The receipt records software,
prior-receipt, input-metadata, and changed-story hashes.

This clears the combined feasibility blocker. The result is a selected-ID
diagnostic, not an exported training package or approval of a final allocation.
Final artifact generation, prose review, sealing, and training remain separate.

No new seeds, numeric quota relaxation, split moves, dataset seal, or training.
