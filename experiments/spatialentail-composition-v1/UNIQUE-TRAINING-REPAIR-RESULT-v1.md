# Bounded Training Repair Result

Status: **approved batch completed; exact training allocation still infeasible**.
The 1,200 A-D / 600 E / 200 F target per query remains unchanged and unachieved.
No further batch, refill, source-mix change, seal or training is authorized.

## Completed Work

Implemented the approved training-only selected-explanation depth predicate,
keeping the numerical minimum at 1,400 for each query. It measures the proof
emitted in the selected A-D/F explanation. Which multiple-entailment E receives
credit only when the emitted pair is a cheapest-pair certificate. Ambiguous and
inconsistent E receive no invented positive-proof depth.

Historical depth flags and existing evaluation artifacts remain unchanged.
A separate audit helper independently checks certificate lengths and verifies
the reference text against replayed evidence before attaching selected-depth
audit markers. Compact historical metadata has not been promoted to audited
selected-explanation evidence.

Implemented a one-menu selector with exact answer/subtype, source, controlled
depth, Count-value, family and inherited coverage constraints. Its focused tests
pass. It has not selected a production dataset: the post-batch source bounds
below prove that joint selection cannot succeed on this pool.

## Fixed Batch

Manifest: [UNIQUE-TRAINING-REPAIR-MANIFEST-v1.json](UNIQUE-TRAINING-REPAIR-MANIFEST-v1.json).
Manifest SHA-256:
`eabb79f399786592895f1aa845d067439178df2feacc975837ea755e01c85a53`.

| Outcome | Query slots |
|---|---:|
| Accepted, independently replayed, canonical-query distinct | 815 |
| Rejected before admission: declared objects absent from premises | 1,187 |
| Excluded before admission: held-out canonical scene overlap | 46 |
| Total fixed slots | 2,048 |

Accepted additions: 403 Direction, 245 Which, 167 Count.
There were no verification failures, token rejections or replacement slots.
The new pool contains 14,865 training candidates, up from 14,050.
Accepted candidates include all qualifying outcomes, not just desired answer
classes. An accepted candidate does not necessarily fill a scarce quota.

The run used the approved fixed source/query caps, disjoint recorded seed
ranges, existing random samplers, and the existing targeted contingent sampler.
The original 70/15/10/5 source quotas and existing seeded omission eligibility
were not changed. Rejected slots still consumed the batch authorization.

Admission receipt:
[UNIQUE-TRAINING-REPAIR-ADMISSION-v1.json](UNIQUE-TRAINING-REPAIR-ADMISSION-v1.json).
Full evidence and compact metadata paths and hashes are recorded there.
The runner checked its fixed input hashes both before and after admission.

## Remaining Necessary Bounds

Post-batch supply receipt:
[UNIQUE-TRAINING-REPAIR-SUPPLY-v1.json](UNIQUE-TRAINING-REPAIR-SUPPLY-v1.json).

| Query / category group | Required | Source-capped upper bound | Shortfall |
|---|---:|---:|---:|
| Direction: all E reasons combined | 600 | 597 | 3 |
| Which: A-D + F + multiple-entailment E | 1,700 | 1,651 | 49 |

Direction's bound is 297 random-world E + at most 200 random-atoms E +
at most 100 targeted E. Existing controlled Direction questions have sole
entailed answers and cannot supply E. The total raw E supply is 680, but
source quotas prevent selecting 600 of them.

Which's bound is 1,400 controlled + (79 sole + 29 multiple) random-world +
(29 sole + 14 multiple) random-atoms + 100 targeted = 1,651.
A-D and F eligibility from the same base is counted once. The desired group
contains 1,400 sole-entailment questions and 300 multiple-entailment questions.

Count passes the checked answer/source/stratum necessary bounds. This is not a
full Count selection result: family, depth, Count-value and selected-explanation
evidence gates still require joint validation if overall supply becomes viable.

The two shortfalls are necessary lower bounds, not a guaranteed sufficient refill
size. The source-bound calculation does not certify that its upper bounds are
jointly attainable with every remaining constraint.

No production MILP was run because necessary supply conditions already fail.
No old candidate was assigned F depth credit from an unseen primary explanation.
No final 6,000-example selection, synthetic export or model-ready package exists
for this allocation yet.

## Verification

- 871 entailment regression tests passed.
- 181 shared training, production, evaluation and context-budget tests passed.
- The final focused repair suite passed 26 tests, including selected F depth,
  no depth for ambiguous/inconsistent E, non-minimal pair rejection, reference
  mutation rejection, one-menu uniqueness and fixed-batch execution guards.
  This suite overlaps some of the larger suites; counts are not additive.
- Ruff and patch whitespace checks passed before the run.
- Original configuration and preserved selected-base inventory match the v4
  receipt. The run's recorded evaluation input hashes remained unchanged.

## Decision Boundary

The single approved supplementary batch is exhausted. Keep the agreed answer
targets and stop generation. Further progress requires an explicit decision
about new supply or source allocation; neither is authorized by this result.
The missing-object rejection rate and scarcity of sole-entailment Which outcomes
should inform that decision, not be hidden by an automatic retry loop.

The active goal remains incomplete. Existing evaluation sets, historical
receipts and the audited external Unique benchmark remain intact.
