# Source Proportion Diagnostic

Status: read-only assessment completed; no quota changes applied.
The user authorized assessing source proportions, not implementing changes or
generating another batch. Answer/subtype targets and controlled-depth allocation
remain fixed.

## Result

Source proportions alone cannot resolve the full training allocation while
retaining 1,400 controlled questions per query. Which has a source-independent
shortage of at least 35 qualifying questions. No MILP is needed to prove this
necessary-condition failure.

| Query | Aggregate diagnostic result |
|---|---|
| Direction | Moving three source slots can remove the aggregate shortage. |
| Which | Still at least 35 short even with unrestricted noncontrolled source proportions. |
| Count | No source change needed for the checked aggregate counts. |

These are aggregate results, not selections satisfying all family, per-depth,
Count-value, negative-evidence and selected-explanation audit requirements.

## Direction: Minimum Three Source Slots

The existing random-world pool has only 297 Direction E questions for 300
source slots. Every controlled Direction question has a sole entailment.
The target needs 600 E questions among the 600 noncontrolled slots, so at least
three slots must leave random-world.

The following arithmetic construction achieves that lower bound:

| Source | Sole entailment | No entailed value | Inconsistent | Total |
|---|---:|---:|---:|---:|
| Controlled | 1,400 | 0 | 0 | 1,400 |
| Random-world | 0 | 297 | 0 | 297 |
| Random-atoms | 0 | 50 | 150 | 200 |
| Targeted | 0 | 103 | 0 | 103 |
| Total | 1,400 | 450 | 150 | 2,000 |

All cell supplies exist in the verified pool. Controlled F eligibility exceeds
200, so splitting the sole-entailment total into 1,200 A-D and 200 F is possible
at the aggregate menu-eligibility level. No claim of per-depth or full joint
feasibility is made. The table is a proposal, not a changed source policy.

## Which: A Source-Independent Obstruction

The target requires 1,400 sole-entailment questions plus 300 multiple-entailment
questions, for 1,700 in that combined group.

The complete noncontrolled pool contains only:

| Source | Sole entailment | Multiple entailments | Combined |
|---|---:|---:|---:|
| Random-world | 79 | 29 | 108 |
| Random-atoms | 29 | 14 | 43 |
| Targeted | 0 | 114 | 114 |
| Total | 108 | 157 | 265 |

Even selecting every one of those questions gives at most
1,400 controlled + 265 noncontrolled = **1,665**, short of 1,700 by **35**.
A-D/F alternatives on one base do not count twice.

Thus adjusting only the random-world/random-atoms/targeted percentages is
insufficient. At least one additional decision would be needed: more qualifying
noncontrolled supply, a change to the controlled cohort size (and its depth
allocation), or a change to the answer/subtype targets. None is authorized here.
If controlled count were the only additional change, 1,435 would be a necessary
lower bound, not a proven sufficient allocation.

## Count: Aggregate Table With No Source Changes

| Source | Sole entailment | No entailed value | Inconsistent | Total |
|---|---:|---:|---:|---:|
| Controlled | 1,400 | 0 | 0 | 1,400 |
| Random-world | 0 | 300 | 0 | 300 |
| Random-atoms | 0 | 50 | 150 | 200 |
| Targeted | 0 | 100 | 0 | 100 |
| Total | 1,400 | 450 | 150 | 2,000 |

Each cell is within the verified supply. This does not replace full Count
selection and proof-depth verification.

## Provenance And Boundary

Source: [post-repair supply receipt](UNIQUE-TRAINING-REPAIR-SUPPLY-v1.json).
SHA-256:
`f72ac05ebedb3e6445ffbce2656d50ef81fcc02b1a489497b5e7347e7e578bcc`.

The diagnostic checked each example table against the receipt's semantic supply,
its column totals against the approved targets, F eligibility, and source-slot
differences. The Which upper bound sums all noncontrolled supply without source
caps, making the impossibility argument independent of their proportions.

No production policy, candidate, evaluation set, manifest or historical receipt
was changed. No generation or selection was run. The overall goal remains
incomplete; the approved supplementary batch remains exhausted.
