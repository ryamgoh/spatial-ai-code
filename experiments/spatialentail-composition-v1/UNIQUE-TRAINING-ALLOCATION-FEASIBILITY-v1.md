# Unique Training Allocation Feasibility

Date: 2026-10-31.
Status: **infeasible under the currently retained requirements**.
The approved 1,200 A-D / 600 E / 200 F per-query target remains the goal,
not a selected dataset. No target, source quota, depth rule or seed was changed.

## Evidence

The diagnostic read all 25,848 verified candidate bases, including 14,050
training candidates, not just the previously selected 6,000 training questions.
It verified the v4 receipt's configuration and candidate SHA-256, unique base
and canonical query IDs, acceptance and independent-audit markers. Unique
projection also checked row protocol and matched, admitted style-token metadata.
It did not rerun spatial solvers or certify new short-question token counts.

Candidate SHA-256:
`a22dfb27f02199842de54e48cf4002c0c29e07de82a2453629df585e72d5e256`.

Machine-readable evidence: [diagnostic receipt](UNIQUE-TRAINING-ALLOCATION-FEASIBILITY-v1.json).
Reproducer: [check_unique_training_allocation.py](check_unique_training_allocation.py).

## Proven Conflicts

| Requirement | Needed | Upper bound | Shortfall |
|---|---:|---:|---:|
| Direction: no entailed value | 450 | 265 | 185 |
| Direction: all E reasons combined | 600 | 431 | 169 |
| Count: no entailed value, with fixed source quotas | 450 | 436 | 14 |
| Count: all E reasons combined, with fixed source quotas | 600 | 552 | 48 |
| Which: A-D + F + multiple-entailment E, with fixed source quotas | 1,700 | 1,610 | 90 |
| Direction: primary substantive-depth coverage | 1,400 | 1,200 | 200 |
| Count: primary substantive-depth coverage | 1,400 | 1,200 | 200 |

These are overlapping necessary bounds, not additive shortages, minimum refill
budgets or claims that the upper bounds are jointly attainable. Any single
violation proves the full unchanged selection impossible. The JSON includes
additional overlapping subset bounds.

### Source Arithmetic

Each query must retain 1,400 controlled, 300 random-world, 200 random-atoms
and 100 targeted questions. Existing controlled Direction/Count candidates
have sole entailments and therefore cannot supply E.

- Direction has 168 + 47 + 50 = 265 no-entailment candidates across random-world,
  random-atoms and targeted sources. Its total E supply is 431 even before any
  further family or depth constraints.
- Count has 498 raw no-entailment candidates, but source caps give
  272 + 64 + min(162, 100) = 436. Combining its two E reasons gives
  272 + min(64 + 116, 200) + min(162 + 50, 100) = 552.
- Which needs 1,400 sole-entailment questions plus 300 multiple-entailment
  questions. Source caps permit at most
  1,400 + (58 + 21) + (23 + 8) + min(114, 100) = 1,610.
  A-D and F eligibility on the same base is counted once in this union.

Changing only menu omission rates cannot create missing semantic cases.
Renaming objects cannot create independent canonical questions or families.

### Depth Rule Conflict

The inherited Direction/Count coverage rules require 1,400 primary rows with
`substantive_depth_support=true`. Admission sets that flag only for A-D,
explicitly excluding E and F. With one selected menu per question and only
1,200 A-D answers, the largest possible qualifying count is therefore 1,200.
This contradiction is independent of how many new candidates are generated.

F can involve an equally deep underlying spatial proof, but the current rule
does not count it. A potential amendment is to measure verified proof depth in
the selected A-D/F explanation, preserving a 1,400 proof-bearing-example
minimum instead of a primary-A-D-only minimum. This requires explicit approval,
implementation, and evidence replay; it has not been applied. An unseen primary
menu must never provide coverage credit to an omission-only training example.

## Method And Limits

For each nonempty subset of answer categories, count distinct eligible bases.
Separately cap their supply by the exact source quotas and depth/semantic
stratum quotas. Compare those upper bounds with the subset's required count.
For inherited coverage minima, require the same selected row to satisfy both
the answer category and coverage predicate; cap its supply by category targets.

The check found contradictions before MILP, so no selection solver was launched.
Passing these bounds would not prove full feasibility: controlled Count-value
quotas, family floors, split isolation and all remaining coverage gates would
still need joint validation on an actual selected set.

Verification: 26 tests passed across the new diagnostic, Unique projection and
retention. Tests include A-D/F overlap counted once, omission rows not borrowing
primary coverage, source-cap handling, E subtype separation and approved totals.

## Decision Needed

Recommendation for discussion: preserve the new answer-learning target, review
the obsolete primary-A-D-only depth gate, and design a bounded source-supply
repair. Present any required source allocation, sampler or seed changes for
approval before execution. A sufficient repair budget is not established here.

The alternative is to keep the pool and inherited rules fixed and reconsider
the answer targets. Even changing the E subtype ratios alone cannot produce
600 Direction E questions from the 431 available.

No new scenes, menus, seeds, selection, prompt/reference export, seal or training
run was produced. The existing internal evaluation selections, external Unique
artifact and historical receipts were not modified. The goal remains incomplete
pending an approved resolution of the conflicting requirements.
