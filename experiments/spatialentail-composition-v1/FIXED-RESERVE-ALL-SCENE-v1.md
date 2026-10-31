# Fixed Reserve All-Scene Check

## Scope

This check reconstructs the unchanged frozen configuration and exclusion
inventory. No seeds, source slots, or allocation quotas are added or relaxed.
It tests a necessary condition for satisfying all requirements together:
whether the controlled Count reserve can supply the existing exact Count quotas
under the new all-scene semantics.

It does not pretend that the earlier development-only multi-answer probe is
already integrated into the production schedule. The fixed Selection slots use
`matches=1`. Replacing their recipes with the new construction and allocating
multi-answer cardinalities needs an explicit versioned mapping and re-audit.

## Method

Reconstruct the full fixed source schedule and check its original story/family
split isolation and development exclusions. For every controlled Count slot:

1. Expand the query to all nonreference scene objects.
2. Independently construct integer-axis constraints for all premises and
   distinct named points.
3. Check consistency, then test every allowed exact total 0-4 for entailment
   by checking the negated count claim for unsatisfiability. Credit a changed
   verified total to its actual quota, regardless of the old intended match count.
4. Compare maximum possible supply to the frozen Count quota for that
   split/depth/value cell.

This is deliberately an optimistic bound. It ignores later duplicate,
depth-support, explanation, token, and coverage rejections. Inconclusive solver
checks remain unresolved and count optimistically as available supply. Therefore
any shortage is sufficient to establish infeasibility for unchanged recipes.
Enough optimistic supply would not prove final retention feasibility.

A retention MILP is unnecessary if a mandatory Count quota has already failed.
No rejection is disguised as a semantic E/F label, and no old scoped labels
are carried forward.

## Results

The completed receipt is `FIXED-RESERVE-ALL-SCENE-v1.json`.
**The unchanged frozen recipes cannot satisfy the combined requirements under
all-scene scope.** Mandatory controlled Count quotas already have insufficient
optimistic supply, so running retention MILP on these recipes cannot fix them.

The original schedule and family/exclusion preflight passed for 18,290 stories
and 31,251 query slots. After frozen batch index selections, every one of the
7,873 controlled Count slots was checked:

| Split | Checked Slots | Entailed Count in 0-4 |
|---|---:|---:|
| Train | 4,185 | 541 |
| Dev | 832 | 122 |
| In-range test | 1,656 | 245 |
| Extrapolation test | 1,200 | 115 |
| Total | 7,873 | 1,023 |

The other 6,850 slots have no entailed exact total in 0-4. This does not
distinguish a fixed total above 4 from a noninvariant total or inconsistent
premises. No solver checks remained unresolved.

There are 124 Count quota cells with shortages, summing to 1,629 missing
cell-specific bases even before duplicate, depth, evidence, token, or coverage
filtering. For example, train/depth-8/Count-1 requires 53 bases but has at most
3; train/depth-8/Count-3 requires 53 but has at most 6.

Raw batch inventories contain 7,920 Count and 7,920 Selection slots before
index selections. The actual selected Selection inventory is also 7,873 slots,
all with the old `matches=1` parameter. Its raw per-depth family counts do not
fall below the existing floors, but accepted all-scene family supply has not
been established.

The Count scan took 262.3 seconds with four workers, including schedule
reconstruction. The frozen configuration and exclusion hashes were verified
unchanged after completion:

```text
config:     c4beecdb71a6dad1e2cc25a6394848c72213156ff1f898dd61420a9c57f59804
exclusions: a72b6bc765ff3cd8e1cb09a6e6bca5ddd421b5a135bc6ffe1aa07857f79870fb
```

No new seeds, quota relaxation, replacement dataset, sealing, or training
occurred. Original scoped artifacts remain unchanged.

## Remaining Boundary

Separately, a repair-core smoke check using the development scaffold seed 3
passed all 400 combinations of Count values 0-4, depths 1-10 and eight directions.
It recomputed all-scene evidence and support depth. This does not include the
production slot mapping, full family inventory, context decoration, independent
Count replay, or token remeasurement; it is not evidence of retained supply.

The combined new-recipe 25% allocation is not yet tested by this necessary
condition check. It requires the following in order:

- A versioned mapping of existing controlled slots to all-scene Selection and
  Count recipes, retaining the existing seeds and related-family grouping.
- Verification that Count 0-4 targets, positive and negative supports, and
  contract-decision depth work for those recipes.
- Reverified random and targeted supply under all-scene scope.
- Retention of all quotas, source proportions, coverage and family floors
  together, plus split isolation and exclusions.

The bounded 544-case probe establishes constructive G1-G4 support, not any of
these production retention results. This report must not be described as a
completed successful full feasibility check unless those requirements pass.

The next required experiment is a fixed-slot recipe mapping and re-audit,
not reserve refill. The current negative result establishes that unchanged
recipes fail; it does not establish that repaired recipes or the proposed
25% multi-Selection allocation are infeasible.
