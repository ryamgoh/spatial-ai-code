# Lineage-Preserving All-Scene Repair

## Scope

The repaired recipe is `all-scene-forest-v2`. It uses the same fixed source
slots and seeds, preserves original negative forests, and changes only the
answer-support construction to make all-scene Count 0-4 and Selection 1-4
compatible with the requested support depth.

Original scoped artifacts and the frozen configuration remain untouched.
Renamings are presentation variants, not extra independent bases or families.

## Progress

The revised strict preflight stopped on an excluded development-scene overlap,
recorded in `REPAIRED-FIXED-RESERVE-v2.json`. The subsequent audit records
overlapping fixed slots as rejected rather than moving or replacing them. All
slots in a canonical scene group that crosses splits are quarantined. Evaluation
slots overlapping the frozen development exclusions are also rejected.

The guarded supply audit is running and will be recorded separately in
`REPAIRED-FIXED-RESERVE-v3.json`. No exclusion or quota has been changed.

The first repair was stopped before admission because rebuilding its negative
forest merged some constructions across splits. Its failed receipt remains
`REPAIRED-FIXED-RESERVE-v1.json`. The revised recipe preserves those forests:
the direct selected-seed anchor inventory contains 1,707 distinct repaired
anchors for 1,707 seeds and no cross-split anchor groups. This preliminary
inventory check does not substitute for complete mapped-story preflight.

The revised semantics and production checks passed 163 tests. Seven focused
tests passed for fixed-slot mapping, preservation of original negative forests,
split grouping, unchanged bounds, and the existing Unique policy conflict.
The revised full entailment regression suite passed 806 tests. The five
focused audit tests, including explicit overlap rejection, also passed after
the subsequent guard addition. Compatibility checks for old and new production,
scene sampling, and depth passed 284 tests.

A bounds-only scan found 7,869 of 7,873 Count slots within the original limits,
with no quota shortfall at that optimistic stage. It is not verified,
deduplicated, overlap-filtered Count supply.

## Method

- Reconstruct and preflight the unchanged original schedule and exclusions.
- Map existing controlled slots to repaired recipes with their original seeds.
- Preserve Count targets and query-slot identities; do not add reserve slots.
- Merge any related original/repaired anchors and check split isolation again.
- Check the complete mapped schedule for canonical story overlap and exclusions.
- Check every mapped Count slot for bounds, exact entailment, support depth,
  context invariance, independent SMT entailment, and membership proof replay.
- Deduplicate canonical Count queries before comparing supply and family floors.
- Fully render and independently audit one Count example per
  split/depth/direction/value, including forced omissions and token limits.
- Probe Selection cardinalities 1-4 on existing source seeds. These alternatives
  are not four new retained bases and do not establish a final 25% allocation.

Any oversized construction is rejected under the original scene limits.
No bounds or coverage requirements are silently relaxed.

## Retention Boundary

The old extrapolation requirement asks for 400 substantive primary Unique
Selection answers out of 400 Selection bases. Applying 25% multi-answer
coverage within this split would make 100 of them correctly answer E under
Unique, leaving at most 300 substantive answers. These requirements cannot
both hold unchanged.

This is an illustrative per-split calculation, not an approved per-split
allocation. The proposed overall 25% does not require 25% in every split or
depth. An overall allocation could instead leave extrapolation singleton-only
and place its multi-answer examples elsewhere. No such allocation has been
approved or tested.

The proposed resolution is to require requested-depth support for all Unique
decisions, including the cheapest two-member certificate that justifies E.
Singleton answers retain substantive-depth support. This change has not been
applied to the frozen policy.

Full combined retention remains untested: random and targeted all-scene
admission, the policy conflict, and the proposed 25% retained-base allocation
must be resolved and checked. No replacement dataset, sealing, or training
has occurred.
