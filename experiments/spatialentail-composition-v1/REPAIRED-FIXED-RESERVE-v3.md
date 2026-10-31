# Guarded Fixed-Reserve Results

## Outcome

The completed audit is `REPAIRED-FIXED-RESERVE-v3.json`, using
`all-scene-forest-v2`. It checks repaired controlled recipes, not a final
replacement dataset or combined 25% retention.

The unchanged old recipes had 124 insufficient Count quota cells. Repair,
overlap rejection, and canonical-query deduplication reduce that to seven
cells, all at depth 1.

| Count Outcome | Fixed Slots |
|---|---:|
| Passed verification | 6,986 |
| Rejected for scene overlap | 883 |
| Rejected for original size limits | 4 |
| Total | 7,873 |

After canonical query deduplication, 6,806 verified Count bases remain.
There are no Count depth-family-floor shortages. This does not establish all
combined retained-subset requirements.

| Remaining Count Cell | Required | Verified Unique Supply | Shortfall |
|---|---:|---:|---:|
| Train / depth 1 / count 3 | 22 | 15 | 7 |
| Train / depth 1 / count 4 | 22 | 12 | 10 |
| Dev / depth 1 / count 1 | 7 | 6 | 1 |
| Dev / depth 1 / count 3 | 7 | 5 | 2 |
| Dev / depth 1 / count 4 | 7 | 4 | 3 |
| In-range / depth 1 / count 3 | 14 | 13 | 1 |
| In-range / depth 1 / count 4 | 14 | 4 | 10 |

The cell-specific shortfalls sum to 34, not 34 missing independent families.
Renaming existing scenes would not satisfy independent-base supply.

## Selection and Explanations

All 832 Selection cardinality/depth/direction probes passed. They use existing
fixed seeds at cardinalities 1-4, including requested-depth two-member Unique E
decisions. They are alternative construction probes, not 832 extra fixed bases
or proof of a final 25% allocation.

The maximum measured target lengths in the fully rendered probe subset were
2,933 Natural and 3,298 Symbolic tokens, within the existing stage-1 cap.
Count membership certificates and exact total entailment were independently
replayed; full paired-reference and omission audits were run on the configured
representative subset.

## Isolation

The complete mapped-scene check rejected 1,332 story slots: 1 development
exclusion overlap and 1,331 canonical-scene cross-split overlaps. All slots in
a cross-split overlap group are quarantined rather than moved or relabeled.
The Count table includes 883 of these rejected query slots.

No exclusion list, seed inventory, source quota, size limit, or numeric
coverage minimum was changed. Runtime was 1,176.3 seconds with four workers.

## Approved Unique Depth Revision

The user approved including multiple-entailment Selection scenes in depth-9/10
evaluation. The new diagnostic policy preserves the 400-base Unique Selection
minimum but checks decision-depth support rather than requiring every output
to be substantive A-D.

A singleton still needs substantive support. A Unique E answer for multiple
entailments needs its cheapest pair of positive membership certificates at the
requested depth. This is not permission to label a shallow E decision deep.
The old frozen configuration and scoped datasets remain unchanged. Final 25%
allocation, sealing, and training are not approved by this decision.

## Depth-One Follow-Up

`all-scene-forest-v3` adds seed-controlled acyclic context links at depth 1.
Every context object still has a direct opposite-direction exclusion; removing
the added redundant links preserves the all-scene answer. Higher-depth scene
premises and presentation stay unchanged.

The changed depth-one inventory was fully rendered and independently
re-audited against all canonical scene identities from the completed reserve
receipt. Its fresh receipt is `DEPTH-ONE-RESERVE-REPAIR-v1.json` and its review
is `DEPTH-ONE-RESERVE-REPAIR-v1.md`. It retains 631 distinct verified depth-one
Count bases, with no Count quota shortages and no changes to higher-depth
overlap statuses. All 96 Selection probes passed.
The targeted revised recipe/policy tests passed 182 cases. The preceding full
entailment suite passed 806 cases, and explicit overlap rejection tests passed.

Full combined retention still requires random/targeted all-scene admission,
coverage, source mixture, split isolation, and any proposed gold allocation.
No replacement dataset, sealing, or training has occurred.
