# Documentation

## Start here

For the active study, start with
[SpatialEntail Composition v1](../experiments/spatialentail-composition-v1/README.md)
and its [authoritative pivot](../experiments/spatialentail-composition-v1/pivot_20261026.md).
The pivot supersedes earlier research plans, training arms, benchmark campaigns,
and prompt contracts for new-study work. It is a specification, not evidence
that those changes have already been implemented.

For implementation entry points, use the
[package map](../spatial/v2/README.md). The
[report draft](../fyp-report/README.md) follows the same composition direction
and explicitly identifies pending implementation and empirical results.

## Technical references

These documents describe existing shared software. They are not specifications
for the pending composition release; reuse applicable implementation contracts
without importing retired research scope.

| Document | Responsibility |
|---|---|
| [Semantics](spatialentail/semantics.md) | Language, world assumptions, queries, and solver boundary |
| [Correctness](spatialentail/correctness.md) | Soundness/completeness arguments and implementation trust limits |
| [Certificates](spatialentail/certificates.md) | Checked derivations, refutations, witnesses, and answer evidence |
| [Trace formats](spatialentail/trace-formats.md) | Natural supervision and replayable Symbolic audit grammar |

## Research references

- [SpatialMap audit](research/spatialmap-answer-generation-review.md): local
  answerability evidence, menu-conditioned interpretation, and reporting limits.
- [Literature landscape](research/literature-landscape.md) and
  [citation ledger](research/annotated-sources.md): research context and sources.

## Supporting material

The [archive index](archive/README.md) retains detailed research investigations,
audit receipts, superseded research plans, old presentations, and working notes.
The retired task taxonomy and Boolean/randomized generation guide are also
archived for historical reproduction. Neither those plans nor the old
presentation diagrams govern the active study.
Other archived evidence may remain applicable under its stated assumptions.

Run status and commands belong beside each experiment. Start with
[composition-v1](../experiments/spatialentail-composition-v1/README.md);
[randomized-v2](../experiments/spatialentail-randomized-v2/README.md) retains
historical run records.

## Authority

Use shared implementation references for software contracts, experiment artifacts
for executed runs, the report draft for its narrative, and archived audits for historical
evidence. The composition pivot governs intended new-study behavior; existing
software contracts describe current behavior until explicitly migrated and
tested. Preserve legacy behavior for historical callers. Research plans and
documentation edits do not establish implementation completion.
Pruned presentation sources are preserved in the archive, not discarded.
