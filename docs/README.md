# Documentation

## Start here

1. [Problem definition](spatialentail/semantics.md): what a question means,
   exact directions, possible answers, and training assumptions.
2. [Generation guide](spatialentail/generation-and-knobs.md): how we build
   problems backward, substitute rules, sample paths, reject proposals, and
   configure experiments. Generation pseudocode is consolidated here.
3. [Research plan](research/research-plan.md): why we are doing this and what
   the two research streams test.

For a short presentation narrative, use the
[professor brief](professor-brief-research.md).

## Technical references

There are five canonical implementation documents; the PDFs are illustrations,
not additional contracts.

| Document | Responsibility |
|---|---|
| [Semantics](spatialentail/semantics.md) | Language, world assumptions, queries, and solver boundary |
| [Generation](spatialentail/generation-and-knobs.md) | Algorithm, rule chaining, sampling, admission, and controls |
| [Correctness](spatialentail/correctness.md) | Soundness/completeness arguments and implementation trust limits |
| [Certificates](spatialentail/certificates.md) | Checked derivations, refutations, witnesses, and answer evidence |
| [Trace formats](spatialentail/trace-formats.md) | Natural supervision and replayable Symbolic audit grammar |

Presentation assets: [generation overview](spatialentail/generation-overview.pdf)
and [rule substitution](spatialentail/rule-substitution.pdf), with editable
[generation](spatialentail/generation-overview.typ) and
[substitution](spatialentail/rule-substitution.typ) Typst sources.

## Research references

- [Evaluation plan](research/part-1-evaluation.md) and
  [training plan](research/part-2-training.md): detailed study design.
- [SpatialMap audit](research/spatialmap-answer-generation-review.md): local
  answerability evidence, menu-conditioned interpretation, and reporting limits.
- [Task taxonomy](research/task-taxonomy.md): query/capability dimensions.
- [Literature landscape](research/literature-landscape.md) and
  [citation ledger](research/annotated-sources.md): research context and sources.

## Supporting material

The [archive index](archive/README.md) retains detailed research investigations,
audit receipts, earlier drafts, and working notes. Archive placement means
supporting material, not necessarily obsolete or invalid evidence. Do not use
it to override the canonical implementation contracts.

Run status and commands belong beside each experiment. Start with
[randomized-v2](../experiments/spatialentail-randomized-v2/README.md);
the [package map](../spatial/v2/README.md) locates executable code. The
[submitted report](../fyp-report/README.md) has its own chapter structure.

## Authority

Use `spatialentail/` for software contracts, experiment artifacts for executed
runs, the submitted report for its narrative, and archived audits for historical
evidence. Research plans describe hypotheses and do not override implemented
behaviour. No supporting evidence was deleted during consolidation.
