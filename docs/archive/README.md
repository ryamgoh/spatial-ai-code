# Supporting documentation archive

Return to the [documentation entry point](../README.md) for current contracts.

This archive reduces the active reading list without deleting research evidence,
audit receipts, or drafts. "Archived" means supporting material, not necessarily
obsolete. Original claims, dates, and limitations remain in each document;
relative links were rebased when files moved.

## Superseded plans and presentations

These documents are explicitly retired by the
[composition pivot](../../experiments/spatialentail-composition-v1/pivot_20261026.md),
not alternative plans for new work:

- [Research canvas](research/research-plan.md),
  [evaluation plan](research/part-1-evaluation.md), and
  [training plan](research/part-2-training.md).
- [Professor brief research notes](presentations/professor-brief-research.md).
- Generation overview: [PDF](presentations/generation-overview.pdf) and
  [Typst source](presentations/generation-overview.typ).
- Rule substitution: [PDF](presentations/rule-substitution.pdf) and
  [Typst source](presentations/rule-substitution.typ).
- [Historical task taxonomy](research/task-taxonomy.md) and
  [generation controls](research/generation-and-knobs.md), retained for the
  retired Boolean and randomized curricula.

The [pruned-materials snapshot](pruned-materials-20261028.tar.gz) preserves
all 16 files exactly as they were immediately before this pruning pass.
It includes the eight removed `fyp-report/` presentation files: the professor
brief, four figure sources, trace JSON, trace exporter, and diagram-package
notes. It also preserves the eight relocated documents and diagram assets
before relative-link rebasing and archive notices.

Snapshot SHA-256:
`4b3ba92fdd5299683a1c3f32162b9958595385a6c21922f7064f452cabf3f895`.
Paths inside the snapshot are relative to the repository root. Extract into
a separate directory; do not overlay the active working tree.
The [pre-composition report archive](../../experiments/archive/precomposition-report-2372c13.tar.gz)
retains the report chapters cited by the historical presentations. Old
line-number citations describe those historical sources, not today's draft.

The [final-pruning snapshot](final-pruning-20261028.tar.gz) preserves the two
guides above before their relocation and the retired `finetune/convert_json.py`
utility. That converter embeds coordinates in old-format training targets and
has no remaining in-repository callers; it is not part of the composition
pipeline. Its source remains recoverable here rather than in active `finetune/`.
Snapshot SHA-256:
`0672803b770a403040c3304b2fd7ae92a6503df0bb927e56e28a3c2c021f75dd`.
This snapshot also uses repository-relative paths and should be extracted
separately, not over the active working tree.

## Where consolidated material lives

| Earlier material | Consolidated reference |
|---|---|
| Detailed generation pseudocode and LaTeX algorithms | [Historical generation guide](research/generation-and-knobs.md#generation-algorithm) |
| SpatialMap alignment follow-up | [SpatialMap audit](../research/spatialmap-answer-generation-review.md#follow-up-options-can-be-part-of-the-task-contract) |
| Numerous capability and task reviews | [Historical task taxonomy](research/task-taxonomy.md) |
| Individual literature investigations | [Literature landscape](../research/literature-landscape.md) and [citation ledger](../research/annotated-sources.md) |
| Earlier work-in-progress research notes | [Superseded research plan](research/research-plan.md) |

## Study and audit evidence

- [Randomized-v2 design](audits/randomized-proof-v2-design.md),
  [SFT preparation](audits/sft-job-preparation.md), and
  [study implementation review](audits/study-review-implementation.md).
- Capability pilots [v1](audits/capability-pilot-v1.md),
  [v2](audits/capability-pilot-v2.md), and [v3](audits/capability-pilot-v3.md).
- [Remediation](audits/critique-remediation.md),
  [certificate-search bottleneck](audits/certificate-search-bottleneck.md), and
  [all-possible benchmark draft](audits/all-possible-bench-draft.md).
- [Experiments 10-14](audits/experiments-10-14.md) and
  [SpatialEval world assumptions](audits/spatialeval-world-assumption.md).

## Focused research

- [Capability validation](research/capability-validation-and-framework.md),
  [framework literature](research/capability-framework-literature-review.md),
  and [SODA taxonomy](research/soda-task-taxonomy.md).
- [External human capabilities](research/external-human-capability-review.md),
  [external relational capabilities](research/external-relational-capability-review.md),
  and [propositional coverage](research/spatialentail-propositional-coverage.md).
- [Generation provenance](research/generation-provenance-validity.md),
  [randomized pipeline review](research/randomized-benchmark-pipeline-review.md),
  and [world-state supervision](research/rank-map-supervision-literature.md).
- [Human review workflow](research/human-review-workflow.md) and
  [data-size freeze protocol](research/size-freeze-protocol.md).
- [SpatialMap contract alignment](research/spatialmap-contract-alignment.md).
- [Frontier assessment](research/frontier-assessment.md),
  [model-evaluation landscape](research/model-evaluation-landscape-2026.md), and
  [spatial-specialist models](research/spatial-specialist-models-2026.md).

## Consolidated generation references

The superseded standalone pseudocode and detailed LaTeX algorithms were
removed after consolidation. Use the [generation guide](research/generation-and-knobs.md)
and its linked archived Typst/PDF presentation assets only for historical
curriculum reproduction, not for the composition release.

## Working notes and source notes

- [Dynamic research report](notes/dynamic-research-report.md),
  [experiment progress](notes/experiments-10-14-progress.md), and
  [raw contribution notes](notes/raw-contribution-notes.md).
- [AIMA propositional-logic citation note](sources/aima-propositional-logic.md).

## Maintenance

Update an existing canonical document rather than creating a new overlapping
guide. Add a focused investigation here only when its source evidence or audit
provenance needs to remain separately inspectable. Keep live run status beside
the corresponding experiment, not in a new root-level progress document.
