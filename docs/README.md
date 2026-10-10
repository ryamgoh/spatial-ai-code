# Documentation map

This directory separates current implementation contracts, research plans,
audit records, and informal notes. The documents under `spatialentail/` are the
authoritative technical references for the active V2 system.

## Current SpatialEntail contracts

- [Semantics](spatialentail/semantics.md) defines the formal language, world
  model, query semantics, solver boundary, and completeness limits.
- [Certificate system](spatialentail/certificates.md) defines checked proofs,
  refutations, witnesses, answer certificates, and their assurance boundary.
- [Generation and experiment knobs](spatialentail/generation-and-knobs.md) is
  the canonical catalogue of generator, curriculum, supervision, split, and
  context controls.
- [Trace formats](spatialentail/trace-formats.md) defines the Natural and
  Symbolic training/audit views and the replayable Symbolic grammar.

The runnable bounded solver/data integration fixture is currently located in
[`experiments/15-v2-ablation`](../experiments/15-v2-ablation/README.md).
The package-level code map is [`spatial/v2/README.md`](../spatial/v2/README.md).

## Research

- [Research plan](research/research-plan.md) is the unified FYP canvas.
- [SpatialEval evaluation](research/part-1-evaluation.md) and the
  [SpatialEntail learning study](research/part-2-training.md) contain the
  detailed stream-specific plans.
- [Literature landscape](research/literature-landscape.md) summarizes nearby
  textual-spatial and verified-reasoning work.
- [Frontier assessment](research/frontier-assessment.md) records the calibrated
  novelty analysis.
- [Annotated sources](research/annotated-sources.md) is the citation evidence
  ledger.
- [Explicit world-state supervision review](research/rank-map-supervision-literature.md)
  compares rank maps with ASCII-layout, graph, path, and latent-map precedents.
- [SpatialEntail task taxonomy](research/task-taxonomy.md) defines the three
  query families, three capability lenses, compositional cell identifiers, and
  the dimensions that must be frozen before corpus generation.
- [Capability validation and framework](research/capability-validation-and-framework.md)
  evaluates scope, external task alignment, and the proposed teachable evidence
  procedure. The [SODA taxonomy check](research/soda-task-taxonomy.md) records
  its exact primary-paper tier and task names.

## Audits and notes

- [Remediation audit](audits/critique-remediation.md) records the completed V2
  review findings, fixes, verification, and remaining research limits.
- [SpatialEval world assumptions](audits/spatialeval-world-assumption.md) and
  [Experiments 10–14 evidence](audits/experiments-10-14.md) are evidence audits.
- `notes/` contains explicitly non-authoritative working material. It must not
  be cited as an implementation contract.
- [AIMA citation note](sources/aima-propositional-logic.md) is the retained
  narrow source note; `sources/` is not system documentation.

Experiment-specific historical documents live beside their experiment instead
of at this root. The Typst report and its tooling notes live under
[`fyp-report/`](../fyp-report/README.md).

## Authority rule

When documents disagree, use the current `spatialentail/` contract for software
behaviour, the experiment directory for an executable run, the report for the
submitted narrative, and the audit ledger for verification history. Research
plans and notes describe hypotheses; they do not override implemented behaviour.
