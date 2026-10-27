# SpatialEntail Composition v1

New experiment for atomic spatial reasoning, verified evidence, and matched
Unique Answer, Listed Possibilities, and Exhaustive Possibilities contracts.

## Working Documents

- [Main pivot](pivot_20261026.md): source of truth for research decisions.
- [Evidence examples](research-analysis/PIVOT-EVIDENCE.md): worked examples and
  validation requirements supporting the pivot.
- [Literature note](research-analysis/PIVOT-LITERATURE.md): primary-source
  findings and dated recommendations, not a second experiment specification.

Future generation configs, datasets, training jobs, and results for this study
belong here. This directory currently contains planning and evidence notes,
not a generated dataset or completed training run.

## Execution commitment

The research direction is settled. Implement the pivot rather than reopening
project ideation. Changes to research scope require explicit user approval;
correctness failures must still be surfaced and fixed, never hidden to preserve
the schedule. This section is an execution guardrail, not a second specification.

- Deliver the verified atomic spatial-composition dataset first, covering
  Direction, Selection, and Count across all three answer contracts.
- Use the six-option interface and matched Natural/Symbolic explanations from
  shared verified evidence. Freeze remaining formats and sampling settings
  after feasibility checks and before training.
- Keep story/family splits intact. SpatialMap is external evaluation only;
  preserve Corr SINGLE and version new contract views separately.
- Train two matched mixed-contract adapters with one shared initial seed.
  Evaluate both and the style-matched untuned baselines. Select checkpoints
  using synthetic Stage-1 development accuracy; report both evaluation stages.
- Do not add Boolean-proof baselines, an answer-only arm, broad generalization
  campaigns, extra model sweeps, or automatic data/seed expansion.
- Reuse shared infrastructure and make targeted, regression-checked changes.
  A repository-wide rewrite or solver replacement is not a study deliverable.
- Follow the pivot's three-week boundary and protect the writing week.
  Negative or mixed training results do not invalidate the dataset contribution.

The next implementation milestone is evidence-category feasibility and an
audited generation path with matched renderers and release gates, not bulk
generation or training. The evidence fixtures do not establish production
readiness.

## Boundaries

Keep historical results and receipts in
[`spatialentail-randomized-v2`](../spatialentail-randomized-v2/).
Reuse shared solver, tests, and training infrastructure in their existing
locations. The evidence fixtures remain in
[`test_pivot_evidence.py`](../../spatial/v2/tests/test_pivot_evidence.py).

Train, develop, and test on our own story-grouped synthetic splits. SpatialMap
contract variants are additional external evaluation only, never training or
checkpoint-selection data.
