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

## Boundaries

Keep historical results and receipts in
[`spatialentail-randomized-v2`](../spatialentail-randomized-v2/).
Reuse shared solver, tests, and training infrastructure in their existing
locations. The evidence fixtures remain in
[`test_pivot_evidence.py`](../../spatial/v2/tests/test_pivot_evidence.py).

Train, develop, and test on our own story-grouped synthetic splits. SpatialMap
contract variants are additional external evaluation only, never training or
checkpoint-selection data.
