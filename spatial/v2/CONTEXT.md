# SpatialEntail V2 vocabulary

- **Calibration pool**: generated, solver-checked examples used to measure cell
  difficulty before the benchmark matrix is fixed. Calibration scores are not
  frozen benchmark results.
- **Freeze candidate**: the selected cell matrix and allocation after model
  calibration, but before fresh examples pass uniqueness and disjointness
  checks.
- **Frozen benchmark**: fresh examples from the freeze candidate whose manifest,
  canonical structures, allocations, and software inputs are sealed. Frozen
  examples are never used for SFT.
- **Anchor cell**: an intentionally easier cell retained to verify matched
  competence. It is not presented as the main evidence of generalisation.
- **Discriminator cell**: a cell with structural novelty and measured headroom,
  used as the main evidence of composition or shift performance.
- **COMP**: composed reasoning under structures represented in the task design.
- **SHIFT**: held-out structural combinations or depths intended to test
  generalisation beyond matched composition.
- **MULTI-SFT**: training data containing SINGLE and set-valued answer contracts.
  It is generated only after frozen benchmark structures are excluded.
