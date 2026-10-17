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
- **Semantic length (L)**: the longest dependency path of semantic proof rules
  in the complete replay certificate. Premise loading, assumptions, and proof
  bookkeeping do not increase L. L is not the old relational-chain depth.
- **Distractor count (X)**: the exact number of top-level atomic premises that
  are individually and jointly removable without changing the complete answer.
- **MATCHED**: fresh, structurally disjoint evaluation examples inside the SFT
  ranges L1--L5, X0--X5, and training-seen ordered transitions.
- **One-factor OOD**: evaluation that changes exactly one factor: LENGTH-OOD
  uses L6--L8, NOISE-OOD uses X6--X8, and COMPOSITION-OOD uses exactly one
  held-out ordered transition.
- **VISIBLE-POSSIBLE**: select only possible values displayed in the fixed
  five-option menu. It shares complete semantics and certificates with an
  ALL-POSSIBLE base but may hide semantic possibilities.
