# SpatialEntail vocabulary

## Current study

Follow the
[composition pivot](../../experiments/spatialentail-composition-v1/pivot_20261026.md)
for all new research work. This package contains reusable infrastructure and
historical workflows; package version `v2` does not select randomized-v2 as the
active experiment.

The vocabulary below is historical. In the new study, use chain length,
verified distractor count, and semantic case, not redefined `L/X/G` or the
MATCHED/OOD campaign. Natural and Symbolic are required matched supervision
arms, rendered from the same verified evidence. Shared Boolean solver support
does not imply Boolean premises belong in the new dataset.

## Dataset lifecycle

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
- **MULTI-SFT**: training data containing SINGLE and set-valued answer contracts.
  It is generated only after frozen benchmark structures are excluded.

## Problem semantics

- **Query family**: what is predicted: Direction (`DIR`), Selection (`SEL`), or
  Count (`CNT`).
- **Answer contract**: how answers are returned: one invariant answer
  (`SINGLE`), the complete possibility set (`ALL-POSSIBLE`), or the possible
  values displayed in the menu (`VISIBLE-POSSIBLE`).
- **Gold-set size (G)**: the number of correct displayed answers. G1 belongs to
  the SINGLE study; G2--G4 belong to the first ALL-POSSIBLE study.
- **Semantic outcome**: a property of the all-model answer set, such as unique
  (`UNI`), alternatives across worlds (`ALT`), multiple certain Selection
  matches (`MUL`), or no match (`NOM`). It is not a reasoning operation.
- **Evidence obligation**: what an audit certificate must establish, such as a
  positive proof, possibility witness, candidate exclusion, or correlated
  joint-Count result (`JCR`). It is not a reasoning operation.

## Proof structure

- **Linear transformation**: one principal derived fact feeds the next stage.
  Randomized v2 samples five types: modus ponens (`MP`), equivalence
  elimination (`IFF`), disjunctive elimination (`DJE`), double-negation
  elimination (`DN`), and transitive relational composition (`TRC`).
- **Case-split reasoning (`CSR`)**: a branching proof motif that opens two or
  more cases, proves a result within each branch, and merges the branch results.
  CSR is measured in replayed certificates but is not a member of the linear
  transformation alphabet.
- **Atomic relation comprehension (`ARC`)**: direct use of a visible relation;
  this is an input structure rather than a derived transformation.
- **Axis composition (`AXC`)**: a spatial proof motif that combines axis-level
  facts into a direction; it is not a sampled linear-transition label in v2.
- **Formula connective**: visible syntax such as `AND`, `OR`, `NOT`, implication,
  or equivalence. A connective is not automatically an inference step. In
  particular, visible `NOT` and legacy `NEG` must not be confused with an
  actual `DN` proof step.

## Complexity and evaluation

- **Semantic length (L)**: the longest dependency path of semantic proof rules
  in the complete replay certificate. Premise loading, assumptions, and proof
  bookkeeping do not increase L. CSR contributes when it occurs in a realised
  certificate. L is not the old relational-chain depth.
- **Distractor count (X)**: the exact number of top-level atomic premises that
  are individually and jointly removable without changing the complete answer.
- **Proof size**: the total number of semantic inference nodes, including work
  performed in parallel branches. Two proofs can have the same L but different
  proof sizes.
- **Branch metrics**: whether CSR occurs, how many branches it opens, and the
  depth of branch-local reasoning. These are separate from the linear rule
  signature.
- **MATCHED**: fresh, structurally disjoint evaluation examples inside the SFT
  L/X ranges and training-seen ordered transitions recorded by the frozen study
  manifest.
- **One-factor OOD**: evaluation that changes exactly one factor beyond the
  frozen training boundary: semantic length, distractor count, or one held-out
  ordered linear transition.

## Legacy v1 reporting

- **CORE**: the fixed primary SINGLE-v1 cells.
- **COMP**: the fixed matched-composition SINGLE-v1 cells.
- **SHIFT**: held-out composition and relational-depth SINGLE-v1 cells.

CORE/COMP/SHIFT remain valid historical v1 labels. They are not reused by the
randomized v2 MATCHED/OOD design.
