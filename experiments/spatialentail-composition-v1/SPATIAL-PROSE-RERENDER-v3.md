# Spatial-Prose Re-Rendering Audit

Status: passed. Reference-only update; no question generation, sealing or training.

- All 6,000 original records are preserved except references, reference version and token measurements.
- Problems, menus, labels, evidence, IDs, canonical keys, depth metadata and allocation are unchanged.
- Selected paths and counterexamples were replayed; shared-chain coverage and all Which alternatives were checked.
- Current row audits recheck entailment and reference matching using the existing solver. This is not a new dataset selection.
- Both style exports were read back and checked against the verified records.
- All original training and external payload fingerprints are unchanged.

## Token Comparison

| Style | Previous mean | New mean | Previous max | New max |
|---|---:|---:|---:|---:|
| natural | 590.5 | 333.3 | 1653 | 1146 |
| symbolic | 640.9 | 386.8 | 1825 | 1301 |

## Scope

Natural wording distinguishes 'further east/west' (one axis) from exact compass directions. Direct facts stay direct; Count reuses cited chains; ambiguous Direction/Count uses two saved arrangements with different outcomes; Which rejections are grouped without dropping any other object. Contradiction explanations use the same conflict certificate.

The prose remains systematically generated, not independently human-authored. No claim of new structural diversity, new spatial coverage, or GPU/training validation is made.

Candidate: `/Users/bytedance/Code/spatial-ai-code/experiments/spatialentail-composition-v1/unique-training-candidate-v3`
Examples: `SPATIAL-PROSE-RERENDER-v3.examples.md`
