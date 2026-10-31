# Replacement Training Audit

Status: passed. Unsealed; no model training started.

6,000 distinct canonical questions, with one selected menu and matched Natural/Symbolic supervision.

| Query | A-D | E | F |
|---|---:|---:|---:|
| count | 1200 | 600 | 200 |
| direction | 1200 | 600 | 200 |
| which | 1200 | 600 | 200 |

All preserved depth, Count-value, family and evidence minima passed. Fixed evaluation queries, scenes and anchors were excluded. Existing payload hashes are unchanged.

Maximum target tokens: {'natural': 1653, 'symbolic': 1825}.

## Scene Length Check

Descriptive E shares, not a model prediction experiment. The previous shortcut used the 13-premise boundary.

| Query | Premises | E share |
|---|---|---:|
| count | at-most-13 | 30.0% |
| count | over-13 | 30.0% |
| direction | at-most-13 | 30.0% |
| direction | over-13 | 30.0% |
| which | at-most-13 | 26.4% |
| which | over-13 | 30.4% |

## Limits

- The construction recipes are controlled and restricted; this audit does not establish absence of all structural shortcuts.
- Reference cleanup preserves proof obligations; this is not a claim that every explanation reads like independently written human prose.
- The training payload is exported as chat JSONL. No GPU loading, optimizer step, training smoke test or SFT run was performed.

Independent SMT/path/counterexample replay was performed for every admitted question in the build. This artifact audit verifies exact exported equality and provenance; it does not rerun SMT.

Candidate: `/Users/bytedance/Code/spatial-ai-code/experiments/spatialentail-composition-v1/unique-training-candidate-v2`
