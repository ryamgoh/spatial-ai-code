# SpatialEntail capability-pilot v3 audit

Status: completed feasibility and automated Natural-quality pilot, 2026-10-11.
This supersedes v2 for current registry readiness and generation yield. It does
not select a final training size, estimate human acceptability, or report LLM
task performance.

## Generation receipt

- Registry: `spatial/v2/benchmark-cells.yaml`
- Registry SHA-256:
  `c4168727a1b4d00f5df3073fc650267648eb2ca6aac8af4fb7f19edd3865dab4`
- Seed: `1911`
- Ready cells: 13
- Blocked cells: 0
- Requested and accepted: 50 per cell, 650 total
- Unexpected failed cells: 0
- Candidate attempts: 865
- Rejected candidates before acceptance: 215
- Overall candidate acceptance rate: 75.14%
- Row SHA-256:
  `78161bfc3a16d7cfd7fce1ce843856bab97921b2d20f66567a224e913f9fa7ce`

| Cell | Accepted | Rejected proposals | Acceptance | Mean / max whitespace tokens |
|---|---:|---:|---:|---:|
| `DIR-ARC-UNI` | 50 | 0 | 100.00% | 99.1 / 107 |
| `DIR-TRC-UNI` | 50 | 129 | 27.93% | 209.8 / 245 |
| `CNT-AXC-UNI` | 50 | 0 | 100.00% | 885.6 / 965 |
| `DIR-MP-UNI` | 50 | 0 | 100.00% | 140.2 / 149 |
| `SEL-IFF-UNI` | 50 | 0 | 100.00% | 632.6 / 693 |
| `CNT-NEG-UNI` | 50 | 0 | 100.00% | 780.3 / 830 |
| `DIR-DJE-UNI` | 50 | 0 | 100.00% | 139.5 / 153 |
| `SEL-CSR-UNI` | 50 | 0 | 100.00% | 704.8 / 775 |
| `DIR-ARC-ALT` | 50 | 0 | 100.00% | 887.2 / 971 |
| `SEL-ARC-NOM` | 50 | 85 | 37.04% | 457.0 / 561 |
| `SEL-ARC-MUL` | 50 | 1 | 98.04% | 529.8 / 607 |
| `CNT-CSR-JCR-UNI` | 50 | 0 | 100.00% | 903.5 / 999 |
| `SEL-TRC-UNI-NOISE` | 50 | 0 | 100.00% | 870.3 / 953 |

`DIR-ARC-ALT` now uses a constructive atomic template instead of waiting for a
random coordinate world to produce exactly two possible directions. Two query
objects are constrained equal on one axis through a shared pivot, while their
order on the other axis remains open. The resulting answer set is exactly
North/South or East/West after solver validation. All 50 rows were accepted on
their first attempt, compared with 4,718 rejected proposals in v2. Only this
cell's 50 visible rows changed between the v2 and v3 pilots.

## Automated Natural-quality screen

- Packet: 65 items, five per registry cell
- Packet ID: `spatialentail-pilot-v3-quality-v1`
- Packet fingerprint:
  `2144cbdf19c4da49978bab196e13847c005098ec64eb9a2bb15bbf7a237f67d1`
- `DeepSeek-V4-Pro`: 65 approve, 0 reject
- `GPT-5.5`: 65 approve, 0 reject
- Routing: 65 accept, 0 hold-out, 0 human appeal

The two terminal-agent judges ran sequentially in isolated contexts over the
same refreshed packet. Both receipts passed model, sequence, packet, item-ID,
payload-hash, and rubric-field validation. No human menu was shown because the
unanimous-rejection queue was empty. The selective appeal design cannot
estimate overall human acceptability or false-acceptance rates.

## Local artifacts

Raw generated artifacts remain under the ignored results directory at
`results/spatialentail-capability-pilot/v3/`. The retained hashes are:

| Artifact | SHA-256 |
|---|---|
| `pilot.jsonl` | `78161bfc3a16d7cfd7fce1ce843856bab97921b2d20f66567a224e913f9fa7ce` |
| `pilot_manifest.json` | `9e1ea997aff9440f3deb0e1ee2a251fbf4887a85387dd82ca71cfba2cd8f7ed8` |
| `pilot_audit.json` | `b3b1da593eeee5e1124fcc35e1190425605d16ca483f1f562f593943036bc841` |
| `quality_packet.json` | `311dc22d02311de6906e903bf815afeb97cd2b6d592e528d77874eb684d8d952` |
| `quality_crosswalk.json` | `c6a8cf625e79c2734b9d21a9a87f2300e81fec0bdd0bb8da143b9a7bb481a67b` |
| `deepseek-v4-pro-judge.json` | `f411af30630b2b8e91d591ec980088c24d2ed45d4cb5b0314d868d103ac2e625` |
| `gpt-5.5-judge.json` | `e6c05117765ae1a671b07be9040771e80565904f9a07606d1ecb2aeb250783af` |
| `quality_routing.json` | `1e56e65cb22b70c9fa3c57feb84db44123db158259e32f37ebb1fb10af3ac620` |
| `human_appeal_queue.json` | `901d2247528ccc5ab2ce00b556939b32528888552ec02ab9c1b28e574f063319` |

## Limits

- The registry contains representative feasibility cells, not the final
  reporting allocation.
- The two model judges evaluated Natural presentation quality, not benchmark
  difficulty or model answer accuracy.
- The human appeal branch was executable but not exercised because there were
  no unanimous rejections.
- No SFT, learning curve, final-size selection, or external transfer ran.
