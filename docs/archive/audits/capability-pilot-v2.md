# SpatialEntail capability-pilot v2 audit

Status: completed feasibility and automated Natural-quality pilot, 2026-10-11.
This superseded v1 and is retained as the rejection-sampling baseline for the
current [v3 audit](capability-pilot-v3.md). It does not select a final training
size, estimate human acceptability, or report LLM task performance.

## Generation receipt

- Registry: `spatial/v2/benchmark-cells.yaml`
- Registry SHA-256:
  `c4168727a1b4d00f5df3073fc650267648eb2ca6aac8af4fb7f19edd3865dab4`
- Seed: `1911`
- Ready cells: 13
- Blocked cells: 0
- Requested and accepted: 50 per cell, 650 total
- Unexpected failed cells: 0
- Candidate attempts: 5,583
- Rejected candidates before acceptance: 4,933
- Overall candidate acceptance rate: 11.64%
- Row SHA-256:
  `6f8ca46968c3f024424399132dde513612be1327745947feb3c55496a6fe20fd`

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
| `DIR-ARC-ALT` | 50 | 4,718 | 1.05% | 895.5 / 1,063 |
| `SEL-ARC-NOM` | 50 | 85 | 37.04% | 457.0 / 561 |
| `SEL-ARC-MUL` | 50 | 1 | 98.04% | 529.8 / 607 |
| `CNT-CSR-JCR-UNI` | 50 | 0 | 100.00% | 903.5 / 999 |
| `SEL-TRC-UNI-NOISE` | 50 | 0 | 100.00% | 870.3 / 953 |

The very low `DIR-ARC-ALT` yield is a generator-efficiency warning and must be
addressed before scaling. It is not evidence of model difficulty.

## Automated Natural-quality screen

- Packet: 65 items, five per registry cell
- Packet ID: `spatialentail-pilot-v2-quality-v1`
- Packet fingerprint:
  `81fdf088b1555c8dd9bcb2bf53334686d6105e4dadfcdfe5303da52e8bc39612`
- `DeepSeek-V4-Pro`: 65 approve, 0 reject
- `GPT-5.5`: 65 approve, 0 reject
- Routing: 65 accept, 0 hold-out, 0 human appeal

Both receipts passed model, sequence, packet, item-ID, payload-hash, and rubric-
field validation. No human menu was shown because the unanimous-rejection queue
was empty. The selective appeal design cannot estimate overall human
acceptability or false-acceptance rates.

## Local artifacts

Raw generated artifacts remain under the ignored results directory at
`results/spatialentail-capability-pilot/v2/`. The retained hashes are:

| Artifact | SHA-256 |
|---|---|
| `pilot.jsonl` | `6f8ca46968c3f024424399132dde513612be1327745947feb3c55496a6fe20fd` |
| `pilot_manifest.json` | `a56b2f8bc94fe1627e3bdf1f96e5e9a133a423ece0bb7fdb9fbd96632874f7fc` |
| `pilot_audit.json` | `d4f399189aa471ba6e8329faa6defc62612f03fb1892f788e74f8ea88ebfc3dd` |
| `quality_packet.json` | `ce29335ce8b1e4abbd879322b47e755e46780718ec8f50fd2e434022b1b0759d` |
| `quality_crosswalk.json` | `3e4eb8c94d477cf5e4441ba8165ed0966fb4ef574d73994c812cba96906819d3` |
| `deepseek-v4-pro-judge.json` | `ee25ff62a71103bc41d317d87d7f98780907a2d0bd12397b2dc4cd7b0705efd8` |
| `gpt-5.5-judge.json` | `8ed8ca16b1d8c368011a4743f6ca49a7b8c78170bc40697704c8464fab470026` |
| `quality_routing.json` | `c523e2b7e53a4f604d4c1393bd3ebf79e4e1f64c03d066e92b1cb3f9ee6262fa` |
| `human_appeal_queue.json` | `feeee7831376bac3663f11f07a1d9abff1a13d6a5eece866c848ba53ceab2ebc` |

## Limits

- The registry contains representative feasibility cells, not the final
  reporting allocation.
- The two model judges evaluated Natural presentation quality, not benchmark
  difficulty or model answer accuracy.
- The human appeal branch was executable but not exercised because there were
  no unanimous rejections.
- No SFT, learning curve, final-size selection, or external transfer ran.
