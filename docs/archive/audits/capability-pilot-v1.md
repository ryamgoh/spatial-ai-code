# SpatialEntail capability-pilot v1 audit

Status: completed feasibility and automated Natural-quality pilot, 2026-10-11.
This does not select a final training size, validate blocked cells, estimate
human acceptability, or report LLM task performance.

## Generation

- Registry: `spatial/v2/benchmark-cells.yaml`
- Registry SHA-256:
  `9def1b8f56e2b1e322c0b37a79e64c6945edb0eb902f2175e114046411659b29`
- Seed: `1902`
- Ready cells: 10
- Blocked cells: 3
- Requested and accepted: 50 per ready cell, 500 total
- Unexpected failed ready cells: 0
- Candidate attempts: 4,179
- Rejected candidates before acceptance: 3,679
- Overall candidate acceptance rate: 11.96%
- Row SHA-256:
  `48d96b726695a942658ddf411a35c6b2a754b9e45c6f2246480f09789973639c`

| Ready cell | Accepted | Rejected proposals | Acceptance | Mean / max whitespace tokens |
|---|---:|---:|---:|---:|
| `DIR-ARC-UNI` | 50 | 0 | 100.00% | 98.7 / 107 |
| `DIR-TRC-UNI` | 50 | 106 | 32.05% | 213.6 / 259 |
| `CNT-AXC-UNI` | 50 | 0 | 100.00% | 888.3 / 946 |
| `DIR-MP-UNI` | 50 | 0 | 100.00% | 140.6 / 155 |
| `SEL-IFF-UNI` | 50 | 0 | 100.00% | 636.8 / 693 |
| `CNT-NEG-UNI` | 50 | 0 | 100.00% | 793.3 / 852 |
| `DIR-DJE-UNI` | 50 | 0 | 100.00% | 140.3 / 149 |
| `SEL-CSR-UNI` | 50 | 0 | 100.00% | 701.0 / 774 |
| `DIR-ARC-ALT` | 50 | 3,506 | 1.41% | 898.9 / 1,135 |
| `SEL-ARC-NOM` | 50 | 67 | 42.74% | 440.0 / 549 |

`DIR-ARC-ALT` is highly selective under the current policy. That is a
generation-efficiency warning, not evidence of model difficulty.

The three blocked cells and their prerequisites are:

- `SEL-ARC-MUL`: distinguish multiple entailed matches from contingent
  Selection membership;
- `CNT-CSR-JCR-UNI`: admit Count examples only when the result depends on
  correlated rather than independently fixed memberships; and
- `SEL-TRC-UNI-NOISE`: implement solver-verified removable Selection
  distractors.

## Automated Natural-quality screen

The packet sampled five examples from each ready cell, 50 total.

- Packet ID: `spatialentail-pilot-v1-quality-v1`
- Packet SHA-256:
  `804c3e14f1a4e96f91322bdf177130c7ec7a5242207a30a2270f61da031522ba`
- Rubric: `spatial/v2/hitl/quality-judge-v1.md`
- Judge 1: `trae` / `DeepSeek-V4-Pro`, isolated context, sequence position 1
- Judge 2: `trae` / `GPT-5.5`, isolated context, sequence position 2
- DeepSeek-V4-Pro: 50 approve, 0 reject
- GPT-5.5: 50 approve, 0 reject
- Routing: 50 accept, 0 hold-out, 0 human appeal

The first DeepSeek receipt had payload-hash transcription errors for two items.
The authoritative loader rejected it; the same judge corrected it, and the
final receipt passed packet, ID, and payload-hash validation. GPT-5.5's receipt
passed validation directly.

The human queue is empty because no item received two rejections. The selective
appeal protocol therefore did not involve the human and cannot estimate overall
human acceptability or false-acceptance rates.

## Local artifacts

Generated artifacts live under the ignored `results/` directory at
`results/spatialentail-capability-pilot/v1/`. They are intentionally not treated
as source files. The retained artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| `pilot.jsonl` | `48d96b726695a942658ddf411a35c6b2a754b9e45c6f2246480f09789973639c` |
| `pilot_manifest.json` | `7b2508949dc5673d90fa2dfe92b64162336493ec3d7d60b05106e5f676819b75` |
| `pilot_audit.json` | `5cc2d371a4b461055cc33416af8a0fdec47de9add12b43df82969b478e041d4b` |
| `quality_packet.json` | `323ed1180e6c7572aecca9d3c7319222c071b6debbb4a53a4dd9ceacac8040d2` |
| `quality_crosswalk.json` | `8d287b16f2096d30138a88e2ebdfcc3c0bffa73d93a492e24e6bd8f52ae21965` |
| `deepseek-v4-pro-judge.json` | `8d1ec5f77f168ce56f04780d18b335d225eaa2ebeb2fb2457b6ab376f3a3aef6` |
| `gpt-5.5-judge.json` | `140b48a80a773fef989716532ecd1c699bea0e303a909c6958ecdac289776529` |
| `quality_routing.json` | `38b013d3964eab568fcea2a82eaf1fbdd3907673e37ca8d65b75ce1c6dca2c26` |
| `human_appeal_queue.json` | `b8b647a8364f47c60379fd4fe5f3ac168f7abfd346c39819f01f4d8fad20f590` |

## Verification commands

```bash
python -m spatial.v2.check_registry spatial/v2/benchmark-cells.yaml --probe \
  --seed 1901 --max-attempts 500
pytest spatial/v2 -q
typst compile fyp-report/report.typ /tmp/spatialentail-report.pdf
```
