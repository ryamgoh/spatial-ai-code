# Spatial V2

The active research direction is
[Composition v1](../../experiments/spatialentail-composition-v1/README.md),
governed by its [Unique-only specification](../../experiments/spatialentail-composition-v1/UNIQUE-ONLY-v1.md).

## Active Generation

Current entry points:

| Module | Responsibility |
|---|---|
| `entailment_generator.py` | SceneRequest / build_scene interface |
| `entailment_scene_protocol.py` | All-scene questions and system rules |
| `entailment_dataset.py`, `entailment_evidence.py` | Verified labels and paired explanations |
| `entailment_thinking_training.py` | Native Qwen3 thinking serialization and training preparation |
| `entailment_evaluation.py` | Batched two-stage inference, dev selection and resumable results |
| `composition_depth.py` | Supporting atomic-axis path depth |

Server generation and internal evaluation preparation are driven by
`experiments/spatialentail-composition-v1/generate_and_prepare.py` and
`prepare_evaluation.py`. Use that experiment's Slurm instructions.
The generic `generate_all.py` CLI is not the current study's dataset launcher.
Retired repair, reserve-assembly, and migration diagnostics have been removed.

## Shared Infrastructure

Keep `solver.py`, `proofs.py`, the certificate modules, `text.py`, `grading.py`,
and existing renderers available to shared and historical callers.
Internal logical operations support semantic checks and evidence even when
visible premises are atomic. Existing Symbolic NDJSON is an audit grammar,
not automatically the new study's Symbolic supervision format.

Software contracts in [docs/spatialentail](../../docs/spatialentail/) describe
existing behavior. Version the new interface without changing frozen historical
contracts or Corr SINGLE.

## Historical Workflows

The legacy Boolean/random-proof generators and benchmark/matrix runners have
been removed with their exclusive launchers and tests. Frozen historical YAML
files retain their paths for provenance; they are not runnable current workloads.
`study/` now retains only historical result-analysis and token-audit utilities.
Its dataset/job orchestration and the HITL package have been retired; original
source remains available in Git at `a85c6c7`. Saved results and receipts remain intact.

```sh
python -m spatial.v2.generate_all --help
python -m pytest spatial/v2 -q
```

Do not import `spatial.v1` into the active generation path.
Historical reproduction requires its original source checkout, not this package.
