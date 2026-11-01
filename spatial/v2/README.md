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

The [legacy namespace](legacy/README.md) contains the retired Boolean
curriculum, mixed-rule compositions, randomized proofs, possibility skeletons,
and benchmark/matrix runners. Historical `study/` workflows and tests import
that namespace. The old benchmark YAML files retain their paths.

```sh
python -m spatial.v2.generate_all --help
python -m spatial.v2.legacy.check_registry spatial/v2/benchmark-cells.yaml --probe
python -m pytest spatial/v2 -q
```

Do not import `spatial.v1` or `spatial.v2.legacy` into the active generation path.
Historical reproduction uses those packages explicitly.
