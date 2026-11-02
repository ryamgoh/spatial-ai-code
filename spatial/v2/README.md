# Spatial V2

The current experiment is
[SpatialEntail Final v1](../../experiments/spatialentail-final-v1/README.md).
Its [SFT plan](../../experiments/spatialentail-final-v1/SFT-PLAN.md) records the
current overrides to the inherited
[Unique-only specification](../../experiments/spatialentail-composition-v1/UNIQUE-ONLY-v1.md).

## Active Generation

Current entry points:

| Module | Responsibility |
|---|---|
| `entailment_generator.py` | SceneRequest / build_scene interface |
| `entailment_generation.py` | Configured candidate build, export, verification and worker orchestration |
| `entailment_artifact_audit.py` | Independent SMT, counterexample, path, selected-depth and render-plan audits |
| `entailment_scene_protocol.py` | All-scene questions and system rules |
| `entailment_dataset.py`, `entailment_evidence.py` | Verified labels and paired explanations |
| `entailment_thinking_training.py` | Native Qwen3 thinking serialization and training preparation |
| `entailment_evaluation.py` | Batched two-stage inference, dev selection and resumable results |
| `composition_depth.py` | Supporting atomic-axis path depth |

Final-v1's `generate.py` is a thin CLI launcher; its `generation_policy.py`
supplies experiment-specific quotas, identities and paths to the shared
`GenerationWorkflow`. Outputs remain under the experiment's run directory.
Shared generation imports no experiment implementation. Historical audit
entry points re-export the shared auditor for existing callers.

Historical composition generation and internal evaluation preparation use
`experiments/spatialentail-composition-v1/generate_and_prepare.py` and
`prepare_evaluation.py`. Do not use those launchers for final-v1.
The generic `generate_all.py` CLI is not the current study's dataset launcher.
Retired repair, reserve-assembly, and migration diagnostics have been removed.
The old CPU coordinator, timing runner, coverage-draft compiler, and menu
selection/export workflow are also removed. Current allocation owns its answer
classifier; `serialization.py` owns the atomic JSON writer. The experiment's
current generation and evaluation-preparation entry points remain unchanged.

## Shared Infrastructure

Keep `solver.py`, `proofs.py`, the certificate modules, `text.py`, `grading.py`,
and existing renderers for current semantic checks and evidence.
Visible study premises are flat conjunctions of exact spatial relations.
Internal `And`, `Not`, and `Or` remain for entailment, counterexamples, and
joint-count certificates. Implication/equivalence formula types, their parser
and codec support, and modus-ponens/iff-elimination proof rules are removed.
Old text and trace payloads using them are rejected, not converted.
Existing Symbolic NDJSON is an audit grammar,
not automatically the new study's Symbolic supervision format.

Software contracts in [docs/spatialentail](../../docs/spatialentail/) include
historical interfaces. They do not imply that retired formula types remain
supported. Replay old Boolean traces under their original source revision;
frozen artifacts and Corr SINGLE are not rewritten by this cleanup.

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
