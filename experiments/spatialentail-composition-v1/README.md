# SpatialEntail Composition v1

Compare matched Natural and Symbolic explanation supervision for atomic
Direction, Which and Exact Count entailment under the Unique answer contract.

## Active Workflow

From the repository root on the Slurm server:

```bash
bash experiments/spatialentail-composition-v1/slurm/submit.sh
```

One `normal` CPU job generates and audits 6,000 questions and prepares both
training arms. Two dependent `gpu-long` jobs each request one `h100-47` GPU.
No dataset transfer is needed.

Both arms use **Qwen/Qwen3-4B-Thinking-2507**, two-epoch QLoRA, matched settings
and seed 42. Per query: 1,200 A-D, 600 E, 200 F. W&B and Slurm email are enabled.
See [the runbook](TRAINING-RUNBOOK.md) for operation and monitoring.

Saved-adapter evaluation is a separate workflow: CPU preparation reconstructs
the fixed internal benchmark, dev jobs select each arm by Stage-1 accuracy, and
final jobs evaluate selected adapters and style-matched untuned baselines.
The batched runner preserves Stage-2 token replay and supports validated resume.
SFT itself does not report epoch dev loss. Local tests do not certify server execution.

## Essential Files

| File | Role |
|---|---|
| [UNIQUE-ONLY-v1.md](UNIQUE-ONLY-v1.md) | Authoritative semantics, allocation and study requirements |
| [active-model.json](active-model.json) | Pinned model/tokenizer and declared readiness |
| [generate_and_prepare.py](generate_and_prepare.py) | Server generation, evidence audits and training preparation |
| [slurm/](slurm/README.md) | Submission and CPU/GPU launchers |
| [TRAINING-LAUNCH-CHECKLIST.md](TRAINING-LAUNCH-CHECKLIST.md) | Future-launch checks, not an automatic gate |
| [unique-training-candidate-v3/](unique-training-candidate-v3/README.md) | Audited local reference dataset |
| [spatialmap-unique-v1/](spatialmap-unique-v1/README.md) | External evaluation only |
| [entailment-candidate-v5/](entailment-candidate-v5/README.md) | Historical nontraining Unique source, not the runnable internal benchmark |
| [unique-only-selection-v1/](unique-only-selection-v1/README.md) | Fixed synthetic evaluation inventory |

Generated server outputs live under `runs/thinking-*/`; logs live under `logs/`.
Superseded repair, reserve, migration, and coverage-runner scripts and their
exclusive tests have been removed. Independent auditors and current runtime
helpers retain their paths. Frozen manifests preserve historical provenance;
they are not rewritten to disguise source changes.
