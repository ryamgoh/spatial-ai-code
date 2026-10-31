# Three-Job Training Submission

From the repository root on the Slurm server, after pulling the launcher code:

```bash
bash experiments/spatialentail-composition-v1/slurm/submit.sh
```

The wrapper submits exactly three jobs:

1. `prepare.sh`: `normal`, CPU only. Generates 6,000 questions, independently
   audits evidence and selected depths, renders both explanation styles,
   validates allocation and overlap, and prepares both tokenized training sets.
   Then it synchronizes the locked Linux training environment once.
2. `train-natural.sh`: `gpu-long`, `gpu:h100-47:1`, after CPU success.
3. `train-symbolic.sh`: `gpu-long`, `gpu:h100-47:1`, after CPU success.

`h100-47` is the GPU resource type, not the partition name in the repository's
existing cluster configuration. Both GPU jobs use `afterok` and can run
independently. Failed dependencies are cancelled. Submission failures stop the
wrapper, but do not cancel jobs already submitted; their IDs remain in the
printed `runs/thinking-*/jobs.txt` record.

No training-data transfer or archive is needed. `generate_and_prepare.py` uses
the existing controlled generator, fixed seed schedule and quota policy.
`generation-exclusions-v1.json` contains only fixed evaluation fingerprints,
copied byte-for-byte from the audited build's exclusion inventory. Its SHA-256
is checked before generation. All inputs travel with the code; no local
temporary pool, previous candidate payload or external dataset is read.

The bounded preflight permits at most three declared constructions per slot
(18,000 total), followed by exactly 6,000 solver admissions. Any shortage,
failed evidence/prose audit, overlap, quota error or token overflow fails the
CPU job; there is no refill. The historical admission tokenizer gate is
retained using tokenizer files only, followed by independent native thinking
tokenization with Qwen3-4B-Thinking-2507. Training uses only the latter model.
Network access is needed to download tokenizers, dependencies and model weights.

Every submission creates a fresh `runs/thinking-*/prepared` package, avoiding
the tracked `prepared-thinking-v1` directory and its overwrite guard. Run paths
are relative to the repository root and shared by the three jobs. Keep the
checkout and active model settings unchanged until all jobs finish.
Generated questions, evidence and generation receipts remain beside `prepared`
under that run directory. Historical candidates are never overwritten.

Logs appear under `experiments/spatialentail-composition-v1/logs/`. GPU jobs do
not concurrently synchronize the shared environment. They verify the prepared
package, report GPU allocation and run two-epoch QLoRA training. Existing model
output directories are rejected rather than overwritten or implicitly resumed.
Check memory, finite loss and gradient norms during opening steps; no separate
GPU smoke-test job is submitted.

## Monitoring

Monitoring follows randomized-v2. Both SFT arms log online to W&B project
`spatialentail-composition-v1-sft`, with separate run IDs per arm and submission.
The default run group is `spatialentail-composition-v1`; `WANDB_RUN_GROUP` can
override it. The shared randomized-v2 credential loader accepts `WANDB_API_KEY`
from the environment or `${XDG_CONFIG_HOME:-$HOME/.config}/spatialentail/wandb.env`
(or `WANDB_ENV_FILE`). Keep the credential file private, with mode 600.
Missing credentials fail the GPU job before training; there is no silent
offline fallback. Credentials are never stored in generated training configs.

All three jobs send Slurm email to `e0958116@u.nus.edu`. CPU events are
`END,FAIL,TIME_LIMIT`; GPU events are
`BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90`, matching randomized-v2.
Email delivery depends on the cluster's Slurm mail service. W&B connectivity
and email delivery have not been tested from this local machine.

These settings apply to newly generated server-run configs. Historical prepared
packages retain their original configs and audit hashes; do not launch those
old configs expecting the new monitoring settings.

This workflow trains and saves both epoch checkpoints. Fixed-dev evaluation,
checkpoint selection and Stage-2 thinking replay are not launched by these scripts.
No actual Slurm or GPU execution has been tested locally.
