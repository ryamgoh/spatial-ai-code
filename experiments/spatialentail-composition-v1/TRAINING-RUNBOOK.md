# Training Runbook

## Submit

Use a fixed checkout on the Slurm server. Do not pull into it while jobs run.
From the repository root:

```bash
bash experiments/spatialentail-composition-v1/slurm/submit.sh
```

This submits generation/auditing/preparation on `normal`, then Natural and
Symbolic training on `gpu-long`, each with `gpu:h100-47:1` and `afterok` on
the CPU job. No data transfer is required. Failed CPU dependencies prevent
training; partial submission IDs remain recorded rather than silently cancelled.

Each submission creates `runs/thinking-*/` containing `jobs.txt`, generated
candidate/evidence and `prepared/` with both configs, tokenizer files and masks.
Old outputs are not overwritten. Slurm logs are under `logs/`.

## Training and Monitoring

- Model: pinned Qwen3-4B-Thinking-2507. The native thinking prefix is masked;
  explanation, closing tag, final answer and turn-end token are supervised.
- Matched two-epoch QLoRA: rank 64, alpha 128, 4-bit loading, BF16 compute,
  learning rate 5e-5, microbatch 1, accumulation 8, seed 42.
- Both epoch checkpoints are saved. No random dev split is created.
- W&B: `spatialentail-composition-v1-sft`, online, separate arm/submission IDs.
  Credentials use `WANDB_API_KEY` or `~/.config/spatialentail/wandb.env`;
  `WANDB_ENV_FILE` and `XDG_CONFIG_HOME` follow the randomized-v2 helper.
- Email: `e0958116@u.nus.edu`; CPU `END,FAIL,TIME_LIMIT`, GPU
  `BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90`.

At startup check actual loaded rows/masks, GPU allocation, memory, finite loss
and optimizer progress. No separate GPU smoke-test job is required.
`logging_nan_inf_filter: false` exposes invalid losses; it is not an automatic
abort mechanism. W&B/email delivery and GPU execution require server evidence.

## Evaluation Status

**Not implemented in this workflow:** epoch dev loss, current-model generated
dev accuracy/checkpoint selection, and Stage-2 thinking replay.
Keep both epoch checkpoints. The agreed selection rule is Unique Stage-1
synthetic dev view accuracy; earliest epoch wins ties. Do not select by train
loss, test results or Stage 2. External data is never training/selection data.

Before future launches, use [the launch checklist](TRAINING-LAUNCH-CHECKLIST.md).
It is a review requirement, not an enforced submission gate. The server workflow
uses fresh run-local preparation; stale local prepared packages were removed.

Detailed launcher behavior: [slurm/README.md](slurm/README.md).
