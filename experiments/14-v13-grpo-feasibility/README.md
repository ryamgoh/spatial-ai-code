# Experiment 14 — V13 GRPO feasibility probe

This is the first RL experiment after the successful V13 delta-state SFT
foundation. It is intentionally a feasibility probe, not a claim that GRPO is
already the best training method.

Provisional follow-up ideas and a possible research sequence are recorded in
[`ROUGH-RESEARCH-THOUGHTS.md`](./ROUGH-RESEARCH-THOUGHTS.md).

## Question

Can exact-outcome GRPO improve a frozen hard V13 holdout without materially
damaging the original V13 capabilities?

## Two-H200 layout

One allocation requests two full `h200-141` devices:

```text
GPU 0: GRPO policy training
GPU 1: vLLM rollout generation and pre-training calibration
```

The delta-state 8K QLoRA is first merged into a new BF16 model directory. GRPO
trains a separate rank-64 LoRA on that merged policy and never mutates the SFT
adapter. Axolotl synchronizes the live GRPO LoRA to the vLLM server.

## Data and reward

The deterministic pool contains 680 prompt-only training rows and a disjoint
300-row frozen holdout. Both are disjoint by prompt and premise-world
fingerprint from V13 SFT data and the earlier frozen diagnostics. Difficulty is
sampled across depths/loop lengths 10, 12, and 14, disconnected/query-branch
interference, unequal axes, closed/open controls, and large which/count worlds.

The oracle is produced by the V13 symbolic solver. The primary reward is 1.0
for an exact answer-letter-set match and 0.0 otherwise. A small 0.05 format
reward encourages a parseable `Answer:` line. No subjective reasoning reward
or learned reward model is used.

## Calibration gate

Before training, 48 seeded prompts receive four stochastic SFT-policy rollouts
each on the inference H200. GRPO starts only if:

- rollout pass rate is between 15% and 90%;
- at least 15% of prompt groups contain both correct and incorrect rollouts;
- at least four groups are not uniformly correct.

This gate ensures the within-group reward variance required by GRPO. A failed
gate writes `results/CALIBRATION.json` and exits before any optimizer step.
Run calibration alone with:

```bash
CALIBRATE_ONLY=1 \
sbatch experiments/14-v13-grpo-feasibility/slurm/run-grpo-probe-h200x2.sh
```

## GRPO probe

The first run uses four generations per prompt, two prompt groups per optimizer
step through gradient accumulation, 40 optimizer steps, learning rate `5e-6`,
KL coefficient 0.04, and clipping epsilon 0.2. Submit with:

```bash
sbatch experiments/14-v13-grpo-feasibility/slurm/run-grpo-probe-h200x2.sh
```

The default post-training evaluation compares delta SFT and GRPO on the frozen
300-row holdout. Original V13 and V13.1 retention can be included with:

```bash
SKIP_CALIBRATION=1 SKIP_TRAIN=1 RUN_RETENTION=1 \
sbatch experiments/14-v13-grpo-feasibility/slurm/run-grpo-probe-h200x2.sh
```

The report is `results/SUMMARY.md`. A technically successful probe requires
pre-training reward variance, an improvement on the frozen hard holdout, and no
more than a two-point original-V13 retention drop.

### Queue retention behind the GRPO job

Retention needs only one GPU and can be queued immediately without occupying it
while GRPO runs. Submit it with an `afterok` dependency:

```bash
sbatch --dependency=afterok:<GRPO_JOB_ID> \
  experiments/14-v13-grpo-feasibility/slurm/eval-retention-h200.sh
```

The job becomes eligible only after GRPO exits successfully. It uses one H200,
evaluates original V13 and V13.1, skips completed outputs, and refreshes
`results/SUMMARY.md`. A failed or calibration-aborted GRPO job does not trigger
the dependent retention job.
