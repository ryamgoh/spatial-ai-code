# Slurm Launchers

From the repository root:

```bash
bash experiments/spatialentail-composition-v1/slurm/submit.sh
```

| Script | Role |
|---|---|
| `submit.sh` | Creates fresh run paths and submits exactly three jobs |
| `prepare.sh` | `normal`: generate, audit, tokenize both arms, sync dependencies |
| `train-natural.sh` | `gpu-long`, one `h100-47`, after CPU success |
| `train-symbolic.sh` | `gpu-long`, one `h100-47`, after CPU success |
| `train-common.bash` | Shared credential loading, verification and trainer invocation |

No dataset transfer required. Do not change the checkout while jobs run.
Full operating instructions, W&B/email settings and evaluation limitations:
[Training Runbook](../TRAINING-RUNBOOK.md).

## Saved Adapter Evaluation

Run these individually from the server repository root. H200 jobs use `gpu`
and `gpu:h200-141:1`; they do not retrain. First materialize the fixed internal
benchmark once (CPU only):

```bash
sbatch experiments/spatialentail-composition-v1/slurm/prepare-evaluation.sh
```

After it finishes successfully, evaluate both epochs and select on dev:

```bash
STYLE=natural sbatch --export=ALL experiments/spatialentail-composition-v1/slurm/eval-dev-h200.sh
STYLE=symbolic sbatch --export=ALL experiments/spatialentail-composition-v1/slurm/eval-dev-h200.sh
```

After each arm's dev job succeeds, run its selected adapter for each split:

```bash
STYLE=natural SPLIT=in-range-test sbatch --export=ALL experiments/spatialentail-composition-v1/slurm/eval-selected-h200.sh
```

Repeat for `STYLE=symbolic` and `SPLIT=extrapolation-test` / `external-test`.
For untuned style-matched baselines use `eval-untuned-h200.sh` with the same
STYLE/SPLIT combinations. All jobs report both stages. Results and selected
adapter records live under the training run's `evaluation/` directory.

The evaluator discovers one complete `runs/thinking-*/prepared` automatically.
If several exist, set `PREPARED` to the intended directory; it refuses to guess.
Epochs come from `trainer_state.json`, not checkpoint filenames. Internal data
must exactly reconstruct the 4,800 fixed exclusion query fingerprints.
The external `spatialmap-unique-v1` payload must already exist on the server.
Missing/old-scoped data, changed adapter hashes and existing results are rejected.
Actual H200 inference remains untested locally; no jobs were submitted.
