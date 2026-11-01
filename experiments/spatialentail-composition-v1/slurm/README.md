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

Preparation uses one worker per allocated CPU. For a larger allocation, for example:

```bash
sbatch --cpus-per-task=32 --mem=128G experiments/spatialentail-composition-v1/slurm/prepare-evaluation.sh
```

Each worker loads its own tokenizer and solver; increase memory with workers.
Scaling to 128 workers is supported but has not been benchmarked. Inventory
reconstruction and final output writing remain serial.

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
Missing/old-scoped data and changed adapter hashes are rejected.
Generation batches default to 16; set `EVAL_BATCH_SIZE` to override.
Each stage shows vLLM progress bars in stderr; stdout reports batch progress,
throughput, ETA, and final accuracy/token summaries.

Completed matching results are reused. New partial results resume only with
matching benchmark/model/adapter hashes, runtime versions, and batch settings.
Legacy partial files have no provenance receipt and cannot safely resume.
Cancel affected evaluation jobs before updating the server checkout. Move only
legacy `*.partial.jsonl` files aside (those without a corresponding
`*.resume.json`), then resubmit. Keep completed JSON results, selection files,
adapters, and evaluation data. Do not change batch settings mid-resume.
Batching may change numerical outputs; execution settings and the resumed
example count are recorded. Local tests use a mocked engine; actual GPU
throughput and completion within three hours require server verification.
