# Experiment 15: Spatial V2 trace ablation

This experiment submits independent Slurm jobs for one generated data matrix,
two SFT arms, per-arm evaluation, and a final partial-failure summary.

The checked-in pilot compares:

- `single-natural`: `SINGLE`, Natural CoT, Delta state;
- `single-symbolic`: `SINGLE`, Symbolic CoS, Delta state.

Both arms use the same solver-generated base problems and train/test split.

## Local development machine

The local machine does not need Slurm or a GPU. Validate the complete graph
without writing a run directory or calling `sbatch`:

```bash
uv run --no-project --with typer --with pyyaml \
  python experiments/15-v2-ablation/submit.py --dry-run
```

Run local tests with:

```bash
uv run --python 3.12 --no-project --with pytest --with typer --with pyyaml \
  pytest experiments/15-v2-ablation/test_submit.py experiments/tasks/test_utils_v2.py -q
```

## Slurm login server

From the repository root:

```bash
uv run --no-project --with typer --with pyyaml \
  python experiments/15-v2-ablation/submit.py
```

The submitter creates this graph and exits:

```text
generate
├── train single-natural ──► eval single-natural
└── train single-symbolic ─► eval single-symbolic

summarize waits afterany on all jobs
```

Useful controls:

```bash
# Preview only
uv run --no-project --with typer --with pyyaml \
  python experiments/15-v2-ablation/submit.py --dry-run

# Submit one arm
uv run --no-project --with typer --with pyyaml \
  python experiments/15-v2-ablation/submit.py --only single-natural

# Use existing validated data
uv run --no-project --with typer --with pyyaml \
  python experiments/15-v2-ablation/submit.py \
  --only single-natural --skip-generation

# Explicitly replace generated matrix data
uv run --no-project --with typer --with pyyaml \
  python experiments/15-v2-ablation/submit.py --replace-data

# Reuse an existing run directory and resume checkpoints
uv run --no-project --with typer --with pyyaml \
  python experiments/15-v2-ablation/submit.py \
  --run-id <existing-id> --only single-natural --skip-generation --resume
```

Normal submission fails before creating a run directory when `sbatch` is not
available. Dry-run remains available on any machine.

## Configuration

`run.yaml` defines arms, model, data output, and per-stage Slurm resources.
Adding an arm does not require editing any job script. An aggregate view uses:

```yaml
- name: single-natural
  scope: by_variant
  view: single__natural__delta
```

A depth-specific arm uses:

```yaml
- name: direction-depth-10
  scope: by_cell
  cell: direction-depth-10
  view: single__natural__delta
```

Resources are passed as `sbatch` command-line overrides, so the generic scripts
contain no experiment-specific GPU allocation.

Training and evaluation configs use explicit deep overrides:

```yaml
train:
  template: templates/train.yaml
  overrides:
    num_epochs: 2
    learning_rate: 0.00005

eval:
  template: templates/eval.yaml
  task_template: templates/task.yaml
  overrides:
    model_args:
      max_thinking_tokens: 2048

arms:
  - name: single-natural-low-lr
    view: single__natural__delta
    train_overrides:
      learning_rate: 0.00003
```

Merge precedence is template, run override, arm override, then protected
computed fields. Dataset paths, output directories, evaluation tasks, adapter
paths, and answer choices cannot be overridden.

## Failure behavior

- Downstream jobs use `afterok` and `--kill-on-invalid-dep=yes`.
- The summary uses `afterany` and records missing arms.
- A partially submitted graph leaves all captured IDs in `jobs.json` and prints
  the exact `scancel` command.
- Completion requires verified adapter/results artifacts plus `COMPLETED`.
- Retry history is appended to `jobs-history.jsonl`.
