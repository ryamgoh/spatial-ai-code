# Reproduction and Cluster Handoff

Run local commands from the repository root. Output directories for analysis
and preparation must be new: the tools refuse to replace prior artifacts.

## Reproduce the Internal Audit

The current artifacts used the three imported evaluation packages, the two
candidate archives in Downloads, and the six-job training preparation receipt
named `jobs (1).json`. The latter is not the older pilot `jobs.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --offline --python 3.12 --no-project \
  --with tokenizers python -m spatial.v2.study.analyze_randomized \
  --experiment experiments/spatialentail-randomized-v2 \
  --benchmark-archive "$HOME/Downloads/prepared-benchmark.tar.gz" \
  --training-archive "$HOME/Downloads/prepared-training-candidate.tar.gz" \
  --training-receipt "$HOME/Downloads/jobs (1).json" \
  --qwen-tokenizer "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json" \
  --output /tmp/randomized-v2-research-replay

MPLCONFIGDIR=/tmp/randomized-mpl-cache uv run --offline --python 3.12 \
  --no-project --with matplotlib==3.11.2 --with numpy==2.5.4 \
  python -m spatial.v2.study.gen_fig_randomized /tmp/randomized-v2-research-replay
```

The script hashes and streams archives without extracting their large files.
`input_hashes.json` binds the evaluation receipts and every response JSONL.
`provenance.json` records the benchmark, candidate file, training receipt, and
tokenizer hashes. Alternate extraction is diagnostic only.

## Reproduce the Current Corr Package

The corrected source was rebuilt with the existing Z3 audit implementation:

```bash
uv run --offline --python 3.12 --no-project --with z3-solver --with typer \
  python -m spatial.v2.audit_spatialeval \
  --input data/spatialeval_org.jsonl \
  --output-dir /tmp/randomized-v2-corr-replay --expected-rows 1500

uv run --offline --python 3.12 --no-project --with tokenizers --with pyyaml \
  python -m spatial.v2.study.prepare_randomized_corr \
  --correction /tmp/randomized-v2-corr-replay \
  --sft-prepared experiments/spatialentail-randomized-v2/prepared-eval \
  --output experiments/spatialentail-randomized-v2/prepared-corr-replay \
  --tokenizer "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json"
```

Validated correction SHA-256:
`fc4e56d8921ef416a9d61dfe0d05e8f4dba25e8c9f8ad40a1ccb477f49e8a232`.

There are 1,500 rows, 667 determinate answers, and 833 gold `E` answers.
The maximum rendered prompt is 348 pinned-tokenizer tokens. Jobs use the
existing 4,096-token reasoning and 8,192-token context budgets. Stage 2 selects
exactly one of A-E, appropriate to the current SINGLE Corr contract; internal
randomized-v2 allowed sets. Report the two protocols separately.

The package records newly prepared input hashes; it does not assert that a
previously sealed external package was recovered. Its regenerated semantic
counts match the current report. Manual assumption sensitivity remains pending.

## Run External Transfer on the Cluster

The experiment code, configs, task definitions, corrected data, and receipts
are versioned in Git. Model weights, raw responses, logs, and transfer archives
are not included in this experiment commit. Push the experiment commit to the
intended remote branch before pulling it on the cluster.

From the **cluster repository root**, on that same branch:

```bash
git pull --ff-only
git log -1 --oneline
mkdir -p experiments/spatialentail-randomized-v2/prepared-corr-eval/logs \
  experiments/spatialentail-randomized-v2/prepared-corr-eval/results
```

Confirm the expected experiment commit before submission. Git does not preserve
empty log/result directories, so create them before Slurm opens its log files.
If an earlier tar extraction causes an untracked-file conflict on pull, stop
and move only the conflicting files aside, preserving any existing results.
Do not use `git clean` or discard local cluster changes to force the pull.
No archive upload or extraction is needed for the Git-based setup.

The cluster must already have the source/environment versions pinned in
`prepared-corr-eval/jobs.json` and the six trained adapters at
`prepared-sft/models/<name>`. The shared wrapper checks adapter completion,
prepared file hashes, and environment hashes before inference.
Do not bypass a hash mismatch; align the intended source versions first.

Start with the untuned run to verify the cluster environment:

```bash
sbatch experiments/spatialentail-randomized-v2/prepared-corr-eval/slurm/untuned.sh
```

After confirming it completes and its saved responses are sensible, submit
the six adapters:

```bash
for arm in single-natural single-answer-only multi-natural; do
  for seed in 42 43; do
    sbatch "experiments/spatialentail-randomized-v2/prepared-corr-eval/slurm/${arm}-seed-${seed}.sh"
  done
done
```

These are full evaluation jobs, not training jobs. Each requests the existing
H200 configuration; scheduler availability and model loading were not tested
locally. No jobs have been submitted by this analysis.

After all seven runs complete:

```bash
tar -czf ~/corr-eval-results.tar.gz \
  -C experiments/spatialentail-randomized-v2/prepared-corr-eval \
  jobs.json results/ logs/
```

Download locally:

```bash
scp xlogin:~/corr-eval-results.tar.gz "$HOME/Downloads/"
```

Next analysis should verify the complete 1,500-ID coverage per model and report
strict accuracy by query and determinate/underdetermined status, native stage-1
accuracy, stage changes, `E` recall, and false abstention on determinate rows.
An always-`E` predictor gets 55.53% overall, so aggregate accuracy alone is
insufficient. Pair checkpoint contrasts by source ID, keep seeds separate, and
retain the original stage-2 metric. External scores were validated on 2026-10-26.
The internal randomized summarizer does not accept the new Corr schema.

## Summarize Downloaded Corr Results

All seven runs in `~/Downloads/corr-eval-results.tar.gz` passed full coverage,
source/prompt/config checks, original scorer replay, and aggregate consistency.
The [Corr summary](../corr-evaluation-summary/SUMMARY.md) reports scores and
paired comparisons; `summary.json` records archive/member hashes, and
`predictions.csv` retains per-ID predictions and diagnostics.

This command reads the archive directly without extracting or overwriting raw
results. Choose a new output directory for each replay:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --offline --python 3.12 --no-project \
  --with pyyaml python -m spatial.v2.study.summarize_corr \
  --archive "$HOME/Downloads/corr-eval-results.tar.gz" \
  --prepared experiments/spatialentail-randomized-v2/prepared-corr-eval \
  --output-dir /tmp/randomized-v2-corr-summary-replay
```

Original archive SHA-256:
`021d9cf7d84449042d947fae716a2cd3a7dff7f87df1608c535f5cc58ba70f75`.

Keep the archive as the raw evidence, outside Git. The preparation receipt's
`submitted: false` is historical preparation metadata; it is not a live job
status. Do not alter that hash-bound receipt to record completion.

## Matched Transfer Diagnostic

A separate post-Corr diagnostic is prepared in `prepared-transfer-diagnostic/`:
60 fresh bases, four policy/layout views each, and seven unsubmitted checkpoint
jobs. It uses one fixed set-capable decoder across views. Do not substitute
its scores for frozen Corr scores. The [diagnostic design and handoff](TRANSFER-DIAGNOSTIC.md)
includes Git-based deployment, submission, download, and summarization commands.
No training changes are required.

## Remaining Provenance

Bring back the six final adapter completion markers, hashes of
`adapter_model.safetensors` and `adapter_config.json`, training configuration
and data receipts, trainer state, and training logs. We do not need the adapter
weights locally for summarization. These establish training consumption and
checkpoint identity beyond the candidate/preparation evidence audited here.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --offline --python 3.12 --no-project \
  --with pytest --with pyyaml python -m pytest \
  spatial/v2/tests/test_analyze_randomized.py \
  spatial/v2/tests/test_summarize_randomized.py \
  spatial/v2/tests/test_summarize_corr.py \
  spatial/v2/tests/test_evaluation_chat.py \
  spatial/v2/tests/test_study_metrics.py -q -p no:cacheprovider
```

Result after the Corr import: 31 passed, 18 subtests passed. A second full
archive replay exactly reproduced the saved Corr JSON and Markdown.
Separate local package validation checked
all input/environment hashes, all 1,500 gold-answer scoring replays, all seven
adapter/config paths, and Bash syntax. These checks are not GPU smoke tests.
