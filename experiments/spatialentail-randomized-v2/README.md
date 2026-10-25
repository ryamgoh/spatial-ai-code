# SpatialEntail randomized proof-graph v2

Track remaining required, optional, and deferred work in [CHECKLIST.md](CHECKLIST.md).

## Research analysis

The [research report](research-analysis/REPORT.md) audits all 61,200 saved
responses, checks hash-bound candidate-data separation, analyzes contract
transfer and stage-2 regressions, and includes four figures with separate
training seeds. Primary evaluation scores remain unchanged.

The [reproduction and cluster runbook](research-analysis/RUNBOOK.md) documents
the prepared seven-job external-transfer package. It uses the current
1,500-row SINGLE Corr correction, not the legacy 1,329-row dataset. External
GPU evaluation and final training-consumption provenance remain pending.

## Study scope

This experiment implements the structural calibration and non-headline
diagnostics for the randomized v2 study. The calibration must pass before a
fresh benchmark or SFT release is frozen.

## Cluster preparation

Submit the 2,400-example calibration from the repository root:

```sh
sbatch --partition=normal --cpus-per-task=128 --mem=128G \
  experiments/spatialentail-randomized-v2/slurm/prepare-calibration.sh
```

The frozen default boundary is train L1--L6/X0--X6 and evaluate through
L8/X8. The values can also be supplied explicitly when submitting the job:

```sh
sbatch --partition=normal --cpus-per-task=128 --mem=128G \
  --export=ALL,TRAIN_MAX_L=6,TRAIN_MAX_X=6,EVAL_MAX_L=8,EVAL_MAX_X=8 \
  experiments/spatialentail-randomized-v2/slurm/prepare-calibration.sh
```

Benchmark preparation reads the boundaries back from that calibration receipt,
so the train/OOD split cannot silently drift between jobs.

Prepare the special-response and paired consistency diagnostics separately:

```sh
sbatch --partition=normal --cpus-per-task=16 --mem=32G \
  experiments/spatialentail-randomized-v2/slurm/prepare-diagnostics.sh
```

After the calibration manifest records all 2,400 rows, prepare the fresh
3,600-example benchmark candidate:

```sh
sbatch --partition=normal --cpus-per-task=128 --mem=128G \
  experiments/spatialentail-randomized-v2/slurm/prepare-benchmark.sh
```

The benchmark command refuses incomplete calibration receipts and still calls
the result a candidate until its quotas, certificates, uniqueness, hashes, and
disjointness checks are reviewed.

After the benchmark candidate passes its audit, prepare the nested SINGLE and
MULTI training/development candidate. Benchmark canonical structures are hard
exclusions; MULTI shares its G1 core with SINGLE, and only SINGLE receives an
answer-only supervision view:

```sh
sbatch --partition=normal --cpus-per-task=128 --mem=128G \
  experiments/spatialentail-randomized-v2/slurm/prepare-training.sh
```

The output remains a training candidate until the pinned Qwen tokenizer admits
every complete training sequence and the persisted files pass their audit.

Run exact tokenizer admission after the structural candidate is audited:

```sh
sbatch --partition=normal --cpus-per-task=8 --mem=32G \
  experiments/spatialentail-randomized-v2/slurm/audit-training-tokens.sh
```

This verifies every candidate file hash and measures the complete rendered
Qwen chat sequence, evaluation prompt, and assistant target. The receipt
selects the smallest tested common training envelope that fits every row.

After tokenizer admission passes, export lean message-only JSONL and prepare
the six Axolotl jobs:

```sh
sbatch --partition=normal --cpus-per-task=2 --mem=16G \
  experiments/spatialentail-randomized-v2/slurm/prepare-sft-jobs.sh
```

The prepared jobs contain SINGLE Natural, SINGLE answer-only, and MULTI Natural
at seeds 42 and 43. They use QLoRA, the known-fit Qwen3.5 target modules, and
the measured common `sequence_len: 4096` on one H100-47 GPU.

Prepare the frozen 48-task evaluation package while training runs:

```sh
sbatch --partition=normal --cpus-per-task=8 --mem=32G \
  experiments/spatialentail-randomized-v2/slurm/prepare-evaluation.sh
```

The package contains the untuned model plus all six adapter paths. Its Slurm
scripts default to one H200, but `sbatch -p/-t/--gres` may override the resource
profile. Only the untuned job is runnable before adapters finish.

Prepare the separate six-condition general-instruction comparator batch:

```sh
sbatch -p normal -t 01:00:00 \
  experiments/spatialentail-randomized-v2/slurm/prepare-general-evaluation.sh
```

This package renders the frozen system/user messages with each checkpoint's
own pinned tokenizer and chat template. It does not reuse the Qwen-rendered P0
prompt. After preparation, verify the receipt:

```sh
python3 - <<'PY'
import json
from pathlib import Path

receipt = json.loads(Path(
    "experiments/spatialentail-randomized-v2/prepared-general-eval/jobs.json"
).read_text())
print(receipt["total_bases"], len(receipt["tasks"]), len(receipt["jobs"]))
PY
```

The expected values are `3600 48 6`. First submit the six-condition smoke
chain. Each job evaluates one example from every task, or 48 examples total:

```sh
bash experiments/spatialentail-randomized-v2/prepared-general-eval/submit-h200-smoke-chain.sh
```

Confirm that all six directories under `results/smoke/` contain `COMPLETED`
and `results.json`. Then submit all six full evaluations to the one-H200
allocation as an `afterok` chain:

```sh
bash experiments/spatialentail-randomized-v2/prepared-general-eval/submit-h200-chain.sh
```

The scheduler runs one model at a time. If a job fails, later jobs remain
blocked instead of silently producing an incomplete comparison. Inspect the
failed log, then resubmit the failed model or rebuild the remaining dependency
chain. Gemma 3 folds the frozen system instruction into the user message
because its native template lacks the same system-role contract; the semantic
instruction is retained and the receipt records this policy.

After the general package exists, prepare the four-condition reasoning batch:

```sh
sbatch -p normal -t 00:30:00 \
  experiments/spatialentail-randomized-v2/slurm/prepare-reasoning-evaluation.sh
```

The package reuses the general package's exact 3,600 model-neutral rows; it
does not regenerate or copy benchmark data. It contains fresh thinking-on runs
for Qwen3.5-4B and Qwen3.5-9B plus Ministral 3 8B Reasoning and
DeepSeek-R1-Distill-Qwen-7B. Stage 1 uses each model card's supported sampling
subset with seed 42; Stage 2 remains deterministic and constrained.

Run the reasoning smoke chain:

```sh
bash experiments/spatialentail-randomized-v2/prepared-reasoning-eval/submit-h200-smoke-chain.sh
```

After all four `results/smoke/*/COMPLETED` markers exist, launch the full chain:

```sh
bash experiments/spatialentail-randomized-v2/prepared-reasoning-eval/submit-h200-chain.sh
```

Prepare the fixed five-model spatial-specialist batch from the same general
package:

```sh
sbatch -p normal -t 00:30:00 \
  experiments/spatialentail-randomized-v2/slurm/prepare-spatial-evaluation.sh
```

The spatial package contains no images, image placeholders, or visual tokens.
All five models receive the same text-only system/user messages. Stage 1 and
Stage 2 are deterministic for this category so the comparison isolates spatial
post-training rather than model-specific sampling presets.

Run the one-question-per-model load and text-only preflight first:

```sh
bash experiments/spatialentail-randomized-v2/prepared-spatial-eval/submit-h200-preflight-chain.sh
```

Only after all five `results/preflight/*/COMPLETED` markers exist, run the
48-cell smoke chain:

```sh
bash experiments/spatialentail-randomized-v2/prepared-spatial-eval/submit-h200-smoke-chain.sh
```

After all five smoke jobs complete, launch the full chain:

```sh
bash experiments/spatialentail-randomized-v2/prepared-spatial-eval/submit-h200-chain.sh
```

RoboBrain and Cosmos require accepted Hugging Face gates. A checkpoint that
cannot complete a genuine no-image preflight is recorded as interface-
incompatible; do not add a dummy image or alter the frozen prompt.

Outputs and logs are written below this experiment directory. Preparation
fails rather than overwriting an existing output.

## Summarize downloaded results

The 2026-10-25 import includes six general runs, four reasoning runs, and the
untuned baseline plus six SFT adapters. See
[the validated summary](evaluation-summary/SUMMARY.md) and
[machine-readable results](evaluation-summary/summary.json).
No spatial-specialist results were supplied.

Each downloaded package needs its original `jobs.json` and
`results/<model>/{COMPLETED,results.json,responses_*.jsonl}`. Extract each archive
into its own prepared directory; all three archives use the same top-level
names. These result-only imports are not complete runnable evaluation packages:
their generated configs, task definitions, and data shards are absent.

From the repository root, summarize locally or on a CPU cluster node:

```sh
uv run --python 3.12 --no-project python -m spatial.v2.study.summarize_randomized \
  experiments/spatialentail-randomized-v2/prepared-general-eval \
  experiments/spatialentail-randomized-v2/prepared-reasoning-eval \
  experiments/spatialentail-randomized-v2/prepared-eval \
  --general-prepared experiments/spatialentail-randomized-v2/prepared-general-eval \
  --output-dir experiments/spatialentail-randomized-v2/evaluation-summary
```

This uses only the Python standard library (Python 3.11+); it does not load
models or install the GPU training environment. The reasoning receipt refers
to a hash-bound general receipt, which supplies its task counts. The command
validates all inputs before writing `SUMMARY.md` and `summary.json`, replacing
previous reports in the output directory. It leaves raw results untouched and
never reads smoke/preflight runs as full evaluations.

Per-example metadata supports contract, partition, query, G, measured L/X,
and held-out-transition breakdowns without another benchmark download. Missing
completion markers are reported as incomplete; inconsistent task coverage,
sample counts, duplicate base IDs, or sample/aggregate metrics stop reporting.
The reports retain input hashes and runtime configurations, but do not verify
training provenance, adapter contents, or the original cluster environment.

SINGLE is the 900-row G1 subset. Do not use the full mixed-contract score alone
to assess SINGLE training. The low stage-1 extraction scores of the reasoning
panel need inspection before making model-capability claims; the P0 untuned
baseline and fresh thinking-on comparator runs have different configurations.

## Local smoke

```sh
uv run python -m spatial.v2.study.randomized_v2
uv run pytest -q spatial/v2/tests/test_randomized_v2_study.py
uv run --python 3.12 --no-project python -m unittest \
  spatial.v2.tests.test_summarize_randomized
```
