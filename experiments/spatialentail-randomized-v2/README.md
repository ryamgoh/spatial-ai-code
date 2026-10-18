# SpatialEntail randomized proof-graph v2

Track remaining required, optional, and deferred work in [CHECKLIST.md](CHECKLIST.md).

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

The expected values are `3600 48 6`. Submit all six to the one-H200 allocation
as an `afterok` chain:

```sh
bash experiments/spatialentail-randomized-v2/prepared-general-eval/submit-h200-chain.sh
```

The scheduler runs one model at a time. If a job fails, later jobs remain
blocked instead of silently producing an incomplete comparison. Inspect the
failed log, then resubmit the failed model or rebuild the remaining dependency
chain. Gemma 3 folds the frozen system instruction into the user message
because its native template lacks the same system-role contract; the semantic
instruction is retained and the receipt records this policy.

Outputs and logs are written below this experiment directory. Preparation
fails rather than overwriting an existing output.

## Local smoke

```sh
uv run python -m spatial.v2.study.randomized_v2
uv run pytest -q spatial/v2/tests/test_randomized_v2_study.py
```
