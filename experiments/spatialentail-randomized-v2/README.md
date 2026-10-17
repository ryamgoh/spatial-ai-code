# SpatialEntail randomized proof-graph v2

This experiment implements the structural calibration and non-headline
diagnostics for the randomized v2 study. The calibration must pass before a
fresh benchmark or SFT release is frozen.

## Cluster preparation

Submit the 2,400-example calibration from the repository root:

```sh
sbatch --partition=normal --cpus-per-task=128 --mem=128G \
  experiments/spatialentail-randomized-v2/slurm/prepare-calibration.sh
```

The default boundary is train L1--L5/X0--X5 and evaluate through L8/X8. To
train through L6/X6 without changing code, freeze those values in the
calibration manifest:

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

Outputs and logs are written below this experiment directory. Preparation
fails rather than overwriting an existing output.

## Local smoke

```sh
uv run python -m spatial.v2.study.randomized_v2
uv run pytest -q spatial/v2/tests/test_randomized_v2_study.py
```
