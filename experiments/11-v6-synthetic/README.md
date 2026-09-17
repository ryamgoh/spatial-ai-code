# Experiment 11 — v6 synthetic SFT (2B / 4B × 1.5k / 6k / 20k)

**Question.** After dropping SpatialMap-shaped 4-ans gold, does SFT on
`generate_all_v6` traces (solver-labeled, 5 options) scale with n and
size, on a held-out **synthetic 20% test** vs **SpatialMap with v6 gold**?

Ignores 0.8B. Nested train: **1.5k ⊂ 6k ⊂ 20k**.

## Data

Generate 20k train (seed 42) then slice nested prefixes per bucket.
Held-out test: 4k (seed 43), same mix × 1/5. SpatialMap eval:
`make_spatialmap_v6.py` appends a fifth option and re-grades with
`SpatialSolver` (`data/spatialeval_v6_corr.jsonl`).

| n | dir-1 | dir-2 | undet | cycle | incompl. | omit | which-1 | which-2 | which-3 | which-4 | which-0 | count | count-omit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,500 | 150 | 125 | 75 | 50 | 50 | 50 | 150 | 125 | 75 | 50 | 100 | 400 | 100 |
| 6,000 | 600 | 500 | 300 | 200 | 200 | 200 | 600 | 500 | 300 | 200 | 400 | 1,600 | 400 |
| 20,000 | 2,000 | 1,667 | 1,000 | 667 | 667 | 666 | 2,000 | 1,667 | 1,000 | 667 | 1,333 | 5,333 | 1,333 |
| test 4k | 400 | 333 | 200 | 133 | 133 | 134 | 400 | 333 | 200 | 133 | 267 | 1,067 | 267 |

See `docs/question-types.md` for what each bucket is.

## Cells (6)

Qwen3.5 Instruct **2B / 4B** × n ∈ {1.5k, 6k, 20k}. Same QLoRA as Exp 10
(r=64, 2 epochs, acc 8, 1 GPU). `eval_steps` 45 / 180 / 600.

| tag | adapter |
|---|---|
| `{2b,4b}-1.5k` | `models/qwen3.5-{2,4}b-sft-v6-1500/` |
| `{2b,4b}-6k` | `models/qwen3.5-{2,4}b-sft-v6-6000/` |
| `{2b,4b}-20k` | `models/qwen3.5-{2,4}b-sft-v6-20000/` |

## Eval (after each SFT)

One `eval_new.py` run, two tasks:

1. `spatial_eval_v6_synth` — 4k synthetic test
2. `spatial_eval_v6_spatialmap` — SpatialMap 1500 with v6 fifth option + solver gold

## Run (three `h100-47:1` jobs)

`--cpus-per-task=16` (same as Exp 10). Launchers source
`slurm_pin_srun_cpus.sh`. Data gen is `flock`'d.

```bash
sbatch run_batch_11_sft_h100_47_2b.sh          # 2B  1.5k + 6k + 20k
sbatch run_batch_11_sft_h100_47_4b_small.sh    # 4B  1.5k + 6k
sbatch run_batch_11_sft_h100_47_4b_20k.sh      # 4B  20k
```

`SKIP_EVAL=1` to train only. `ONLY=2b-6k` to run one cell.

```bash
cd eval && uv run --no-project python ../experiments/11-v6-synthetic/scripts/summarize.py
```

## Artifacts

| artifact | location |
|---|---|
| 20k train pool | `data/spatial_sft_v6_scale_20000_train.jsonl` |
| Nested slices | `data/spatial_sft_v6_scale_{1500,6000}_train.jsonl` |
| Synth test | `data/spatial_sft_v6_scale_test.jsonl` |
| SpatialMap v6 | `data/spatialeval_v6_corr.jsonl` |
| Adapters | `experiments/11-v6-synthetic/models/` |
| Eval | `experiments/11-v6-synthetic/results/{tag}/` |
| Summary | `experiments/11-v6-synthetic/results/SUMMARY.md` |
