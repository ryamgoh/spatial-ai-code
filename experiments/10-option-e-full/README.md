# Experiment 10 — Option E (“None of these is proven”) / TQA-Corr-Full

**Source.** Harder SFT after Exp 8 saturated Single and Exp 8b zeroed
multi. First run: **Qwen3.5-4B Instruct** only, data-scaling like Exp 8
2.1 (`n` ∈ {1,500, 5,000, 20,000}, nested, 2 epochs, no early stop).

**Question.** Can 4B Instruct SFT learn to emit **E** when no A–D letter
is proven, still not pick E when A–D *are* proven, and does that
improve with more Full-mix traces?

## SpatialMap-TQA-Corr-Full (n = 1,500)

Same 1,500 cleaned SpatialMap rows. Every item gets

`E. None of these is proven`

Gold is **select only what the graph proves**. If nothing in A–D is
proven, gold is **E**.

| gold | n | what |
|---|---|---|
| **E** | **369** | dir empty 75 + count empty 96 + which-4 fallback 198 |
| A–D (1 or 2 letters) | 1,131 | original Corr gold; E is a distractor |

which-4 is remapped to E. The type-1 fallback had labeled every
in-passage name `A,B,C,D` when first pass proved none of them.

Build (idempotent):

```bash
cd eval && uv run --no-project python ../experiments/10-option-e-full/scripts/make_corr_full.py
```

Writes `data/spatialeval_corr_full.jsonl`. Task:
`spatial_eval_gen_cleaned_full` (regex `[A-E]`, `choices` A–E, stage-2
grammar includes E).

## SFT (4B Instruct, nested Full mix)

Synthetic traces **with E on every item**, not the 1,500 eval maps.
Generate 20k once (`generate_all.py`, seed 42, `--include-option-e`,
`--test-split 0`). `scripts/make_full_scale_data.py` writes nested 5k
and 1.5k (1.5k ⊂ 5k ⊂ 20k), stratified to Full proportions
(1.5k = census; 5k = 10/3; 20k = 40/3).

| n | dir-1 | dir-2 | dir-E | which-1 | which-E | count-1 | count-E |
|---|---|---|---|---|---|---|---|
| 1,500 | 332 | 93 | 75 | 302 | 198 | 404 | 96 |
| 5,000 | 1,107 | 310 | 250 | 1,007 | 660 | 1,346 | 320 |
| 20,000 | 4,428 | 1,240 | 1,000 | 4,028 | 2,640 | 5,384 | 1,280 |

Gold-E traces: A–D are all *wrong* (wrong dirs / entities / counts);
answer `E`. No synthetic which-4: eval has no which-4 gold. Same QLoRA
recipe as Exp 8 4B (2 epochs, no early stop). `eval_steps` /
`save_steps` scale with n (45 / 150 / 600).

| tag | n | adapter |
|---|---|---|
| `4b-1.5k` | 1,500 | `models/qwen3.5-4b-sft-full1500/` |
| `4b-5k` | 5,000 | `models/qwen3.5-4b-sft-full5000/` |
| `4b-20k` | 20,000 | `models/qwen3.5-4b-sft-full20000/` |

**Hypothesis.** Strict on Full rises with n (or saturates like Exp 8
Single at 1.5k–5k). Gold-E recall high on dir/count/which-E; Single and
dir-2 A–D do not collapse to E.

## Run (2× H100-96, `gpu-long`)

Submit together. Data gen is `flock`'d: whichever job starts first
writes the 20k pool + nested 1.5k/5k; the other waits and reuses.
Each job trains its cells, then evals them on the same 96
(`results/full/<tag>/`).

```bash
sbatch run_batch_10_sft_h100_96.sh        # 4b-1.5k + 4b-5k
sbatch run_batch_10_sft_h100_96_20k.sh    # 4b-20k
```

`SKIP_EVAL=1` to train only. Optional later eval on H200:

```bash
sbatch run_batch_10_eval_full_h200.sh
```

```bash
cd eval && uv run --no-project python ../experiments/10-option-e-full/scripts/summarize.py
```

## Artifacts

| artifact | location |
|---|---|
| Eval jsonl | `data/spatialeval_corr_full.jsonl` |
| 20k SFT pool | `data/spatial_sft_full_scale_20000_train.jsonl` |
| Nested slices | `data/spatial_sft_full_scale_{1500,5000}_train.jsonl` |
| Adapters | `models/qwen3.5-4b-sft-full{1500,5000,20000}/` |
| Eval | `results/full/{4b-1.5k,4b-5k,4b-20k}/` |
| Summary | `results/full/SUMMARY.md` |

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
