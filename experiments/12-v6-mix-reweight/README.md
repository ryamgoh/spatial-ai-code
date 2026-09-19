# Experiment 12 — hierarchical-uniform mix + three letter rules

**Question.** Does 4B-1.5k SFT close Type-0 holes (`dir-2`,
`dir-incomplete`, `dir-cycle`) if we (1) write the three letter rules into
the system prompt, (2) use a two-level **equal** mix (1/3 types, equal
subtypes inside each type), (3) shuffle the JSONL (standard SFT), and (4) pick the
checkpoint on those three buckets — without HBO actors or a custom
dataloader?

Same generator (`generate_all_v6.py`) and solver. One **seed-52** 21k
draw, three-way split: **2k test** (report once) + **1k val** (overfit /
ckpt pick) + **1.5k ⊂ 6k ⊂ 18k** train. Same two-level equal mix in all
three. SpatialMap v6 is unmatched transfer. The old seed-43 synth file
is **not** the Exp 12 headline.

**Split vs distribution** (what is matched, what is leakage, what is
cited): [`docs/split-and-distribution.md`](../../docs/split-and-distribution.md).

## What changed vs the previous mix

| lever | previous v6 SFT mix | Exp 12 |
|---|---|---|
| System prompt | “one axis derived → remaining compounds” | three Type-0 rules (pair listed / incomplete → CBD / cycle → CBD) |
| Mix | Type 0 unequal; `count-1` ~27% of the file | **1/3 dir, 1/3 which, 1/3 count**; equal subtypes inside each type |
| JSONL order | one shuffle (Axolotl shuffles again) | shuffle the file; `shuffle_merged_datasets: true` (default SFT) |
| Early stop | `eval_loss` on a 5% random split | `eval_loss` on 200 hard val traces; **letter-set pick** on the hard synth slice |
| Models | 2B/4B × 1.5k/6k/20k | 4B × 1.5k ⊂ 6k ⊂ 18k |

## Mix

| n | dir-1 | dir-2 | undet | cycle | incompl. | omit | which-1 | which-2 | which-3 | which-4 | which-0 | count | count-omit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2k test** | 111 | 111 | 111 | 111 | 111 | 111 | 134 | 134 | 134 | 133 | 133 | 333 | 333 |
| **1k val** | 56 | 56 | 55 | 55 | 55 | 56 | 67 | 67 | 67 | 66 | 66 | 167 | 167 |
| 1,500 | 84 | 84 | 83 | 83 | 83 | 83 | 100 | 100 | 100 | 100 | 100 | 250 | 250 |
| 6,000 | 334 | 334 | 333 | 333 | 333 | 333 | 400 | 400 | 400 | 400 | 400 | 1,000 | 1,000 |
| 18,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,200 | 1,200 | 1,200 | 1,200 | 1,200 | 3,000 | 3,000 |
| val hard | 0 | 80 | 0 | 60 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

21k pool = 2k test + 1k val + 18k train (disjoint). 1.5k ⊂ 6k ⊂ 18k **by bucket**; each file is shuffled.

## Cells

| tag | adapter | default job |
|---|---|---|
| `4b-1.5k` | `models/qwen3.5-4b-sft-v12-1500/` | default |
| `4b-6k` | `models/qwen3.5-4b-sft-v12-6000/` | `ONLY=4b-6k` |
| `4b-18k` | `models/qwen3.5-4b-sft-v12-18000/` | `ONLY=4b-18k` or `experiments/12-v6-mix-reweight/slurm/sft-4b-18k-h100-47.sh` |
| `baseline` | none (untuned `Qwen/Qwen3.5-4B`) | `experiments/12-v6-mix-reweight/slurm/eval-baseline-h200.sh` |
| `baseline-oneshot` | none (untuned `Qwen/Qwen3.5-4B`) | `ONLY=baseline-oneshot sbatch experiments/12-v6-mix-reweight/slurm/eval-baseline-h200.sh` |
| `baseline-threeshot` | none (untuned `Qwen/Qwen3.5-4B`) | `ONLY=baseline-threeshot sbatch experiments/12-v6-mix-reweight/slurm/eval-baseline-h200.sh` |

Baseline: same 2k test + SpatialMap v6, same two-stage protocol, `lora_path`
removed. `baseline` uses the three-letter-rule system prompt (matches the SFT
cells), so its only difference from the SFT cells is the adapter — it isolates
the SFT gain.

Few-shot baseline (zero/one/three-shot ladder): `baseline-oneshot` and
`baseline-threeshot` keep the letter-rules prompt and append full v12 demos
(question + CoT + `Answer:`). Demos are the shortest (≥350-word) 1k-val rows of
`dir-2`, `dir-cycle`, `count-omit` — covering the two-letter gold, the
cycle→CBD rule, and None-of-the-Options — and are verified disjoint from the
2k test (`experiments/prompts/fewshot_demo_provenance.txt`). `max_model_len` is
raised to 16384 for these two cells. They answer: can in-context demos close
the Type-0 holes without the adapter?

## Prompt family (canonical copies)

Round-indexed in `experiments/prompts/` — a new iteration gets a new index,
files are never edited in place (git history holds the old ones):

| file | round | contents |
|---|---|---|
| `nonshot_1..4.txt` | Exp 1 zero-shot rounds 1–4 | pre-v6: no fifth option, no letter rules — **do not use for v12** |
| `oneshot_3.txt`, `threeshot_3.txt` | Exp 1 few-shot round 3 | pre-v6 — **do not use for v12** |
| `nonshot_5.txt` | Exp 12 zero-shot | the three Type-0 letter rules (= `12-v6-mix-reweight/system_prompt.txt`, the training/eval stamp) |
| `oneshot_5.txt` | Exp 12 one-shot | `nonshot_5` + dir-2 demo |
| `threeshot_5.txt` | Exp 12 three-shot | `nonshot_5` + dir-2 / dir-cycle / count-omit demos |

`experiments/prompts/check_prompt_sync.py` fails if any inline copy in the
yamls or the training stamp drifts from its canonical file — run it after
touching prompts:

```bash
uv run --no-project python experiments/prompts/check_prompt_sync.py
```

QLoRA same as Exp 10/11 (r=64, 2 epochs, acc 8, 1 GPU, `cpus-per-task=16`).

## Eval

1. **2k test** — `spatial_eval_v12_synth`. Headline. Do not pick ckpts here.
2. **1k val** — `spatial_eval_v12_val`. Overfit + `SCORE_CKPTS=1`. Axolotl `eval_loss` uses a 200-row NLL slice of this val.
3. **SpatialMap v6** — unmatched transfer. Must not drop.

Easy buckets should stay high (dir-1 / which-* / count-* ≳ 95%). Type 2 equal-split means `count-omit` is 1/6 of the file (was ~7%).

## Run

```bash
sbatch experiments/12-v6-mix-reweight/slurm/sft-4b-h100-47.sh              # 4B-1.5k
ONLY=4b-6k sbatch experiments/12-v6-mix-reweight/slurm/sft-4b-h100-47.sh
sbatch experiments/12-v6-mix-reweight/slurm/sft-4b-18k-h100-47.sh          # 4B-18k (own 47)
SKIP_EVAL=1 sbatch experiments/12-v6-mix-reweight/slurm/sft-4b-h100-47.sh
sbatch experiments/12-v6-mix-reweight/slurm/eval-sft-h200.sh
SCORE_CKPTS=1 sbatch experiments/12-v6-mix-reweight/slurm/eval-sft-h200.sh     # letter-set pick on saves
sbatch experiments/12-v6-mix-reweight/slurm/eval-baseline-h200.sh          # baseline + baseline-nonshot2 (untuned 4B)
```

Data gen is `flock`'d on `data/.spatial_sft_v12.lock`.
Pool seed **52**; val seed **53**. `--system-prompt-file` is Exp 12 only; omit it and `generate_all_v6.py` still writes the built-in v6 prompt.

```bash
cd eval && uv run --no-project python ../experiments/12-v6-mix-reweight/scripts/summarize.py
uv run --no-project python experiments/12-v6-mix-reweight/scripts/score_checkpoints.py 4b-1.5k
```

## Artifacts

| artifact | location |
|---|---|
| System prompt | `system_prompt.txt` (stamped onto traces; eval yaml matches) |
| 21k pool | `data/spatial_sft_v12_21000_pool.jsonl` |
| 2k test | `data/spatial_sft_v12_2000_test.jsonl` |
| 1k val | `data/spatial_sft_v12_1000_val.jsonl` |
| 200 NLL slice | `data/spatial_sft_v12_val_nll.jsonl` (from the 1k val) |
| Train 1.5k / 6k / 18k | `data/spatial_sft_v12_{1500,6000,18000}_train.jsonl` |
| Adapters | `experiments/12-v6-mix-reweight/models/` |
| Eval | `experiments/12-v6-mix-reweight/results/{tag}/` |

## Decision after 4B-1.5k

- Three buckets jump, SpatialMap stays high → lock the mix; run 4B-6k only if you still want a size curve.
- Synth holes die, SpatialMap drops → restore dir-1 mass.
- Still ~dir-cycle 37% → GRPO on prompt-only items from those three buckets. Still no HBO.
