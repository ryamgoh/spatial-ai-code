# Experiment 4a — Qwen3.5-4B full-pipeline smoke

**Source.** Dissertation H036660. Smoke test of the full
SFT → merge → GRPO → eval pipeline on a smaller base model
(Qwen3.5-4B instead of DeepSeek-R1-Qwen3-8B), run on 2×H100.

**Question.** Does the full pipeline (QLoRA SFT, LoRA merge, short GRPO
run, two-stage vLLM eval) run end-to-end on Qwen3.5-4B, and does the
trained model change answers on a 16-row SpatialMap-TQA-CORR slice?

**Why 16 rows.** A stratified slice of the cleaned 1,329-row
SpatialMap-TQA-CORR set covering **every question-type / answer-count
case that exists in the dataset**:

| type | answer count | rows |
|---|---|---|
| dir ("In which direction…") | 1-answer | 4 |
| dir | 2-answer | 1 |
| which ("Which object is in…") | 1-answer | 3 |
| which | 4-answer | 2 |
| count ("How many objects…") | 1-answer | 6 |

Total 16. (dir-4-answer and which-2-answer rows do not exist in
TQA-CORR; empty-oracle rows are filtered by the task.)

**Systems (all on the same 16-row benchmark).**

| id | system | train config | eval config |
|---|---|---|---|
| B | Qwen3.5-4B base, no prompt | — | `eval-base-4b.yaml` |
| S | B + QLoRA SFT (AxisDecomposition traces) | `train-sft-4b.yaml` | `eval-sft-4b.yaml` |
| G | S (merged) + 20-step GRPO smoke | `train-grpo-4b-smoke.yaml` | `eval-grpo-4b.yaml` |

**Pipeline artifacts (created by the batch launcher).**

| artifact | location |
|---|---|
| 16-row benchmark (seeded, stratified) | `data/spatialeval_4a_smoke16.jsonl` — regenerate with `scripts/make_smoke16.py` |
| lm-eval task (shared, next to `utils.py`) | `../tasks/spatial_eval_4a_smoke16.yaml` |
| SFT adapter | `models/qwen3.5-4b-sft/` |
| Merged SFT (bf16) | `models/qwen3.5-4b-sft-merged/` |
| GRPO data (~300 prompts, seeded) | `spatial_grpo_data_smoke4a.jsonl` (repo root) |
| GRPO adapter | `models/qwen3.5-4b-grpo-smoke/` (or `checkpoint-*/`) |
| Per-config eval results | `results/smoke16/{base,sft,grpo}/` |
| Summary table | `results/smoke16/SUMMARY.md` |

## Run (H100)

```bash
sbatch run_batch_4a_smoke_h100.sh
```

One submission runs the whole pipeline: SFT (2 epochs, 16k context,
QLoRA r=64) → merge → ~300 GRPO prompts → 20-step GRPO (GPU0 vLLM
serve + GPU1 train) → three evals → `SUMMARY.md`. Every stage is
idempotent: re-submit to resume/continue (SFT/merge skipped when the
artifacts exist; GRPO resumes from the newest checkpoint).

Overrides:

- `SKIP_TRAIN=1` — re-run only the evals on existing adapters.
- `SKIP_EVAL=1` — train only.

The launcher moves each eval's timestamped results dir into
`results/smoke16/<tag>/` so the three configs sit side by side, then
`scripts/summarize.py` builds `SUMMARY.md` (strict/loose accuracy per
config + deltas vs. base).

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
