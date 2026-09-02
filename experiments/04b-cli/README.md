# Experiment 4b — same 4a smoke, Axolotl CLI instead of `finetune.py`

**Source.** Dissertation H036660. Process check, not a new scientific
condition: can the 4a Qwen3.5-4B pipeline (SFT → merge → GRPO → eval)
run from `axolotl` CLI + a SLURM script, without `finetune.py`.

**Question.** Does Axolotl 0.18's CLI (`train`, `merge-lora`, `vllm-serve`)
cover everything `finetune.py` was doing, or do we still need a Python
shim for axolotl/TRL bugs?

**What is the same as 4a.** Model, data, LoRA targets, 20-step GRPO,
16-row SpatialMap-TQA-CORR slice (`spatial_eval_4a_smoke16`), 2×H100-47.

**What is different.** Launch path only.

| stage | 4a | 4b |
|---|---|---|
| SFT | `uv run python finetune.py train-sft-4b.yaml` | `axolotl train … --launcher python` |
| merge | `merge_sft.py` → sibling `*-sft-merged/` | `axolotl merge-lora` → `<sft output_dir>/merged/` |
| GRPO data | `generate_grpo.py` | same (not an Axolotl command) |
| GRPO serve | `axolotl vllm-serve` | same |
| GRPO train | `uv run python finetune.py train-grpo-…` | `axolotl train … --launcher python` |
| eval | `eval_new.py` | same (spatial task, not `axolotl lm-eval`) |

`--launcher python` is required: CLI default is `accelerate`, which would
see both MIG slices. 4a's `finetune.py` already ran as a single process.

## Still not CLI

These are ours, not Axolotl:

- `finetune/generate_grpo.py` — synthetic GRPO prompts
- `eval/eval_new.py` — two-pass SpatialMap eval
- this SLURM script — GPU pin, skip/resume, vLLM health, result layout

Axolotl is pinned to git `7d77580` (0.19.dev), which uses
`self._dist.is_fsdp` instead of `VLLMGeneration.is_fsdp_enabled`.
GRPO train is CLI again.

## Artifacts

| artifact | location |
|---|---|
| 16-row benchmark | `data/spatialeval_4a_smoke16.jsonl` (shared with 4a) |
| SFT adapter | `models/qwen3.5-4b-sft/` |
| Merged SFT | `models/qwen3.5-4b-sft/merged/` |
| GRPO data | `spatial_grpo_data_smoke4b.jsonl` (repo root) |
| GRPO adapter | `models/qwen3.5-4b-grpo-smoke/` |
| Eval results | `results/smoke16/{base,sft,grpo}/` + `SUMMARY.md` |

## Run (2× H100-47)

```bash
sbatch run_batch_4b_cli_h100.sh
```

Overrides: `SKIP_TRAIN=1`, `SKIP_EVAL=1` (same as 4a).

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
