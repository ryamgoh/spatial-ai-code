# Experiment 6 — Baseline Task Feasibility

**Source.** Post-thesis follow-up, same eval stack as Exp 4b (`eval_new.py`,
two-pass vLLM, no `finetune.py`).

**Question.** Do un-tuned foundation models show structural performance
differences between single-answer and multi-answer spatial tasks?

**Hypothesis.** Even under Non-shot-2 (Exp 1’s best prompt, 77.6% on
DeepSeek-8B), Qwen3.5-4B-Base and Qwen3.5-4B stay weak. Multi-answer items
show higher rates of structural collapse (token loops, missing `Answer:`
lines, over-selecting letters). That justifies isolating spatial reasoning
on SpatialMap-TQA-Corr-Single during core training. Instruct vs base is an
open comparison: neither is tuned for this task, but the instruction-tuned
checkpoint may follow Non-shot-2 more reliably.

## Independent / dependent variables

| | |
|---|---|
| IV | Gold format: Single (`Answer: A`) vs Multi (`Answer: A, B` / `A,B,C,D`) vs full TQA-Corr |
| DV | Zero-shot strict/loose accuracy; generation anomalies |
| Constants | `Qwen/Qwen3.5-4B-Base` and `Qwen/Qwen3.5-4B`; two-pass eval; **Non-shot-2** system prompt (inlined from Exp 1); full remaining set (not a 1500 cap) |

## Datasets

From `data/spatialeval_cleaned.jsonl` (1,500 SpatialMap TQA rows).
Counts live next to the filters in `../tasks/utils.py`.

| set | n | gold |
|---|---|---|
| cleaned file | 1,500 | includes empty-oracle rows |
| empty oracle | 171 | `oracle_option` wiped: **dir 75** + **count 96** + **which 0** (see `../tasks/utils.py`) |
| SpatialMap-TQA-Corr | 1,329 | nonempty oracle |
| **SpatialMap-TQA-Corr-Single** | **1,038** | exactly one A–D letter |
| TQA-Corr multi (slice) | 291 | 2+ letters |

| type | 1-ans (Single) | 2-ans | 4-ans |
|---|---|---|---|
| count | 404 | 0 | 0 |
| dir | 332 | 93 | 0 |
| which | 302 | 0 | 198 |
| **total** | **1,038** | **93** | **198** |

Same file as Corr (`data/spatialeval_cleaned.jsonl`); the Single task is
only a `process_docs` filter (`utils.filter_single_letter_oracle`), the
same pattern as Corr’s `filter_nonempty_oracle`. Extract regex is unchanged,
so a model that emits `A, B` on a one-letter gold fails strict acc.

The default eval runs **only the 1,329-row Corr task**. Single vs multi
numbers are sliced from those generations (paired, no second sample of the
overlapping 1,038). Dedicated Single configs exist if you want a standalone
task score later.

## Systems

| id | system | eval config |
|---|---|---|
| B | Qwen3.5-4B-Base + Non-shot-2 | `eval-base.yaml` |
| I | Qwen3.5-4B instruct + Non-shot-2 | `eval-instruct.yaml` |
| B-S / I-S | same checkpoints on the Single task only | `eval-base-single.yaml`, `eval-instruct-single.yaml` |

Protocol matches Exp 4b `eval-base-4b.yaml` plus Exp 1 Non-shot-2:
`vllm_staged_pass`, 4096 thinking tokens, constrained stage-2 `A(,A)*`
decode, `batch_size: 4`, `max_model_len: 8192`, `system_instruction`
copied from `../01-few-shot-prompting/nonshot_2.yaml`. Stage 2 still
*allows* multi-letter strings on Single items — the IV is the gold set,
not the decoder grammar. Prompt is a constant, not an arm.

## Run (1× H100-47)

```bash
sbatch run_batch_06_feasibility.sh
```

Idempotent per tag: a tag with `results.json` is skipped. Overrides:

- `SKIP_INSTRUCT=1` / `SKIP_BASE=1`
- `SINGLE_EVAL=1` — also run the dedicated Single-task configs

Each eval's timestamped dir is moved to `results/feasibility/<tag>/`.
`scripts/summarize.py` writes `results/feasibility/SUMMARY.md` (accuracy
by slice + anomaly rates + type × cardinality).

Anomaly flags (on the logged thinking+answer string):

| flag | meaning |
|---|---|
| `format_fail` | no `Answer: A-D` match |
| `missing_entry` | empty predicted letter set |
| `token_loop` | 20–80 char chunk repeated ≥5 times in thinking |
| `over_select` | more predicted letters than gold |
| `all_four` | predicted `A,B,C,D` |

## Artifacts

| artifact | location |
|---|---|
| Shared TQA jsonl | `data/spatialeval_cleaned.jsonl` |
| Single filter | `../tasks/utils.py` → `filter_single_letter_oracle` |
| lm-eval task | `../tasks/spatial_eval_gen_cleaned_single.yaml` |
| Eval results | `results/feasibility/{base,instruct}/` |
| Summary | `results/feasibility/SUMMARY.md` |

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
