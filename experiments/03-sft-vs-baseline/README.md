# Experiment 3 — Baseline vs. Finetuned

**Source.** Dissertation H036660, §4.5 / Table 1 + Tables 4.1–4.2.

**Question.** Does SFT on Enhanced-SpatialMap-TQA (AxisDecomposition CoT
traces) beat the un-tuned DeepSeek baseline on SpatialMap-TQA-CORR?

**Systems (same base: DeepSeek-R1-0528-Qwen3-8B).**

| id | system | config |
|---|---|---|
| M0 | Baseline, no structured prompting | `../00-baseline-model/eval-baseline.yaml` |
| M1 | QLoRA SFT on AxisDecomposition CoT | `train-sft-8b.yaml` + `eval-sft-finetuned.yaml` |

Training data: Enhanced-SpatialMap-TQA — 1,500 synthetic samples
(600×1-answer / 600×2-answer / 300×4-answer), 1,200 train / 300 test
(`data/spatial_sft_data_all_train.jsonl`). Adapter lands at
`experiments/03-sft-vs-baseline/models/deepseek-r1-qwen3-8b`.

**Headline result (validated set, n=1,329).**

| | M0 | M1 |
|---|---|---|
| Correct | 980 / 1329 | 1168 / 1329 |
| Accuracy | **73.7%** | **87.9%** |
| Gain | — | **+14.2 pp** (+188 questions) |

**Secondary (Table 4.2).** M1 is also cheaper/faster: avg response tokens
5,816 → 708 (8.22×), wall-clock 11.2 s → 2.1 s (5.33×). M0 shows unbounded
reasoning; M1 follows the fixed 4-step procedure.

**Regressions (Table 4.1).** M1 loses 54 questions M0 got right:
[A] genuine Step-3 error 34, [B] overthinks direct relation 11,
[C] wrong option letter 6, [D] missing `ANSWER:` line 3.

Post-SFT GRPO is a separate follow-up — see `../05-grpo/` (not in the report).

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
