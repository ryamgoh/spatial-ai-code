# Experiment 0 — Baseline Model Comparison

**Source.** Dissertation H036660, §4.2 / Appendix Fig. 4.

**Question.** Which 2025/2026 reasoning / instruction model is the strongest
base architecture for SpatialMap-TQA-CORR (1,329 validated questions)?

**Setup.** Un-tuned models, two-pass vLLM eval (free thinking + constrained
`Answer:` extraction). Reasoning models tend to outperform instruction-tuned
ones — some spontaneously use coordinate notation.

**Config.** `eval-baseline.yaml` — DeepSeek-R1-0528-Qwen3-8B (the winner).
Other models from this sweep live under `../02-spatial-baseline-models/`
(Exp 2) and were also used as Exp 0 candidates.

**Result.** **DeepSeek-R1-0528-Qwen3-8B: 73.7%** overall.
Next best in the reported comparison: Ministral-3-8B-Instruct at 62.5%
(−11.2 pp). Other reported points: ~57.5%, 45.9%, 43.7%.

This model is locked in as the base for Experiments 1 and 3.

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
