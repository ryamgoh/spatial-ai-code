# Experiment 2 — Comparison with Spatial Baseline Models

**Source.** Dissertation H036660, §4.4 / Appendix Fig. 6.

**Question.** On SpatialMap-TQA-CORR, how do specialised spatial / embodied
models compare to the general-purpose DeepSeek baseline — under their native
prompt, and under our best Non-shot-2 structured prompt?

**Setup.** Un-tuned third-party models, two-pass vLLM. Two conditions per
model: (1) native / baseline prompt, (2) Non-shot-2 structured prompt from
Exp 1.

**Configs.**

- `other_models/` — Cosmos-Reason2-8B, Falcon-H1R-7B, RoboBrain2.5-8B-NV,
  SpaceQwen3-VL-2B-Thinking, SpaceR (baseline + nonshot_2 each)
- `experiments_two/` — cogito-v1-preview-llama-8B, DeepSeek-R1-Distill-Llama-8B,
  gemma-3-12b-it, Ministral-3-8B-Instruct-2512-BF16, Mistral-NeMo-12B-Instruct,
  NVIDIA-Nemotron-3-Nano-4B-BF16 (baselines; also used in Exp 0)

Launcher: `run_batch_h100.sh` (active: Ministral / gemma / Mistral-NeMo).

**Result (from thesis Fig. 6).** Specialised spatial models land in the
**~40–57%** range with native prompts — well below DeepSeek's 73.7%.

- **Cosmos-Reason2-8B**: largest gain from Non-shot-2, **+9.8 pp**
  (≈52.8% → 62.6%).
- **Falcon-H1R-7B** and several others **degraded** under Non-shot-2 —
  already tuned to their native format.
- SpaceQwen / SpaceR remain lowest (~27–41%).

Domain-specific spatial architectures do not automatically beat a strong
general reasoning model on this text-only benchmark.

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
