# Experiment 5 — GRPO (post-SFT)

**Not in the dissertation.** Post-thesis follow-up on top of Exp 3's SFT
adapter. Numbering skips 4 deliberately so 0–3 stay aligned with H036660 Ch. 4.

**Question.** Can GRPO on outcome/format rewards push the SFT model past
~87% on the 1329 cleaned SpatialMap set without changing the eval protocol?

**Status: inconclusive / not very useful.** The 20-step probe showed the RL
loop is no longer degenerate (reward variance > 0), but no measured accuracy
gain over SFT was achieved. **Do not present the probe as a SpatialMap gain.**
Full discussion: `../../docs/grpo-training-approach.md`.

**Setup.** GRPO on the SFT QLoRA (or on the merged SFT weights for the
vLLM-serve variant). Prompts only; `oracle_option` used for rewards, not fed
to the model. Rewards: `rewards.outcome_reward`, `format_reward`,
`length_penalty` (weights 1.0 / 0.1 / 0.1).

## Training configs (run from `cd finetune`)

| config | what |
|---|---|
| `train-grpo-8b.yaml` | GRPO on the SFT base, single-GPU HF generate |
| `train-grpo-8b-probe.yaml` | 20-step probe (proved the loop is non-degenerate) |
| `train-grpo-8b-vllm.yaml` | GRPO on merged SFT, 2-GPU vLLM serve + train (MIG slices) |
| `train-grpo-8b-vllm-h100.yaml` | same recipe, full H100-96/H200-141 (2000 prompts) |

The GRPO configs expect merged SFT weights at
`experiments/05-grpo/models/deepseek-r1-qwen3-8b-merged`; the launchers
(`run_grpo_*.sh`) create it via `merge_sft.py` if missing.

## Eval config (run from `cd eval`)

| config | what |
|---|---|
| `eval-grpo-1329.yaml` | GRPO LoRA on merged SFT, 1329 cleaned set |

Launcher: `run_eval_grpo_1329.sh`.

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
