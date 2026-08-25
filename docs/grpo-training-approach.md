# SpatialMap GRPO: approach, status, and discussion points

Lab-meeting note. Headline result is still **SpatialMap-TQA** (static, extrinsic, text-only). GRPO is a **post-SFT** experiment: can format/outcome rewards push the SFT model past 87% on the **1329 cleaned** SpatialMap set without changing the eval protocol.

**Do not present the 20-step adapter as a SpatialMap gain.** Present it as: the RL loop is no longer degenerate.

---

## 1. What the project is actually about

**Question.** Can an 8B reasoning model solve SpatialEval **SpatialMap TQA** from text alone (no vision, no agent loop) if we (a) clean the gold, (b) SFT on synthetic axis-graph traces, and optionally (c) RL on outcome?

**Eval that matters (keep this number stable):**

| Model | Set | Protocol | Strict acc |
|---|---|---|---|
| Base DeepSeek-R1-0528-Qwen3-8B | 1329 cleaned SpatialMap | two-pass vLLM, unique-letter gold | **71%** |
| SFT QLoRA (same base) | same 1329 | same | **87%** |
| GRPO on SFT | same 1329 | same | **not measured yet** |

**Not the headline:** `results.json` (~28% strict / ~75% loose on 1500 *generated* SpatialMap) is a different protocol. Do not mix it with 71/87.

**Cleaned gold.** Offline SpatialMap JSONL is 1500 rows. Graph-closure oracle leaves **171 empty** (undetermined). We **drop** those and score **1329**. Mix is roughly Type0 direction / Type1 which-object / Type2 count; golds are 1-, 2-, or 4-letter sets.

**SFT, in one line.** QLoRA on DeepSeek-R1-0528-Qwen3-8B; synthetic traces that build X/Y axis graphs and end `</think>\nAnswer: A` (or `A, C`). Mean completion ~708 tokens. Train mix was 1500 synthetic (1-answer 600, 2-answer 600, 4-answer 300). Original adapter: `finetune/outputs/deepseek-r1-qwen3-8b` — **keep it**; SFT eval depends on it.

---

## 2. Why GRPO after SFT (the scientific bet)

SFT already teaches the *procedure* (axis graph + final `Answer:` line). The remaining ~13% (173 misses vs 87%) is not “the model has never seen CoT.” GRPO is only justified if:

1. Groups have **reward variance** (some of G samples better than others).
2. Rewards match the **eval** (exact letter set + parseable `Answer:`).
3. We do **not** RL on gold traces — only on **prompts + oracle letters**.

If (1) fails, GRPO is a no-op (`frac_reward_zero_std = 1`, advantages 0). That is what the first long run was.

**What GRPO cannot magically do.** It will not invent a new spatial calculus. It can (a) make the model emit the SFT last line more often, (b) prefer completions whose letter set matches gold, (c) mildly shorten traces. 87% → 90% is a **hypothesis**, not a result.

---

## 3. Method (what to put on a slide)

```
cleaned SpatialMap eval (1329)          ← frozen protocol
        ▲
SFT QLoRA  →  merge to bf16 (sibling dir)  →  new GRPO LoRA
        │                                      ▲
        │         prompt-only synthetic JSONL  │
        └────────  G samples from policy  ─────┘
                   rewards: outcome / format / length
```

**Algorithm.** Group Relative Policy Optimization (TRL/Axolotl `rl: grpo`). For each prompt, sample **G** completions, score them, **normalize advantages inside the group**, PPO-style clip, small KL to the SFT policy.

**Why not PPO with a critic.** No value head; group baseline is enough for a verifiable letter-set reward.

### 3.1 Data (prompt-only)

Generator: `finetune/generate_grpo.py` wrapping `generate_all.py`.

- Conversational `prompt` + `oracle_option` (`A` / `A,C`). **No gold CoT.**
- Mix aligned to cleaned SpatialMap (~32% direction, ~38% which-object, ~30% count), not the SFT 40/40/20 uncertainty mix.
- Every user turn is annotated with: close `</think>` and a last line `Answer: A` or `Answer: A, C`.
- Full pool 4000 rows; the 3h H200 run uses a **2000-row** subset (`spatial_grpo_data_2k.jsonl`).

### 3.2 Rewards (`finetune/rewards.py`)

Weighted sum in yaml: **1.0 / 0.1 / 0.1**.

| Signal | Rule | Weight |
|---|---|---|
| **outcome** | Predicted `{A–D}` set **equals** oracle set | 1.0 |
| **format** | A parseable `Answer: A[,B…]` exists | 0.1 |
| **length** | 0 until 1024 tokens, then linear to −1 at 2048 | 0.1 |

Truncated completions (`max_completion_length` 1536) are **masked** (`mask_truncated_completions: true`) so hitting the cap does not become a fake “wrong” gradient.

### 3.3 Optimization (working recipe)

| Knob | Value | Why |
|---|---|---|
| Base for RL | **Merged SFT bf16**, not HF base + SFT QLoRA | vLLM cannot reload 4-bit LoRA into fused Qwen3 weights |
| New adapter | LoRA r=64, α=128, same targets as SFT | SFT adapter **never overwritten** |
| G | 4 (probe and H200) | 8 was the original wish; 4 fits time/memory |
| clip ε | 0.2 | standard |
| KL β | 0.04 | stay near SFT |
| lr | 1e-5, cosine | small; SFT already strong |
| temperature | 0.7 | some group diversity |
| prompt / completion | 1024 / 1536 | SFT traces ~708; 1536 is a hard cap |
| vLLM | server, LoRA sync, `max_model_len` **3072** | native context is 131072; that OOMs KV |

**Split of GPUs.** GPU 0: vLLM serve (rollouts). GPU 1: trainer. They do **not** overlap compute; the second GPU is so bf16 serve and LoRA train both fit. Current 2000-prompt job is **2× H200 141GB** (`h200-141:2` on `xgpk*`). H100 96GB is the same yaml if those queue first. Extra VRAM is headroom; **do not** raise `max_model_len`.

---

## 4. How GRPO works in this run (say this if they ask “what is a rollout?”)

**SFT** showed gold scratch work. **GRPO** does not. It only knows the gold **letters**. It writes several answers to the same question, keeps the better ones, and is taxed for leaving SFT.

### 4.1 One step (H200 yaml)

1. Take **8 questions** from the 2000-row file (prompt + gold letters `oracle_option`).
2. vLLM writes **4 answers per question** = **32 writings**. Temperature 0.7. Cap **1536** tokens; if it hits the cap, that writing is **masked** (does not train).
3. Score each writing (outcome / format / length).
4. For each question, compare its **4 scores** → **advantages**.
5. Trainer updates the **new LoRA**. SFT adapter on disk is never touched.
6. Sync LoRA back to vLLM.

250 steps × 8 questions ≈ one pass over 2000 (`num_epochs: 1`). `epoch: 0.084` at step 20 ≈ 8% of the file.

**Completion / sample / rollout** = one of those 32 writings. Same object, three names.

### 4.2 Scores

Weighted sum: `1.0×outcome + 0.1×format + 0.1×length`.

| Log name | English |
|---|---|
| **outcome** | Letter set **exactly** equals gold (`A` vs `A,C` is a set). 1 or 0. |
| **format** | There is a parseable `Answer: A` (or `A, C`). 1 or 0. |
| **length** | 0 if ≤1024 tokens; then negative, down to −1 at 2048. |

`outcome_reward/mean: 0.34` = **11 of 32 writings** got the letters right. That is **not** the 87% SpatialMap number (greedy two-pass on 1329). If outcome mean **equals** format mean, misses are “never printed `Answer:`,” not “printed the wrong letters.”

### 4.3 The group (this *is* GRPO)

Four rewards for one question, e.g. `[1.1, 0, 0, −0.1]`. No critic. The other three writings are the baseline: subtract the mean, divide by the std. That is **advantage**.

- **positive** = better than its three siblings → do more of this  
- **negative** = worse → do less  
- **zero / tied** = this question does **not** train this step  

**`frac_reward_zero_std`** = fraction of questions whose 4 scores were identical. **1.0** = dead run. **0.5** = half the questions taught nothing. **You want this to fall.**

A printed Advantage of **+0.83 can still be a wrong answer** — it only means “best of these four.” **−1.50** means the other writings on that question were better.

### 4.4 Clip and KL

- **Clip (`epsilon: 0.2`).** Do not change any token’s probability by more than ~20% in one step. `clip_ratio/*` = 0 means updates are still tiny.
- **KL (`beta: 0.04`).** Tax for drifting from SFT. Logged `kl` **should creep up** if you learn (0.004 → maybe 0.02–0.1). Still ~0.004 = still SFT. Spike to 0.5+ = ran away.

**Loss** is this surrogate, not SFT cross-entropy. It can be negative. Do not optimize for “loss going down.”

### 4.5 How to read a log line

| Field | English |
|---|---|
| `grad_norm` | Size of the LoRA nudge. Small but nonzero = graph is alive. |
| `learning_rate` | ~1e-5 early in the 250-step cosine. |
| `completions/mean_length` | Average tokens per writing. |
| `clipped_ratio` | Fraction that hit 1536 and were masked. |
| `*_terminated_length` | Same, only among writings that stopped themselves. |
| `reward` / `reward_std` | Weighted score, mean and spread over the 32. Need spread **inside** a group. |
| `entropy` | How peaked next-token is. Very low ≈ little variety ≈ more tied groups. |
| `importance_sampling_ratio` / `sampling_logp_difference` | vLLM vs trainer disagree on token probabilities. Trust **outcome / format / frac_zero_std**, not these. |

**Over the H200 run you want:** outcome mean **up**, `frac_reward_zero_std` **down**, `kl` **up a little**, clipped_ratio not exploding. Then one 1329 eval vs **87%**.

### 4.6 Term cheat sheet

| They say | They mean here |
|---|---|
| Prompt | The question (system + user). No gold scratch work. |
| Completion / sample / rollout | One model writing. |
| Group / G=4 | The 4 writings of **one** question. |
| Advantage | Better/worse **than the other 3**, not “correct.” |
| Reference / SFT policy | Merged SFT; GRPO must not forget it. |
| Policy | Current GRPO LoRA + that SFT base. |
| Gold / oracle | Letter set in the JSONL. |
| Masked | Hit 1536 tokens; dropped from the loss. |
| Two-pass eval | **Not** GRPO. Think, then force `Answer:`. That is 71/87. |

---

## 5. Engineering path (only as much as the room needs)

Say this only if asked “why did this take so long.”

1. **Axolotl + TRL GRPO** is the trainer (Axolotl ≥0.18, TRL 1.8).
2. **1-GPU QLoRA + colocated vLLM** died: 4-bit tensors vs vLLM fused shapes.
3. **Merge** SFT QLoRA into `…/deepseek-r1-qwen3-8b-merged`. Merge **refuses** to write inside the adapter dir.
4. **KV:** if `max_model_len` is omitted, vLLM uses 131072 (~18 GiB KV) and dies. We force **3072** in yaml and patch TRL.
5. Cluster **`h100-47` is MIG 3g.47gb**, two slices on one H100 NVL. Pin **CUDA 0/1**, not `MIG-` UUIDs (vLLM does `int(CUDA_VISIBLE_DEVICES)`).
6. **`h200-141` and `h100-96` are full cards**, not MIG. Same 0/1 pin. `xgpk0` has four H200s; `--gres=gpu:h200-141:2` takes two of them.
7. Torch dynamo `has_triton` IndexError on `CUDA_VISIBLE_DEVICES=1` is patched in `finetune.py` before importing Axolotl.

None of this is the research contribution. It is why the probe could run at all.

---

## 6. What we have actually run

### 6.1 Failed / uninformative

- Early GRPO with **all rewards 0** for hundreds of steps (`frac_reward_zero_std = 1`). Cause: SFT last line never sampled, so format and outcome never fired. Fix: annotate prompts; cap 1536; **do not resume** that run.

### 6.2 20-step probe (done) — this is the slide

**Setup.** 2× MIG 47GB, vLLM + train, **20 steps**, ~**160 unique prompts**, ~**640 completions**. ~32 min (`train_runtime` 1901 s). Saved under `finetune/outputs/deepseek-r1-qwen3-8b-grpo-vllm`.

**A logged step (representative):**

| Metric | Value | Reading |
|---|---|---|
| `rewards/outcome_reward/mean` | 0.1875 | 6/32 completions exact-match |
| `rewards/format_reward/mean` | 0.1875 | same; format and outcome moved **together** |
| `frac_reward_zero_std` | 0.5 | half the groups have a contrast (was 1.0) |
| `kl` | ~0.003 | policy barely left SFT |
| `completions/mean_length` | ~1035 | still long CoT |
| `clipped_ratio` | 0.22 | 22% hit 1536 and were masked |
| `entropy` | ~0.005 | little exploration |
| `grad_norm` | ~0.013 | tiny but nonzero |
| `train_loss` | −0.033 | GRPO surrogate; negative is OK |

**End-of-run print** showed mixed groups: one sample outcome=1 / format=1, sibling 0/0, advantage ~0.87 vs 0.

**Interpretation to say out loud.**

- The loop is **alive**. Last time GRPO could not learn; now it can.
- Bottleneck on this batch is still **emitting `Answer:`**, not “formats but picks the wrong letter” (format mean = outcome mean).
- **20 steps will not move 87%.** Do not eval this adapter as the GRPO paper number.

### 6.3 2000-prompt H200-141×2 run (this is the train job)

Submit **`run_grpo_h200.sh`** (not `run_grpo_h100_96.sh` unless 96s are the only ones free). Do not queue both for the same run.

- `--gres=gpu:h200-141:2`, 3h, same yaml `qwen3-8b-spatial-grpo-vllm-h100.yaml`
- ~2000 prompts, 8 unique/step, **max_steps 250**, checkpoints every 10, `--resume` on resubmit
- Output dir still `…-grpo-h100` (does not clobber SFT or the 20-step LoRA)
- KV still **3072**, utilization **0.70** — 141GB does **not** mean a longer context
- **No 1329 eval inside this job**
- Fallback: `run_grpo_h100_96.sh` is the same recipe on 2× H100 96GB

### 6.4 1329 eval (separate job)

`run_eval_grpo_1329.sh` — one GPU, two-pass vLLM, task `spatial_eval_gen_cleaned_1329` (drop empty oracles). **Base = merged SFT**, **LoRA = GRPO adapter**. Same system prompt / `strict_acc` as the 87% SFT yaml.

Compare **that** number to 87%. Anything else is a different experiment.

---

## 7. Claims vs non-claims

**Can say**

- Post-SFT GRPO with verifiable SpatialMap rewards is implemented (outcome / format / length).
- Prompt-only data + last-line annotation makes format/outcome **non-degenerate**.
- On a 20-step probe, ~19% of rollouts exact-match and half of groups have nonzero advantage.
- SFT weights are intact; GRPO is a new LoRA on a merged copy.

**Cannot say yet**

- GRPO improves SpatialMap (no 1329 number).
- We will hit 90%.
- 20 steps / 160 prompts is a trained RL model.
- vLLM two-GPU is “2× faster because of parallelism” (decode is faster than HF generate; the two GPUs are sequential).

---

## 8. Things to be ready to discuss

**Q. Why RL if SFT is already 87%?**  
SFT fits traces. RL can reinforce *getting the letter set right* on prompts without cloning a single gold CoT. Only useful if groups are mixed. They were not, then they were.

**Q. Is the reward the same as eval?**  
Almost. Eval is two-pass (free think, then constrained `Answer:`). GRPO is single-sample until 1536, regex on `Answer:`. Distribution shift is real. If 1329 does not move, this is a prime suspect.

**Q. Why G=4 not 8?**  
Time/memory on 3h slices. G=8 would estimate advantages better and cost ~2× decode.

**Q. Why not RL from base 71%?**  
We already have a competent SFT policy. GRPO from a 71% model would spend budget relearning format that SFT already has.

**Q. Data leakage?**  
GRPO prompts are **synthetic** (`generate_all`), not the 1329 eval items. Oracle letters come from the same generator rules as SFT gold, not from the cleaned eval JSONL.

**Q. Why merge?**  
QLoRA 4-bit ≠ vLLM fused weights. Merge is an **infra** step, not a training trick. Eval of GRPO must load **merged SFT + GRPO LoRA**, never HF base + GRPO LoRA.

**Q. Length penalty vs SFT CoT?**  
SFT was trained to write long graphs. Penalty only starts at 1024 (above mean 708). Probe still averaged 1035 with 22% clipped — the cap is binding. If we lower 1536, we may cut CoT that SFT needs.

**Q. What would falsify GRPO here?**  
A longer run where format/outcome stay ~0.2, `frac_reward_zero_std` stays high, and 1329 is ≤87% within noise. Then the honest story is: SFT did the work; outcome RL on this recipe does not.

---

## 9. Suggested talk order (10–12 min)

1. SpatialMap-TQA, cleaned 1329, 71 → 87. That is the paper spine.
2. Residual errors: not “no CoT,” often format/termination and hard multi-letter items.
3. GRPO as **optional** follow-up: prompt-only, G samples, three rewards.
4. Degenerate run (all zero) → annotation fix → 20-step probe (table in §6.2).
5. Next: 2000-prompt / ~250-step job on **2× H200**, then **one** 1329 eval vs 87%.
6. What we will not claim until that number exists.

---

## 10. File map (if someone asks where it lives)

| Piece | Path |
|---|---|
| SFT adapter (do not delete) | `finetune/outputs/deepseek-r1-qwen3-8b` |
| Merged SFT (vLLM / GRPO base) | `finetune/outputs/deepseek-r1-qwen3-8b-merged` |
| 20-step GRPO LoRA | `finetune/outputs/deepseek-r1-qwen3-8b-grpo-vllm` |
| 2000-prompt GRPO LoRA | `finetune/outputs/deepseek-r1-qwen3-8b-grpo-h100` |
| Rewards | `finetune/rewards.py` |
| Prompt-only data gen | `finetune/generate_grpo.py` |
| Probe yaml | `finetune/config/qwen3-8b-spatial-grpo-vllm.yaml` |
| 2000-prompt yaml (H200 or H100) | `finetune/config/qwen3-8b-spatial-grpo-vllm-h100.yaml` |
| Train **2× H200 141GB** (preferred) | `run_grpo_h200.sh` |
| Train 2× H100 96GB (fallback) | `run_grpo_h100_96.sh` |
| 1329 eval | `run_eval_grpo_1329.sh`, `eval/config/Deepseek-R1-Qwen_8B_GRPO.yaml` |
| Eval entrypoint | `eval/eval_two_stage.py` (`vllm_two_pass`) |
