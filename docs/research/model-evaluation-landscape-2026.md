# Model evaluation landscape for SpatialEntail

*Primary-source review, checked 18 October 2026.*

## Decision

Keep **`Qwen/Qwen3.5-4B` as the only SFT base**. P0 is the prompt-matched
untuned checkpoint plus the six fixed SFT adapters. The completed untuned run
uses Qwen's native thinking path followed by the common constrained-answer
pass; `untuned` describes the weights, not a non-thinking inference mode. A
separate Qwen3.5-4B thinking-off run is therefore still required if that
response-regime control is reported.

Use every other checkpoint only as an untuned evaluation comparator. The
recommended P1 panel is deliberately bounded to checkpoints at or below about
9B stored parameters:

1. `Qwen/Qwen3.5-9B`, in thinking-off and thinking-on modes, as the clean
   within-family scale and reasoning control;
2. `google/gemma-4-E4B-it`, or Google's separately labelled
   `google/gemma-4-E4B-it-qat-w4a16-ct` deployment checkpoint;
3. `google/gemma-3-4b-it`, as an instruction-only historical Gemma control;
4. `mistralai/Ministral-3-8B-Instruct-2512`;
5. `microsoft/Phi-4-mini-instruct`;
6. `mistralai/Ministral-3-8B-Reasoning-2512`; and
7. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.

Models above this size range, including `openai/gpt-oss-20b` and
`Qwen/Qwen3-30B-A3B-Thinking-2507`, are excluded from the FYP experiment plan,
not deferred experiment candidates.

This comparator set separates three questions that a single leaderboard would confound:
whether SFT beats its own untuned 4B starting point, whether ordinary scale
from 4B to 9B explains the gain, and whether test-time reasoning explains it.
It also stays within the available single-GPU profiles at the repository's
8,192-token frozen evaluation cap. Closed APIs should be reported only as a
separate ceiling because they cannot reproduce the local two-pass vLLM decoder
exactly.

## Scope and interpretation

The executable study is capped at the agreed approximately <=9B model class.
Here that operational label refers to the published model class and includes
the exact shortlisted checkpoints above, even where embeddings or vision
tensors make the stored checkpoint slightly larger than the name. Larger
models may remain in the landscape review as literature context, but they must
not be prepared, submitted, or reported as SpatialEntail experiments.

“Fit” below means **single-sequence inference at `max_model_len=8,192`**, not
serving the model's advertised maximum context. It assumes the official
checkpoint precision, batch size one where necessary, and a recent compatible
vLLM. The estimate is intentionally conservative:

- BF16 weights require about two bytes per stored parameter before runtime,
  activations, kernels, and KV cache;
- FP8 weights require roughly one byte per stored parameter, subject to
  unquantized modules and runtime overhead;
- every MoE expert must reside in memory even though only a subset is active per
  token; and
- `2x47` means tensor parallelism across two 47 GB devices, not a fictitious
  contiguous 94 GB device.

The frozen randomized-v2 protocol uses an 8,192-token evaluation budget with up
to 4,096 new tokens. Its pinned tokenizer audit found a 2,513-token maximum
complete training sequence. Thus all shortlisted contexts are sufficient;
advertised 128K–1M context is not a reason to prefer one model here.

## Current open instruction baselines, 3B–8B

These are the most useful same-scale untuned controls. Parameters are
**total/active**. For multimodal-native checkpoints, both the language-model
size and the complete checkpoint tensor count are shown because the latter
governs memory unless the server can omit the vision tower.

| Exact model ID | Parameters | Declared context | License and access | HF/vLLM status | Hardware fit at 8K |
|---|---:|---:|---|---|---|
| [`Qwen/Qwen3.5-4B`][q35-4] | 4B LM, 4.660B checkpoint / dense | 262,144 native; 1,010,000 extended | Apache-2.0; ungated | Official Transformers, SGLang, and vLLM support; the card currently asks for vLLM main/nightly and documents `--language-model-only` | Easy on 1x47 BF16; also the **only SFT base** |
| [`mistralai/Ministral-3-3B-Instruct-2512`][ministral-3-i] | 3.4B LM + about 0.4B vision, 3.849B checkpoint / dense | 262,144 | Apache-2.0; ungated | Official FP8 checkpoint; vLLM >=0.12 and Transformers documented | Official card says 8 GB VRAM in FP8; easy on 1x47 |
| [`microsoft/Phi-4-mini-instruct`][phi-mini] | 3.836B / dense | 131,072 | MIT; ungated | Transformers >=4.49 and vLLM >=0.7.3 documented | About 7.7 GB BF16 weights; easy on 1x47 |
| [`mistralai/Ministral-3-8B-Instruct-2512`][ministral-8-i] | 8.4B LM + about 0.4B vision, 8.918B checkpoint / dense | 262,144 | Apache-2.0; ungated | Official FP8 checkpoint; vLLM >=0.12 and Transformers documented | Official card says 12 GB VRAM in FP8; easy on 1x47 |
| [`google/gemma-3-4b-it`][gemma3-4] | 4.3B / dense | 131,072 | Gemma license; manual HF access gate | Official Transformers and vLLM support | Easy on 1x47 BF16; instruction-only historical control |
| [`google/gemma-4-E2B-it`][gemma4-e2] | 2.3B effective; 5.1B with embeddings, 5.123B checkpoint / dense | 131,072 | Apache-2.0; ungated | Official card documents current Transformers; Google also publishes a separate vLLM/SGLang QAT checkpoint | Google estimate: 11.4 GB BF16 including 20% load overhead; easy on 1x47 |
| [`google/gemma-4-E4B-it`][gemma4-e4] | 4.5B effective; 8.0B with embeddings, 7.996B checkpoint / dense | 131,072 | Apache-2.0; ungated | Official card documents current Transformers; Google also publishes a separate vLLM/SGLang QAT checkpoint | Google estimate: 17.9 GB BF16 including 20% load overhead; easy on 1x47 |

Gemma 4 materially changes the choice here: E2B and E4B are both below 9B in
stored parameters, ungated, and Apache-2.0. `google/gemma-3-4b-it` is retained
only as the historical Gemma instruction control; it receives no invented
thinking condition. `meta-llama/Llama-3.1-8B-Instruct` remains a gated
historical anchor rather than the newest Meta option.

### Verified Gemma 4 release family

Google has officially released Gemma 4. The owning `google` Hugging Face
organization, the common Google DeepMind model card, and the Gemma 4 technical
report [agree on five instruction-tuned sizes][gemma4-report]. These are real releases, not
search-result or community pages.

| Exact official ID | Model/checkpoint parameters | Context | License/access | Fit at 8K |
|---|---:|---:|---|---|
| [`google/gemma-4-E2B-it`][gemma4-e2] | 2.3B effective; 5.1B including PLE embeddings; 5.123B HF tensors | 128K | Apache-2.0; ungated | Google estimate 11.4 GB BF16; 1x47 |
| [`google/gemma-4-E4B-it`][gemma4-e4] | 4.5B effective; 8.0B including PLE embeddings; 7.996B HF tensors | 128K | Apache-2.0; ungated | Google estimate 17.9 GB BF16; 1x47 |
| [`google/gemma-4-12B-it`][gemma4-12] | 11.95B model; 11.960B HF tensors | 256K | Apache-2.0; ungated | Google estimate 26.7 GB BF16; 1x47 at the frozen 8K cap |
| [`google/gemma-4-26B-A4B-it`][gemma4-26] | 25.2B model / 3.8B active MoE; 25.806B HF tensors including the vision encoder | 256K | Apache-2.0; ungated | Google estimate 57.7 GB BF16; not 1x47, but 1x96, 1x141, or 2x47 |
| [`google/gemma-4-31B-it`][gemma4-31] | 30.7B model; 31.273B HF tensors including the vision encoder / dense | 256K | Apache-2.0; ungated | Google estimate 69.9 GB BF16; not 1x47, but 1x96, 1x141, or 2x47 |

All five canonical cards document Hugging Face Transformers. Google additionally
publishes `google/gemma-4-E2B-it-qat-w4a16-ct`,
`google/gemma-4-E4B-it-qat-w4a16-ct`,
`google/gemma-4-12B-it-qat-w4a16-ct`, and
`google/gemma-4-31B-it-qat-w4a16-ct` explicitly for vLLM/SGLang; no equivalent
26B A4B variant is listed. Canonical BF16 should still pass the repository's
staged-pass smoke test before entering the frozen matrix, and any QAT result
must be labelled as a different precision checkpoint. Google's memory numbers
include 20% load overhead but exclude runtime software and KV cache.

Community models whose names contain “Gemma 4” are not evidence of release;
only exact `google/gemma-4-*` IDs are treated as official. In particular,
`google/gemma-4-4b-it` and `google/gemma-4-27b-it` do not exist; the official
names are E4B, 12B, 26B-A4B, and 31B.

## Stronger open models, 9B–32B

| Exact model ID | Parameters | Declared context | License and access | HF/vLLM status | Hardware fit at 8K |
|---|---:|---:|---|---|---|
| [`Qwen/Qwen3.5-9B`][q35-9] | 9B LM, 9.653B checkpoint / dense | 262,144 native; 1,010,000 extended | Apache-2.0; ungated | Same current Qwen3.5 Transformers/vLLM-nightly path as 4B; text-only loading documented | About 19.3 GB BF16 weights; easy on 1x47 |
| [`mistralai/Ministral-3-14B-Instruct-2512`][ministral-14-i] | 13.5B LM + about 0.4B vision, 13.945B checkpoint / dense | 262,144 | Apache-2.0; ungated | Official FP8 checkpoint; vLLM >=0.12 and Transformers documented | Official card says 24 GB VRAM in FP8; easy on 1x47 |
| [`mistralai/Mistral-Small-3.2-24B-Instruct-2506`][mistral-small] | 24.011B / dense | 131,072 | Apache-2.0; ungated | vLLM >=0.9.1 recommended; Transformers supported | Card states about 55 GB in BF16: no on 1x47, yes on 1x96, 1x141, or 2x47 |
| [`Qwen/Qwen3.5-27B`][q35-27] | 27B LM, 27.781B checkpoint / dense | 262,144 native; 1,010,000 extended | Apache-2.0; ungated | Transformers and current vLLM-nightly supported; official full-context example uses TP=8, but 16K does not require that | About 55.6 GB BF16 weights: no on 1x47; yes on 1x96, 1x141, or 2x47 at 16K |
| [`google/gemma-3-27b-it`][gemma3-27] | 27.432B / dense | 131,072 | Gemma license; manual HF access gate | Transformers and vLLM support exist after accepting terms | About 54.9 GB BF16 weights: no on 1x47; yes on 1x96, 1x141, or 2x47 at 16K |
| [`Qwen/Qwen2.5-32B-Instruct`][q25-32] | 32.764B / dense | 32,768 configured; 131,072 with the card's YaRN recipe | Apache-2.0; ungated | Mature Transformers and vLLM path | About 65.5 GB BF16 weights: no on 1x47; yes on 1x96, 1x141, or 2x47 at 16K |

`Qwen3.5-9B` is the highest-value row: it changes scale while holding model
family, release, prompt template, and thinking switch nearly constant.
`Qwen3.5-27B` is the next scale point but belongs in P2 because the 9B result
should determine whether another dense family point is worth its cost.

### Verified Meta release boundary

Meta [officially released Llama 4 Scout and Maverick][meta-llama4], but neither is a native-
precision fit for this study's hardware. The newest small, pure-text Meta
checkpoint that fits without ad-hoc quantization is Llama 3.2 3B.

| Exact official ID | Parameters/context | License/access and serving | Hardware conclusion |
|---|---|---|---|
| [`meta-llama/Llama-4-Scout-17B-16E-Instruct`][llama4-scout] | 108.642B stored / 17B active MoE; 10M context | Llama 4 Community License; manual gate; official HF page documents Transformers >=4.51 and vLLM | BF16 is about 217 GB and fits none of 1x47/1x96/1x141/2x47. Meta says Scout fits one H100 only with INT4; that is a separate quantized experiment, not the official BF16 baseline. |
| [`meta-llama/Llama-4-Maverick-17B-128E-Instruct`][llama4-maverick] | 401.584B stored / 17B active MoE; 1M context | Llama 4 Community License; manual gate; official HF page documents Transformers >=4.51 and vLLM | Far outside scope. Meta says one H100 **host**, not one GPU; it does not fit any listed profile. |
| [`meta-llama/Llama-3.3-70B-Instruct`][llama33-70] | 70.554B dense; 128K context | Llama 3.3 Community License; manual gate; official Transformers checkpoint | Newest Meta text-only family, but about 141 GB of BF16 weights leaves no runtime/KV headroom even on 1x141; exclude without separately labelled quantization. |
| [`meta-llama/Llama-3.2-3B-Instruct`][llama32-3] | 3.213B dense; 128K context | Llama 3.2 Community License; manual gate; official Transformers checkpoint | Easy on 1x47 BF16; newest officially released Meta checkpoint below 9B suitable for native-weight pure-text evaluation. |
| [`meta-llama/Llama-3.1-8B-Instruct`][llama31-8] | 8.030B dense; 128K context | Llama 3.1 Community License; manual gate; mature HF/vLLM | Easy on 1x47 BF16; better literature anchor, but older than Llama 3.2. |

No later general-purpose Llama weight release appears in Meta's official HF
organization as of the cutoff. Llama 4 Behemoth was described by Meta as still
training, not released. Any `Llama-5`, post-Llama-4 community namespace, or
download page not traceable to Meta AI or `meta-llama/*` is unverified and must
not enter the evaluation plan.

## Open reasoning-specialized models

| Exact model ID | Parameters | Declared context | License and access | HF/vLLM status | Hardware fit at 8K |
|---|---:|---:|---|---|---|
| [`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`][nemotron-4] | 3.973B / dense hybrid Mamba–Transformer | 262,144 | NVIDIA Nemotron Open Model License; ungated | Transformers and vLLM >=0.15.1; custom official reasoning parser required | About 8.0 GB BF16 weights; easy on 1x47 |
| [`mistralai/Ministral-3-8B-Reasoning-2512`][ministral-8-r] | 8.4B LM + about 0.4B vision, 8.918B checkpoint / dense | 262,144 | Apache-2.0; ungated | vLLM >=0.12 with `reasoning-parser=mistral`; Transformers documented | Official card says BF16 fits 24 GB; easy on 1x47 |
| [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`][ds-r1-7] | 7.616B / dense | 131,072 configured | MIT final weights; Qwen-derived base was Apache-2.0; ungated | Official DeepSeek card gives vLLM/SGLang serving; modern Transformers also loads the Qwen2 architecture | About 15.2 GB BF16 weights; easy on 1x47 |
| [`openai/gpt-oss-20b`][gpt-oss] | 21B total / 3.6B active MoE | 131,072 | Apache-2.0; ungated | Official Transformers and vLLM paths; must preserve Harmony formatting | Official MXFP4 checkpoint runs within 16 GB; easy on 1x47 |
| [`microsoft/Phi-4-reasoning-plus`][phi-reasoning] | 14.660B / dense | 32,768 | MIT; ungated | Official Transformers and vLLM `deepseek_r1` reasoning-parser command | About 29.3 GB BF16 weights; fits 1x47 at 16K/batch 1; all larger profiles are comfortable |
| [`Qwen/Qwen3-30B-A3B-Thinking-2507`][q3-thinking] | 30.5B total / 3.3B active MoE | 262,144 native | Apache-2.0; ungated | Transformers, vLLM >=0.8.5, and SGLang officially supported | About 61.1 GB BF16 weights: no on 1x47; yes on 1x96, 1x141, or 2x47 at 16K |
| [`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`][ds-r1-32] | 32.764B / dense | 131,072 configured | MIT final weights; Qwen-derived base was Apache-2.0; ungated | Official card demonstrates vLLM TP=2 with a 32,768 cap | About 65.5 GB BF16 weights: no on 1x47; yes on 1x96, 1x141, or 2x47 at 16K |

Two distinctions matter. First, Qwen3.5 itself is hybrid-mode: thinking is on by
default and `enable_thinking=False` gives an instruction-style response. That
makes paired thinking/non-thinking runs on one checkpoint more informative than
calling it only an “instruction model.” Second, active parameters describe MoE
compute, not weight residency. `gpt-oss-20b` fits a 47 GB GPU because its
official checkpoint is MXFP4, whereas the 30.5B Qwen MoE does not fit there in
BF16 merely because 3.3B parameters activate per token.

## What recent spatial and reasoning papers actually evaluate

The model choices above are not based only on general benchmark rankings.
Recent primary spatial sources reveal a stable set of comparison roles.

| Source | Models used | Implication for SpatialEntail |
|---|---|---|
| [MentalMap, 2026][mentalmap] | Closed: GPT-4o, Gemini-2.5-Flash, DeepSeek-V4-Flash, Qwen3.5-Flash. Open 7–10B: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `google/gemma-2-9b-it`, `tiiuae/Falcon3-10B-Instruct`. Scale controls: `Qwen/Qwen2.5-32B-Instruct`, `google/gemma-3-27b-it`. | Confirms the conventional 7–10B instruction panel and the value of a separate 27–32B scale control. It also reports that chain-of-thought can hurt strict structured output, supporting separate direct and reasoning tracks rather than one mixed ranking. |
| [SiT-Bench, 2026][sitbench] | General backbones include Qwen2.5/3 and Llama 3.1; specialist baselines include Space-Qwen, SpaceThinker, RoboBrain2.0, SpaceR, and Cosmos-Reason2; proprietary/large models include GPT-4o, Gemini-3-Flash, and DeepSeek-V3.2. | Supports reporting spatial specialists separately from general instruction and reasoning models. The final panel keeps complementary specialist roles rather than treating every spatially named checkpoint as useful. |
| [`interwhen`, 2026][interwhen] | Exact open reasoning models: `Qwen/Qwen3-30B-A3B-Thinking-2507`, `Qwen/QwQ-32B`, and `microsoft/phi-4-reasoning`; all served with vLLM. SpatialMap is one of its main 1,500-example tasks. | Provides literature context for reasoning evaluation, but its checkpoints are excluded by this FYP's approximately 9B model ceiling. |
| [SODA/SPOD-Bench, ACL 2026][soda] | Builds a pure-text spatial benchmark and trains with SFT followed by GRPO. | Confirms that training and evaluation checkpoints must be separated in reporting. It does not make its training base an automatic comparator for SpatialEntail's static entailment semantics. |

For reproducibility, the recurring historical IDs have the following verified
deployment envelope:

| Exact ID | Parameters; context | License/access; vLLM | Fit at 8K |
|---|---|---|---|
| [`meta-llama/Llama-3.1-8B-Instruct`][llama31-8] | 8.030B dense; 128K | Llama 3.1 community license, manual gate; mature HF/vLLM | 1x47 BF16 |
| [`Qwen/Qwen2.5-7B-Instruct`][q25-7] | 7.616B dense; 32K configured, 131K with YaRN | Apache-2.0, ungated; mature HF/vLLM | 1x47 BF16 |
| [`mistralai/Mistral-7B-Instruct-v0.3`][mistral7] | 7.248B dense; 32K | Apache-2.0, ungated; official vLLM card | 1x47 BF16 |
| [`google/gemma-2-9b-it`][gemma2-9] | 9.242B dense; 8K | Gemma license, manual gate; HF/vLLM after access | 1x47 BF16 |
| [`tiiuae/Falcon3-10B-Instruct`][falcon3] | 10.306B dense; 32K | Falcon license, ungated; HF/vLLM | 1x47 BF16 |
| [`Qwen/QwQ-32B`][qwq] | 32.764B dense; 40,960 configured, 131K with YaRN | Apache-2.0, ungated; official HF/vLLM | 1x96, 1x141, or 2x47; not 1x47 BF16 |

The older five-model MentalMap panel is useful as literature context. The
executable FYP panel nevertheless remains the smaller, current <=9B shortlist
defined above; neither the full historical panel nor larger newer models are
additional experiment candidates.

## Closed-API ceilings excluded from this study

These are reference ceilings, not substitutes for the open-model study.
Providers do not disclose total or active parameters, weights are unavailable,
and neither HF nor vLLM applies.

| Exact API model | Parameters | Context/output | Access and reproducibility |
|---|---:|---:|---|
| [`gpt-5.4-2026-03-05`][gpt54] | Undisclosed | 1,050,000 context; 128,000 output | Proprietary OpenAI API. Prefer the dated snapshot over alias `gpt-5.4`; supports reasoning effort from `none` to `xhigh`. |
| [`claude-opus-5-5`][claude55] | Undisclosed | 1M context; 128K standard max output | Proprietary Anthropic API; official catalog marks it active and released 22 September 2026. No dated snapshot is listed, so record response metadata. |
| [`gemini-4-argon`][gemini4] | Undisclosed | Unified 1,048,576 limit; 65,536 synchronous output, up to 1,048,576 with long decode | Proprietary Gemini API; official page labels the ID stable. Record the returned model version because the stable alias may move. |

These models are recorded only to delimit the landscape. They are not part of
P0, P1, or P2 execution because the FYP uses the <=9B local-model ceiling.

## Recommended FYP priority

The external comparison is reported in three top-level categories:

1. **general instruction / non-explicit reasoning**;
2. **general reasoning**; and
3. **spatial-tuned**, with external specialists separated from the six proposed
   SpatialEntail adapters.

These categories describe the primary comparison role. They are not claims
that an instruction model performs no internal reasoning or that a spatial
specialist cannot also emit a reasoning trace.

### P0 — current controlled study

P0 remains the untuned Qwen3.5-4B checkpoint and the six SpatialEntail
adapters. These seven models answer the thesis-critical supervision and
transfer questions. The existing untuned result is **native-thinking,
two-stage**. No additional base model may delay this evaluation.

### P1 — recommended open-model comparators at or below 9B

| Checkpoint/mode | Why now | Preferred hardware |
|---|---|---|
| `Qwen/Qwen3.5-4B`, thinking off | Close the missing response-regime control against the completed native-thinking untuned run | 1xH100-47 |
| `Qwen/Qwen3.5-9B`, thinking off and on | Clean within-family scale test in both response regimes | 1xH100-47 |
| `google/gemma-4-E4B-it` or `google/gemma-4-E4B-it-qat-w4a16-ct` | Independent current instruction baseline; report the latter explicitly as W4A16 QAT, never as BF16 | 1xH100-47 |
| `google/gemma-3-4b-it` | Historical Gemma instruction baseline only; Gemma 3 has no official Gemma 4-style thinking toggle | 1xH100-47 |
| `mistralai/Ministral-3-8B-Instruct-2512` | Independent instruction-family control near the 9B boundary | 1xH100-47 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Independent reasoning-specialist control | 1xH100-47 |
| `mistralai/Ministral-3-8B-Reasoning-2512` | Paired reasoning comparison with Ministral Instruct | 1xH100-47 |
| `microsoft/Phi-4-mini-instruct` | Compact independent instruction control | 1xH100-47 |

The fixed external spatial-specialist slice is:

| Checkpoint | Specialist role | Preferred hardware |
|---|---|---|
| [`YuhongLiu/SpatialASM-Qwen3.5-9B`](https://huggingface.co/YuhongLiu/SpatialASM-Qwen3.5-9B) | Text-compatible active spatial mentalization; matched against the general Qwen3.5-9B checkpoint | 1xH100-47 |
| [`sensenova/SenseNova-SI-1.2-InternVL3-8B`](https://huggingface.co/sensenova/SenseNova-SI-1.2-InternVL3-8B) | Broad data-scaled spatial intelligence on an independent InternVL backbone | 1xH100-47 |
| [`remyxai/SpaceQwen3-VL-2B-Thinking`](https://huggingface.co/remyxai/SpaceQwen3-VL-2B-Thinking) | Compact explicitly spatial thinking VLM | 1xH100-47 |
| [`BAAI/RoboBrain2.5-8B-NV`](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV) | Embodied and robotic spatial reasoning | 1xH100-47 |
| [`nvidia/Cosmos-Reason2-8B`](https://huggingface.co/nvidia/Cosmos-Reason2-8B) | Physical-world spatial reasoning | 1xH100-47 |

All five receive a text-only preflight before the full frozen benchmark. A
failed no-image preflight is reported as an interface limitation; it must not
be bypassed with dummy images or a changed task contract.

Run them in this order: Qwen3.5-9B thinking off/on; Gemma 4 E4B canonical or
official QAT; Gemma 3 4B IT; Ministral 3 8B Instruct;
DeepSeek-R1-Distill-Qwen-7B; Ministral 3 8B Reasoning; and
Phi-4-mini-instruct. The missing Qwen3.5-4B thinking-off control may run first
because it reuses the P0 weights, but it remains separately labelled from the
completed native-thinking baseline.

### P2 — no larger-model extension

The FYP will not extend the comparator panel above the agreed approximately
9B class. Any additional time goes to analysis, transfer evaluation, or the
already documented benchmark diagnostics rather than a larger-model sweep.

No P0, P1, or P2 model other than `Qwen/Qwen3.5-4B` should receive SpatialEntail
SFT. Otherwise the study changes from “does the proposed supervision improve a
fixed small base?” to a multi-base training sweep and loses its clean causal
comparison.

## Evaluation controls required for a credible comparison

1. **Freeze exact revisions.** Record HF commit SHA, tokenizer revision,
   vLLM version, dtype/quantization, reasoning parser, and chat-template hash.
   Record provider-returned model version for moving API aliases.
2. **Separate direct and reasoning tracks.** Use each shortlisted model's
   documented thinking switch, reasoning parser, and recommended sampling.
   Report token use and latency with accuracy rather than silently treating
   every reasoning checkpoint as an instruction-only model.
3. **Keep the answer contract identical.** Score native stage-1 answer,
   stage-1 format validity, and local stage-2 constrained answer separately.
   For APIs without equivalent constrained decoding, report stage 1 only or a
   clearly labelled extraction/repair result.
4. **Hold context and output budgets fixed.** Start with the frozen 8,192 /
   4,096 total/new-token limits. A model-card maximum context is not an excuse
   to grant one model more problem-solving budget. If a reasoning model reaches
   the cap, report truncation and run a separately labelled budget sensitivity.
5. **Report by SpatialEntail structure.** Preserve CORE/COMP/SHIFT, query
   family, and proof-depth slices. The prior hardened calibration showed that
   Direction D9, not aggregate Count accuracy, carried much of the remaining
   headroom; an overall number alone would hide the mechanism.
6. **Do not collapse precision regimes.** An official FP8 or MXFP4 checkpoint
   is a valid deployed-model result, but label it. Do not compare it as if it
   were the BF16 checkpoint, and never infer fit from active MoE parameters.

## Sources

Model facts above come from the owning organizations' model cards/configs and
official API catalogs, not leaderboard aggregators. Literature claims come
from the original papers or official proceedings pages.

[q35-4]: https://huggingface.co/Qwen/Qwen3.5-4B
[q35-9]: https://huggingface.co/Qwen/Qwen3.5-9B
[q35-27]: https://huggingface.co/Qwen/Qwen3.5-27B
[ministral-3-i]: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
[ministral-8-i]: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
[ministral-14-i]: https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512
[phi-mini]: https://huggingface.co/microsoft/Phi-4-mini-instruct
[mistral-small]: https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
[gemma3-4]: https://huggingface.co/google/gemma-3-4b-it
[gemma3-27]: https://huggingface.co/google/gemma-3-27b-it
[gemma4-e2]: https://huggingface.co/google/gemma-4-E2B-it
[gemma4-e4]: https://huggingface.co/google/gemma-4-E4B-it
[gemma4-e4-qat]: https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct
[gemma4-12]: https://huggingface.co/google/gemma-4-12B-it
[gemma4-26]: https://huggingface.co/google/gemma-4-26B-A4B-it
[gemma4-31]: https://huggingface.co/google/gemma-4-31B-it
[gemma4-report]: https://arxiv.org/abs/2607.02770
[llama31-8]: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
[llama32-3]: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
[llama33-70]: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
[llama4-scout]: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
[llama4-maverick]: https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct
[meta-llama4]: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
[q25-7]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
[q25-32]: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
[mistral7]: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
[gemma2-9]: https://huggingface.co/google/gemma-2-9b-it
[falcon3]: https://huggingface.co/tiiuae/Falcon3-10B-Instruct
[nemotron-4]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
[ministral-8-r]: https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
[ds-r1-7]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
[ds-r1-32]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
[gpt-oss]: https://huggingface.co/openai/gpt-oss-20b
[phi-reasoning]: https://huggingface.co/microsoft/Phi-4-reasoning-plus
[q3-thinking]: https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507
[qwq]: https://huggingface.co/Qwen/QwQ-32B
[mentalmap]: https://arxiv.org/abs/2605.28277
[sitbench]: https://arxiv.org/abs/2601.03590
[interwhen]: https://arxiv.org/abs/2602.11202
[soda]: https://aclanthology.org/2026.acl-long.1382/
[gpt54]: https://developers.openai.com/api/docs/models/gpt-5.4
[claude55]: https://docs.anthropic.com/en/docs/about-claude/models/overview
[gemini4]: https://ai.google.dev/gemini-api/docs/models/gemini-4-argon
