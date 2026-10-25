# Spatial specialist models for the SpatialEntail panel

*Primary-source review, checked 18 October 2026.*

## Bottom line

There is one strong additional comparator and several defensible diagnostics.
The panel should not grow merely because a checkpoint has “spatial” in its
name.

1. **Add `YuhongLiu/SpatialASM-Qwen3.5-9B` first.** It is the clearest
   text-only transfer test: its spatial post-training teaches view planning and
   imagined-view reasoning, replaces rendered training views with textual
   captions in its SFT trajectories, and inherits Qwen3.5's explicit
   text-only vLLM path. It is also the only candidate here whose specialization
   is materially different from all three existing specialists.
2. **Add `sensenova/SenseNova-SI-1.2-InternVL3-8B` only if a second new
   family is affordable.** It gives a broad, data-scaled spatial-intelligence
   control on an InternVL backbone. The official example runner accepts a
   question without image arguments, but the safe deployment path for this
   exact custom-format checkpoint is Hugging Face Transformers rather than an
   assumed drop-in vLLM command.
3. **Use `etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct` for the closest
   compositional-structure diagnostic.** Its explicit one-to-three-hop spatial
   post-training is closer to SpatialEntail than broad embodied tuning, but it
   remains image-grounded and requires a no-image transfer preflight.
4. **Use `Menlo/AlphaMaze-v0.2-1.5B` only as a text-only boundary probe.** It
   genuinely learns spatial planning from serialized text, but maze movement
   prediction is much narrower than natural-language relational entailment.
5. **Treat `spatialchain/Qwen3-VL-4B-Thinking-SpatialChain` as a
   faithfulness diagnostic, not a headline comparator.** Its scene-graph-
   grounded relation chains are unusually close to SpatialEntail's qualitative
   relation reasoning, but its tuning is visual binary QA and its own card
   reports yes-bias and losses on external spatial benchmarks.
6. **Keep `remyxai/SpaceOm` as a cheap reserve only.** It is public and small,
   but it substantially overlaps the existing
   `remyxai/SpaceQwen3-VL-2B-Thinking` role and uses a non-commercial Qwen
   research license.

No source located a released, first-party, pure-text SODA checkpoint or another
general natural-language spatial specialist. AlphaMaze is pure text but narrow;
the broader additions remain multimodal-native models that can accept text, not
models whose spatial advantage on text-only entailment is already established.
That distinction must be tested rather than assumed.

## Existing panel roles

The marginal-value judgements below use the current specialist set as the
reference:

- [`remyxai/SpaceQwen3-VL-2B-Thinking`](https://huggingface.co/remyxai/SpaceQwen3-VL-2B-Thinking): a compact spatially tuned thinking VLM;
- [`BAAI/RoboBrain2.5-8B-NV`](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV): an embodied/robotics model with novel-view capabilities; and
- [`nvidia/Cosmos-Reason2-8B`](https://huggingface.co/nvidia/Cosmos-Reason2-8B): NVIDIA's physical-AI reasoning model.

The purpose of another model should therefore be to add a new training signal,
backbone, or reasoning mechanism—not simply another 7–8B VLM trained on visual
spatial QA.

## Shortlist

| Priority | Exact checkpoint | Exact size of released tensors | Access and license | Spatial tuning objective | Text-only and local runtime | Value beyond the existing three |
|---|---|---:|---|---|---|---|
| **1 — recommended** | [`YuhongLiu/SpatialASM-Qwen3.5-9B`](https://huggingface.co/YuhongLiu/SpatialASM-Qwen3.5-9B) | **9,409,813,744 BF16 parameters**. This is slightly above nine billion stored parameters but is the official 9B class and fits the study's approximate ceiling. | Public, ungated; HF metadata says Apache-2.0. The card separately says the data and code are for research use, so retain that notice when using project artifacts. | Two-stage SFT then GRPO on SpatialASM-66k. Training synthesizes question-conditioned view plans and imagined views, filters them geometrically, then replaces rendered views with textual captions to create plan–imagine–reason trajectories. The final checkpoint needs no renderer at inference. [Model card](https://huggingface.co/YuhongLiu/SpatialASM-Qwen3.5-9B); [paper](https://arxiv.org/abs/2610.13832). | **Strongest evidence.** Qwen's owning card explicitly documents text-only input and `vllm ... --language-model-only`; the fine-tune preserves the same `Qwen3_5ForConditionalGeneration` architecture. Current main/nightly vLLM or Transformers is required. [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B). | Adds internalized viewpoint planning and mental reconstruction rather than another broad embodied or physical model. It is the best test of whether spatial mentalization learned from visual data transfers to SpatialEntail's text-only partial-information reasoning. |
| **2 — recommended if budget permits** | [`sensenova/SenseNova-SI-1.2-InternVL3-8B`](https://huggingface.co/sensenova/SenseNova-SI-1.2-InternVL3-8B) | **7,944,373,760 BF16 parameters**. | Public, ungated, Apache-2.0. | InternVL3-8B post-trained from the SenseNova-SI-8M collection, described as eight million examples organized by a spatial-capability taxonomy. The release reports evaluation across VSI, MMSI, MindCube, ViewSpatial, SITE, BLINK, 3DSRBench, and EmbSpatial-Bench. [Model card](https://huggingface.co/sensenova/SenseNova-SI-1.2-InternVL3-8B); [paper](https://arxiv.org/abs/2511.13719). | **Hugging Face path verified.** The official card's first example passes `--question "Hello"` with no `--image_paths`; the released `chat` implementation also permits `pixel_values=None`. vLLM lists InternVL 3.0 HF format as accepting text (`T`), images, and video, but this particular repository uses custom InternVL chat code. Prefer its official Transformers runner unless it is converted and smoke-tested. [Released chat code](https://huggingface.co/sensenova/SenseNova-SI-1.2-InternVL3-8B/blob/main/modeling_internvl_chat.py); [vLLM support table](https://docs.vllm.ai/en/stable/models/supported_models/). | Adds an independent InternVL backbone and large-scale spatial data curation. It is a better diversity control than another Qwen-family robotics model, though its tuning is still predominantly visual and benchmark transfer to text is unproved. |
| **3 — compositional diagnostic** | [`etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct`](https://huggingface.co/etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct) | **4,437,815,808 BF16 parameters**. | Public, ungated, Apache-2.0; first-party ETRI VILAB release accepted at ECCV 2026. | GRPO post-training of Qwen3-VL-4B-Instruct on 6,791 MultihopSpatial examples. The benchmark and training cover compositional spatial queries with one to three hops and multiple perspectives. [Model card](https://huggingface.co/etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct); [paper](https://arxiv.org/abs/2603.18892). | Standard `Qwen3VLForConditionalGeneration`; Transformers launch is documented and vLLM lists Qwen3-VL as accepting text. The official example and all demonstrated gains are nevertheless image-plus-grounding conditioned, so text-only transfer is **mechanically possible but not evidenced**. | Adds the closest explicit multi-hop composition objective. Prefer this 4B checkpoint over its 8B sibling for a targeted diagnostic; the 8B adds scale and Qwen-family redundancy without fixing the image/text transfer gap. |
| **4 — text-only boundary probe** | [`Menlo/AlphaMaze-v0.2-1.5B`](https://huggingface.co/Menlo/AlphaMaze-v0.2-1.5B) | **1,777,088,000 BF16 parameters**. | Public, ungated, Apache-2.0. | Qwen2-family model trained by SFT and then GRPO to reconstruct serialized mazes and output movement sequences. The owning card explicitly says the mazes are “presented entirely in text.” [Model card](https://huggingface.co/Menlo/AlphaMaze-v0.2-1.5B); [paper](https://arxiv.org/abs/2502.14669). | **Native text-only.** The official example uses `AutoModelForCausalLM` and a tokenizer with no media processor, so ordinary Transformers/vLLM text generation is straightforward. | The only verified pure-text specialist found, but it learns grid navigation over special maze tokens rather than qualitative natural-language entailment, unknown answers, or candidate sets. A low score would say little; a positive transfer result would be interesting. |
| **5 — faithfulness diagnostic** | [`spatialchain/Qwen3-VL-4B-Thinking-SpatialChain`](https://huggingface.co/spatialchain/Qwen3-VL-4B-Thinking-SpatialChain) | The base [`Qwen/Qwen3-VL-4B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking) has **4,437,815,808 BF16 parameters**. The adapter file contains **858,537,984 tensor entries**, including saved replacement modules as well as LoRA tensors; base plus loaded adapter occupies roughly a 5.30B-parameter runtime envelope, not an 8B-class model. | Public, ungated, Apache-2.0 PEFT adapter. | SFT on 28,350 GQA examples with scene-graph-grounded chains for 11 relation types and binary answers. The associated paper evaluates reasoning faithfulness separately from answer accuracy. [Model card](https://huggingface.co/spatialchain/Qwen3-VL-4B-Thinking-SpatialChain); [paper](https://arxiv.org/abs/2610.06413). | The official card gives a Transformers + PEFT launch path. vLLM lists Qwen3-VL as accepting text, image, and video inputs, so a no-image request is architecturally valid. However, neither the paper nor card demonstrates that the adapter's visual spatial gains survive removal of the image. [vLLM support table](https://docs.vllm.ai/en/stable/models/supported_models/). | Closest available specialist to explicit qualitative relation-chain **faithfulness** and therefore useful as a mechanism probe. Do not interpret it as a clean spatial-language model: training is image-grounded GQA, 62.7% of its data is left/right, and the card reports a 24.8-point yes/no gap plus degradation on ERQA and EmbSpatial-Bench. |
| **6 — reserve** | [`remyxai/SpaceOm`](https://huggingface.co/remyxai/SpaceOm) | **3,754,622,976 FP16 parameters**. | Public, ungated. The repository's actual license file is the **Qwen Research License**, limited to non-commercial research/evaluation; the prose card's “Apache-2.0” line is inconsistent with that file. [License](https://huggingface.co/remyxai/SpaceOm/blob/main/LICENSE). | LoRA-tuned Qwen2.5-VL-3B successor to SpaceThinker. It adds longer synthetic reasoning traces and Robo2VLM-derived robotics/multiple-choice samples to the original quantitative distance and object-relation data. [Model card](https://huggingface.co/remyxai/SpaceOm). | Official Transformers example; vLLM's Qwen2.5-VL entry accepts text as well as image/video. Text-only execution is technically available, but the card only evaluates visually grounded tasks. [vLLM support table](https://docs.vllm.ai/en/stable/models/supported_models/). | A cheap quantitative-spatial/robotics reasoning probe, but it is from the same Remyx ecosystem as SpaceQwen and its card does not show consistent gains over SpaceThinker across all listed benchmarks. It should not displace SpatialASM or SenseNova-SI. |

### Lower-cost SpatialASM sibling

If 9.41B stored parameters is judged outside the operational ceiling, use
[`YuhongLiu/SpatialASM-Qwen3-VL-4B`](https://huggingface.co/YuhongLiu/SpatialASM-Qwen3-VL-4B)
instead. It has **4,826,771,968 BF16 parameters**, is ungated and Apache-2.0,
and uses the same SpatialASM-66k objective. It is less attractive scientifically
because it returns to the Qwen-VL family already represented by SpaceQwen, but
it preserves the active-mentalization intervention at half the size. Do not run
both SpatialASM sizes unless a scale ablation is explicitly part of the study.

## Why the shortlist is not larger

### Public but not panel-ready

| Checkpoint or project | Finding | Decision |
|---|---|---|
| [`RUBBISHLIKE/SpaceR`](https://huggingface.co/RUBBISHLIKE/SpaceR) | The namespace and bare HF repository give weak provenance **on HF alone**, but this is not an unrelated community upload: the authors' [official GitHub repository](https://github.com/OuyangKun10/SpaceR) explicitly links the exact ID and says the checkpoint was released there. It also states that weights follow CC BY-NC 4.0. The tensors contain **8,292,166,656 BF16 parameters**; the paper describes Qwen2.5-VL-7B trained with spatially guided RLVR and map imagination on video spatial reasoning. The HF repository itself still has no model card, declared library, or license metadata. [Paper](https://arxiv.org/abs/2504.01805). | **Hold from the main panel.** Identity and non-commercial terms can be recovered from the first-party code repository, but the checkpoint lacks a self-contained launch contract. Its video/map-imagination role also overlaps Cosmos-Reason2 more than SpatialASM does. Document the provenance chain explicitly if it is ever run. |
| [`BAAI/RoboBrain2.0-7B`](https://huggingface.co/BAAI/RoboBrain2.0-7B) | First-party BAAI checkpoint and Apache-2.0, but access is gated. The [technical report](https://arxiv.org/abs/2507.02029) positions it as the prior embodied perception/reasoning/planning generation. | **Do not add beside RoboBrain2.5-8B-NV.** It is an older same-family control, not a new capability axis. |
| [`remyxai/SpaceThinker-Qwen2.5VL-3B`](https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B) | Public 3,754,622,976-parameter Qwen2.5-VL model tuned on about 12K synthetic spatial reasoning traces. Its own card marks `remyxai/SpaceOm` as the new version. | **Use SpaceOm if this line is needed.** Running both is redundant with each other and with SpaceQwen. |
| [`a8cheng/SpatialRGPT-VILA1.5-8B`](https://huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B) | The official-author checkpoint is public and the [paper](https://arxiv.org/abs/2406.01584) is strong primary evidence for 2D/3D grounded region reasoning. The HF repository has no model card, license, pipeline tag, or standard Transformers entry point; it contains custom VILA, region-extractor, and vision-tower components. | **Exclude from the text-only panel.** It is a useful image/region model but no first-party source establishes ordinary no-image prompting or vLLM compatibility. |
| [`manycore-research/SpatialLM1.1-Qwen-0.5B`](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B) and [`manycore-research/SpatialLM1.1-Llama-1B`](https://huggingface.co/manycore-research/SpatialLM1.1-Llama-1B) | First-party NeurIPS 2025 releases, 603,511,168 and 1,345,883,776 parameters respectively. They generate structured room layouts and boxes from 3D point clouds; the official inference command requires `--point_cloud`. [Project/paper card](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B). | **Exclude.** The language decoder does not make the released system a normal text-only QA checkpoint. |
| [`mindlab-research/MJev-Spatial-4B`](https://huggingface.co/mindlab-research/MJev-Spatial-4B) | Public Apache-2.0 research release built on Qwen3.5-4B. It reformulates spatial questions as candidate scoring through native attention and emits no answer tokens. Every documented request includes image media. [Model card](https://huggingface.co/mindlab-research/MJev-Spatial-4B); [paper](https://arxiv.org/abs/2610.22276). | **Exclude from the current harness.** It requires an image plus typed candidate contract and is not an autoregressive text-answer model. It could be a future interface experiment, not a drop-in panel member. |
| [`OX-PIXL/SpatialThinker-3B`](https://huggingface.co/OX-PIXL/SpatialThinker-3B) | Public, ungated, **4,065,787,904 BF16 parameters**. GRPO on STVQA-7K teaches a task-focused scene graph with dense format, count, accuracy, and grounding rewards. The official prompt explicitly begins by observing an image and includes image dimensions. The card declares Apache-2.0, while the Qwen2.5-VL-3B upstream uses the restrictive Qwen Research License. [Paper](https://arxiv.org/abs/2511.07403). | **Hold.** Scientifically relevant scene-graph reasoning, but text-only transfer is unsupported and derivative licensing needs clarification. MultiHopSpatial is the cleaner Qwen-VL compositional probe. |
| [`Hyggge/EgoMind-7B`](https://huggingface.co/Hyggge/EgoMind-7B) | Public, ungated, Apache-2.0, **8,292,166,656 BF16 parameters**. It uses 5K CoT SFT and 20K RL examples to build linguistic scene graphs across egocentric frames. [Model card](https://huggingface.co/Hyggge/EgoMind-7B); [paper](https://arxiv.org/abs/2604.03318). | **Hold.** Linguistic spatial cognition is interesting, but the learned task is multi-frame visual reasoning and overlaps the existing embodied/physical models. |
| [`allenai/Molmo2-ER`](https://huggingface.co/allenai/Molmo2-ER) | First-party Ai2 release, public, Apache-2.0, **4,850,869,200 parameters**. It uses specialize-then-rehearse training on a 3.3M-example embodied corpus and provides an independent Molmo/Qwen3-4B architecture. [Paper](https://arxiv.org/abs/2605.02881). | **Hold.** It is the best independent-family embodied ceiling found, but uses custom Molmo2 multimodal code and its skills are visual pointing, correspondence, scene understanding, and temporal reasoning rather than text entailment. |
| [`gradient-spaces/respace-sg-llm-1.5b`](https://huggingface.co/gradient-spaces/respace-sg-llm-1.5b) | Public, ungated, CC BY-NC-SA 4.0, **1,543,714,304 BF16 parameters**. It is a genuine text-generation checkpoint for autoregressive 3D indoor-scene synthesis and editing. Its own card says not to use it without ReSpace's add/remove scaffolding. [Paper](https://arxiv.org/abs/2506.02459). | **Exclude.** It generates structured scenes rather than answers about entailed relations. AlphaMaze is the cleaner text-only spatial boundary probe. |

### Strong research idea, but no exact released checkpoint

The ACL 2026 paper
[SODA](https://aclanthology.org/2026.acl-long.1382/) is the closest conceptual
match found: it explicitly trains pure-text spatial reasoning with SFT and
GRPO over SPOD-143k. As of the cutoff, neither the official ACL page nor the
paper identifies a first-party, downloadable HF checkpoint with a stable model
ID and license. A paper result is not an executable model. Revisit SODA only
after its authors release exact weights and an inference contract.

## Required preflight before adding any multimodal specialist

1. **Pin the artifact.** Record exact HF revision, model and tokenizer hashes,
   Transformers/vLLM commit, dtype, and chat-template hash. SpatialASM and
   SpatialChain were released only days or months before this review and have
   little deployment history.
2. **Prove no-image inference.** Send a text-only smoke prompt with no dummy
   image and no fabricated visual tokens. A model architecture accepting text
   is not evidence that spatial post-training transfers to text.
3. **Run the matching base checkpoint.** For SpatialASM-Qwen3.5-9B compare
   against `Qwen/Qwen3.5-9B`; for SenseNova-SI compare against
   `OpenGVLab/InternVL3-8B`; for SpatialChain compare against
   `Qwen/Qwen3-VL-4B-Thinking`. The tuned-minus-base delta is more informative
   than the tuned score alone.
4. **Keep the SpatialEntail answer contract fixed.** Do not add multiple-choice
   options, images, or a specialist-specific repair prompt. Score native output
   validity separately from any constrained second pass.
5. **Audit response regime.** SpatialASM-Qwen3.5 thinks by default but supports
   non-thinking mode; SpatialChain is a thinking adapter. Label direct and
   thinking runs rather than merging them.
6. **Check training overlap by task, not only by examples.** SenseNova-SI and
   SpatialChain train on common spatial benchmark families and relation
   vocabularies. SpatialEntail instances may be novel while the task schema is
   familiar. Report that as transfer, not zero-shot proof of a new capability.

## Recommended final specialist slice

If only one new run is possible, choose
`YuhongLiu/SpatialASM-Qwen3.5-9B`. If two are possible, add
`sensenova/SenseNova-SI-1.2-InternVL3-8B` for backbone and data diversity. Use
`etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct` as the first
structure-focused appendix run and AlphaMaze as a deliberately narrow
text-only boundary probe. The 4B SpatialChain adapter is a second appendix
choice when trace faithfulness is itself under study. Keep SpaceOm and all
excluded models out of the main panel unless a preflight uncovers a specific
question the first two cannot answer.
