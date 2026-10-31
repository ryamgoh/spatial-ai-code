# Qwen3 Thinking Training Preparation

Status: locally validated, not launched or sealed.

This package contains 6,000 examples per arm from the unchanged v3 training
candidate. Both arms use Qwen/Qwen3-4B-Thinking-2507 at revision
`768f209d9ea81521153ed38c47d515654e938aea`.

## Format and Checks

The system and user messages are unchanged. Each original explanation is passed
as `reasoning_content`; its final `Answer: X` line is passed as assistant
`content`. The pinned native template produces:

```text
<|im_start|>assistant
<think>
[original explanation]
</think>

Answer: X<|im_end|>
```

The generation prefix, including the opening thinking tag, is masked with
`-100`. Every subsequent explanation, closing tag, answer and turn-end token is
supervised. All 12,000 examples passed native-template equality, token-boundary,
decoded-mask, answer, context and output-budget checks. No truncation, replacement
questions or random validation split was used. Source payloads and the fixed
evaluation selection were hash-checked before and after preparation.

| Measurement | Natural | Symbolic |
|---|---:|---:|
| Examples | 6,000 | 6,000 |
| Maximum full training tokens | 1,766 | 1,930 |
| Maximum supervised tokens | 1,105 | 1,264 |
| Supervised tokens per epoch | 1,930,746 | 2,255,405 |
| Maximum prompt plus 4,096-token Stage-1 reserve | 4,760 | 4,765 |

These measurements use Transformers 5.16.1, tokenizers 0.23.1 and Jinja2 3.1.6.
They certify training-reference fit, not how much reasoning an untuned model
needs before answering.

## Matched Settings

The existing recipe is retained: QLoRA rank 64 / alpha 128, four-bit loading of
the standard BF16 checkpoint, BF16 compute, learning rate 5e-5, paged AdamW
8-bit, cosine schedule, 5% warmup, clipping 0.05, two epochs, seed 42,
microbatch one and accumulation eight on one GPU. Qwen3 uses the seven standard
attention/MLP projections; Qwen3.5 linear-attention targets are not included.

The configs differ only in dataset, cache and model-output paths. Both use the
same seeded question order, no packing and no length grouping. Every epoch is
saved. Do not select by training loss: use the fixed Unique Stage-1 synthetic
dev view accuracy, with the earlier epoch winning ties.

## Files

- `configs/natural.yaml` and `configs/symbolic.yaml`: matched Axolotl configs.
- `data/*-train-tokenized.jsonl`: explicit IDs, attention masks and loss labels.
- `data/*-train-messages.jsonl`: readable native reasoning/content fields.
- `data/order.jsonl`: shared question order.
- `tokenizer/`: pinned tokenizer assets, not model weights.
- `mask-examples.json`: actual decoded masked/supervised examples.
- `token-summary.json` and `manifest.json`: measurements and provenance.

Large data, tokenizer files and runtime caches are ignored by git. Transfer the
whole package to the GPU host, not just tracked YAML files.

## Runtime Handoff

Run from the repository root on the GPU host. First verify the prepared package:

```bash
uv run --frozen python -m spatial.v2.entailment_thinking_training verify
```

The preparation used Axolotl's documented pre-tokenized dataset interface.
The actual Axolotl loader and CUDA runtime have not been executed on this Mac.
At job startup, confirm the loader retains 6,000 rows and the explicit labels;
reject retemplating, truncation or dropped rows.

These are the training commands for a subsequently authorized run:

```bash
uv run --frozen python finetune/finetune.py \
  experiments/spatialentail-composition-v1/prepared-thinking-v1/configs/natural.yaml
uv run --frozen python finetune/finetune.py \
  experiments/spatialentail-composition-v1/prepared-thinking-v1/configs/symbolic.yaml
```

Use one visible GPU per process and run the arms sequentially on a shared GPU.
There is no separate GPU smoke-test job. Inspect model loading, GPU memory,
loss and gradient norm in the opening training steps; stop on nonfinite values,
OOM or loader mismatches. `logging_nan_inf_filter: false` exposes invalid loss
rather than hiding it. This is an operational check, not an implemented automatic
nonfinite-loss abort callback.

The current-model development evaluation package and Stage-2 native-thinking
replay are still pending. The fixed selection is preserved; do not substitute
the old mixed-contract evaluator or silently split this training set.
No model weights were loaded and no training or scheduler submission occurred.
