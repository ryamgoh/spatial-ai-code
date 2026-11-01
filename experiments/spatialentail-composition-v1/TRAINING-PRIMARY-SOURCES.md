# Training Primary Sources v1

Research date: 2026-11-01. Scope: requirements for clean matched Natural/Symbolic supervised fine-tuning (SFT), not a local pipeline audit. No training code was changed; no training, server, GPU, installed-package, dataset, checkpoint, or Slurm-job verification was performed.

## Readiness Conclusion

**Research supports a conditional readiness contract, not a verified ready-to-train verdict.** Both arms need the same model/tokenizer revisions, paired semantic examples and splits, supervision policy, declared training-budget rule, evaluation protocol, numerical checks, and artifact provenance. The sources below establish framework behavior; they do not establish that the current pipeline satisfies it. Local inspection and runtime evidence remain separate work.

## Source Versions

- Qwen card and tokenizer were read from the official model repository's mutable `main`; the model name's `2507` suffix does not pin downloaded files. Record the resolved Hub revision and template hash for the actual run. [Q1, Q2]
- Axolotl loader source was accessible at the requested immutable git commit `7d775808d0ff71f21f553bd7d761631d7fb089e6`. These findings describe that source, not proof that a server has it installed. Axolotl documentation is current, unversioned documentation. [A1-A5]
- Transformers `main` documentation/source is mutable development material, not the pinned runtime. Historical `v4.57.1` official docs/source were also consulted to cross-check the logging/checkpoint contracts; **v4.57.1 is not asserted to be installed**. PyTorch `stable` and Slurm's online manual likewise are not installed-version evidence. [T1-T6, P1, S1]

## Findings and Implications

### 1. Qwen Thinking and Chat Boundaries

**Source facts.** The model card says this release supports only thinking mode; `enable_thinking=True` is unnecessary. Its generation template appends `<|im_start|>assistant\n<think>\n`, so generated text can contain `</think>` without a generated opening `<think>`. The tokenizer declares `bos_token: null`, `add_bos_token: false`, and EOS `<|im_end|>`. For a final assistant message after the last user query, the template renders a thinking block even when reasoning content is empty. The card warns that Transformers below 4.51.0 cannot recognize `qwen3`. [Q1, Q2]

**Matched-SFT requirement (derived).** Use one pinned template and explicit reasoning policy for both arms. Check that the inference prefix is consistent with the training serialization and label boundary: specify whether opening/closing thinking delimiters, reasoning, final answer, and EOS are supervised. An answer-only target with an empty thinking block is not evidence of disabling thinking mode. Do not blindly invent a BOS token from generic dataset instructions. Verify token boundaries rather than assuming text concatenation preserves tokenization. [Q1, Q2, A1]

**Evaluation implication.** Apply the same final-answer extraction and malformed/missing-`</think>` policy to both arms. The card recommends sampling `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0` and generous output lengths; these are inference recommendations, not mandated SFT hyperparameters. A shorter output budget or greedy decoding is a protocol choice that must be declared and matched; report budget exhaustion separately. Native context capacity does not certify GPU feasibility or adequate task-specific output length. [Q1]

### 2. Pretokenized Data: Responsibility Moves Upstream

**Documentation contract.** Axolotl's pretokenized guide specifies an empty dataset `type:`, exactly `input_ids`, `attention_mask`, and `labels`, `-100` for ignored labels, manually supplied special tokens, and an unmasked EOS target. Instruction-training truncation/padding is the producer's responsibility. Its generic BOS advice must be interpreted against Qwen's actual tokenizer/template. [A1, Q2]

**Pinned loader behavior.** `get_dataset_wrapper` returns the dataset unchanged with `UnsupportedPrompter` when `_is_dataset_already_tokenized` sees a Hugging Face **`Dataset`** with all three named features. Detection checks column presence, not exact column equality, per-row lengths, token validity, semantic correctness, or mask correctness; it does not cover `IterableDataset`. This shortcut precedes prompt-strategy dispatch. Thus the pinned wrapper does not retokenize or reconstruct assistant-only masks for recognized pretokenized data, regardless of a prompt-masking configuration elsewhere. [A2]

**Not an end-to-end pass-through guarantee.** The SFT loader can load prepared data before loading the raw source. On the raw-preparation path, unless preparation is skipped, it can apply length handling, packing, and exact deduplication before saving prepared data. The pinned length handler defaults to dropping overlong sequences and also supports truncation or raising. Prepared-data cache identity is generated from configuration/tokenizer identity, not a demonstrated checksum validation of current input-file bytes. The validation splitter uses `shuffle=False`; a seed alone does not establish random or semantically matched held-out membership. [A3-A5]

**Matched-SFT requirement (derived).** Validate integer IDs, equal aligned array lengths, attention/padding masks, labels equal to the corresponding IDs or `-100`, at least one effective supervised target after causal shifting, intended prompt masking, and supervised EOS. Preserve semantic pair IDs in a sidecar manifest while respecting the documented three-column training schema. Track paired IDs before and after filtering, truncation, deduplication, and packing; enforce common retained membership and explicit split manifests. Prove which raw/prepared artifacts were actually loaded using content hashes and logs, not merely config equality or successful loading. [A1-A5]

**Does not certify:** correct supervision, absence of leakage, intact answers/EOS after truncation, equal arm membership, fresh caches, or equal optimization exposure.

### 3. Evaluation Loss Is Not Generated Accuracy

**Source facts.** Standard Trainer evaluation produces loss and optionally logits/labels; metrics require an explicit `compute_metrics` function. The standard `prediction_step` performs a forward/loss computation, not autoregressive `generate`. The separate Seq2Seq interface documents `predict_with_generate` for generative metrics. In ordinary causal-LM SFT with supplied target tokens, loss is teacher-forced; tokenwise argmax metrics under those inputs are not free-running task accuracy. Custom Axolotl trainers or overrides require separate inspection. [T1, T2, T5]

**Matched-SFT requirement (derived).** Report held-out SFT loss as an optimization diagnostic, and independently generate from prompt-only inputs to score final task answers. Freeze paired held-out IDs, generation settings and seed policy, answer parser, invalid-output handling, and checkpoint-selection rule. Never leak reference completion tokens into generation prompts. Natural/Symbolic losses can reflect different token lengths and target distributions, so neither lower cross-arm loss nor falling loss establishes superior entailment accuracy. [T1, T2, Q1]

**Does not certify:** task accuracy, compositional generalization, reasoning correctness, fair decoding, or a statistically reliable Natural/Symbolic difference.

### 4. Finite Logged Loss Is Not Numerical Safety

**Source facts.** `logging_nan_inf_filter=True` is the documented default in both the consulted current source and historical v4.57.1. It substitutes a logging-window/recent-valid-loss average for NaN/Inf step losses; it affects logging only, not how gradients are computed or applied. [T3, T4]

**Matched-SFT requirement (derived).** Require explicit nonfinite-loss/gradient detection and an agreed failure policy, with the same precision, optimizer and clipping policy for both arms. Disabling the logging filter exposes nonfinite logged losses but is not itself a fail-fast safeguard. Retain numerical diagnostics and skipped-update evidence where applicable.

**Does not certify:** finite gradients or weights, successful optimizer updates, or numerical stability from a smooth loss plot alone.

### 5. Checkpoints Need an Explicit Artifact Contract

**Source facts.** Trainer distinguishes model export (`save_model`, reloadable with `from_pretrained`) from resuming training (`train(resume_from_checkpoint=...)`). The documented `save_only_model=True` omits optimizer/scheduler/RNG state and prevents full training resumption. `save_total_limit` rotates older checkpoints; with best-model loading, the best is retained alongside recent checkpoints, potentially keeping both best and last even at limit 1. Best-model selection defaults to loss in the documented cases; a generated accuracy metric must actually be computed to select on it. Save/eval scheduling constraints are version-sensitive and must be checked against the actual runtime. [T1-T4]

**Documentation caveat.** The historical v4.57.1 Trainer guide says optimizer state is not saved by default, but its `TrainingArguments` docs declare `save_only_model=False`, and its checkpoint implementation saves optimizer/scheduler/scaler/RNG when that flag is false. Use the exact-version implementation and resulting artifacts rather than that guide sentence as the default-state authority. [T4, T6]

**Matched-SFT requirement (derived).** Declare final-vs-best-vs-fixed-step selection before comparing arms; use separate output directories and record selected checkpoint step, hashes, base revision, tokenizer/template, and adapter configuration if relevant. If resumability is required, retain complete runtime-appropriate training state and test resume. If only inference is required, test reloading the complete intended base/adapter/tokenizer combination. Ensure rotation does not remove required comparison artifacts.

**Does not certify:** checkpoint completeness, reloadability, correct adapter/base pairing, successful resume, or best generated accuracy merely because a checkpoint directory exists.

### 6. Reproducibility Is Environment-Bounded

**Source facts.** PyTorch explicitly does not guarantee reproducibility across releases, commits, platforms, or CPU/GPU even with identical seeds. It recommends controlling PyTorch, Python and NumPy RNGs, separately seeded RNG objects, and DataLoader worker/generator seeding. Deterministic algorithms can raise for unsupported operations and can be slower. Disabling cuDNN benchmarking is distinct from making chosen algorithms deterministic; relevant CUDA workloads may also need `CUBLAS_WORKSPACE_CONFIG`. [P1]

**Matched-SFT requirement (derived).** Freeze package revisions, model/data hashes, hardware/world size, precision, kernels and determinism policy; control initialization, data ordering and generation RNGs. Match optimizer settings, effective batch size and update schedule. Predeclare whether the estimand matches examples/epochs or tokens/compute: Natural/Symbolic token lengths can differ, so equal steps alone cannot guarantee equal token exposure. Log actual retained examples, input/supervised tokens and updates; use paired repeated seeds where feasible. These experimental controls are recommendations derived from the reproducibility limits, not framework guarantees.

**Does not certify:** bitwise reproducibility, equal trajectories, representation-only causality, or uncertainty estimates from a single common seed.

### 7. Slurm `afterok` Is an Exit-Status Gate

**Source facts.** `afterok:job_id[:job_id...]` releases a dependent job only after the specified jobs complete with exit code zero. Comma-separated dependencies require all conditions; question-mark separation allows alternatives. Submission returns a job ID before execution. Dependencies that fail are not repaired simply by a later successful requeue; invalid-dependency handling is configurable. The manual also limits when an `afterok` dependency may be submitted relative to the predecessor's completion (`MinJobAge`). [S1]

**Matched-SFT requirement (derived).** Make the predecessor propagate training/validation failures as nonzero exits. Gate evaluation on the intended training job IDs, and independently validate the expected arm/checkpoint manifest, hashes and reloadability before scoring. Use an all-success dependency when an aggregation job requires both arms.

**Does not certify:** that training actually ran, reached intended updates, produced fresh artifacts, or was numerically valid; a wrapper can exit zero despite a failed internal command. A successful `sbatch` submission is not job completion.

## Evidence Needed for a Readiness Verdict

These are acceptance-evidence requests, not assertions about the existing pipeline:

1. Resolved model/tokenizer/package revisions and effective configuration for both arms.
2. Paired split and retained-ID manifests, content hashes, decoded supervision-boundary checks, and actual loaded-cache provenance.
3. Declared matching budget with measured examples/tokens/updates and explicit numerical failure evidence.
4. Reload-tested selected artifacts; complete-state resume evidence only if resumability is claimed.
5. Prompt-only generated evaluation with frozen parsing/budget/selection rules and separately reported loss.
6. Slurm completion/exit evidence plus artifact validation, not submission IDs or `afterok` alone.

## Primary Sources

- **Q1:** [Official Qwen3-4B-Thinking-2507 model card, mutable main](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/blob/main/README.md).
- **Q2:** [Official tokenizer configuration and Jinja template, mutable main](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/blob/main/tokenizer_config.json).
- **A1:** [Axolotl current pretokenized dataset documentation](https://docs.axolotl.ai/docs/dataset-formats/tokenized.html).
- **A2:** [Pinned Axolotl wrappers.py: get_dataset_wrapper and _is_dataset_already_tokenized](https://github.com/axolotl-ai-cloud/axolotl/blob/7d775808d0ff71f21f553bd7d761631d7fb089e6/src/axolotl/utils/data/wrappers.py).
- **A3:** [Pinned Axolotl sft.py: prepared/raw loading and subsequent preparation](https://github.com/axolotl-ai-cloud/axolotl/blob/7d775808d0ff71f21f553bd7d761631d7fb089e6/src/axolotl/utils/data/sft.py).
- **A4:** [Pinned Axolotl shared.py: cache identity/loading and validation split](https://github.com/axolotl-ai-cloud/axolotl/blob/7d775808d0ff71f21f553bd7d761631d7fb089e6/src/axolotl/utils/data/shared.py).
- **A5:** [Pinned Axolotl utils.py: handle_long_seq_in_dataset](https://github.com/axolotl-ai-cloud/axolotl/blob/7d775808d0ff71f21f553bd7d761631d7fb089e6/src/axolotl/utils/data/utils.py).
- **T1:** [Transformers development Trainer API documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer).
- **T2:** [Transformers current main Trainer implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py).
- **T3:** [Transformers current main TrainingArguments documentation in source](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py).
- **T4:** [Historical v4.57.1 Trainer/TrainingArguments API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer) and [versioned TrainingArguments source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/training_args.py).
- **T5:** [Historical v4.57.1 Seq2Seq generation-evaluation arguments](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/training_args_seq2seq.py).
- **T6:** [Historical v4.57.1 Trainer guide, checkpoint section](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/trainer.md#checkpoints) and [versioned checkpoint implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/trainer.py).
- **P1:** [PyTorch stable reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- **S1:** [Official Slurm sbatch manual, dependency and exit-status semantics](https://slurm.schedmd.com/sbatch.html).
