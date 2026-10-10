#pagebreak(weak: true)
#heading(level: 1, numbering: none)[Appendix C — Experimental Configuration]

This appendix records the configuration status for the eventual learning study.
Values marked fixed are predeclared methods, not results; the benchmark-cell
registry and allocation remain pending.

#heading(level: 2, numbering: none)[Model and Training Matrix]

#figure(
  table(
    columns: (1fr, 1fr),
    table.header([*Field*], [*Status and value*]),
    [Models], [`Qwen/Qwen3.5-2B`; `Qwen/Qwen3.5-4B`],
    [Training seeds], [`42`, `43`],
    [Epochs], [Two fixed epochs],
    [Core arms], [Answer-only; checked Natural; checked Symbolic],
    [4B mechanism arms], [Corrupted Symbolic; Symbolic local proof],
    [Size calibration], [Candidate maximum budgets: $4"K" subset 8"K" subset 17"K"$; cell quotas pending],
    [Training depth], [1–4],
    [Depth test], [1–8; 5–8 marked extrapolative],
  ),
  caption: [Training configuration. Exact model/tokenizer revisions, cell allocation, and LoRA fields must be copied from the frozen or executed manifests.],
) <appendix-training-matrix>

The executed artifact must record base-model and tokenizer revisions, adapter
rank and targets, quantization, sequence length, batch and accumulation sizes,
learning rate and schedule, checkpoint-selection metric, exact data paths,
split fingerprints, prompt template, and generation settings. Configuration
files alone are not proof that a run completed.

#heading(level: 2, numbering: none)[Task and Split Record]

The release manifest must contain the requested and accepted counts for every
frozen benchmark cell, query/capability marginals, continuous difficulty distributions, generation provenance,
structural signatures, rejection reasons, answer-position distributions,
context-token distributions, and train/development/test fingerprints. All arms
for one base problem remain in one split.

#heading(level: 2, numbering: none)[Quality-Review Rubric]

The source-blinded screen runs `DeepSeek-V4-Pro` and `GPT-5.5` sequentially in
isolated terminal-agent contexts over the same anonymized prompt and checked
Natural trace. A human sees only items rejected by both judges and decides an
Approve/Reject appeal without seeing either model's reasons.

#figure(
  table(
    columns: (1fr, 1.5fr),
    table.header([*Criterion*], [*Question*]),
    [Prompt clarity], [Can the task be understood without unstated information?],
    [Answer contract], [Is the requested answer cardinality and menu policy clear?],
    [Readability], [Can a reader follow the explanation without decoding implementation internals?],
    [Self-containment], [Does the trace state the evidence needed for its conclusion?],
    [Redundancy], [Is repetition avoidable rather than semantically necessary?],
    [Answer leakage], [Does the answer appear before the evidence intended to justify it?],
    [Acceptability], [Is the item suitable for the declared training/research use?],
  ),
  caption: [Human-facing review rubric. Formal correctness remains checker-authoritative.],
) <appendix-quality-rubric>

The executed review record must pin both requested model identifiers, provider,
system-prompt version, output schema, sample IDs, randomized order, routing
decisions, and human instructions. Any generation control unavailable through
the terminal-agent interface is recorded as unavailable rather than invented.
It reports double-approval, split, double-rejection,
human-override, and final disposition rates, plus agreement and Cohen's kappa
between the two model judges. Selective human appeal cannot estimate overall
human acceptability or LLM–human agreement.

Let $C$ denote the number of frozen reporting cells. Review covers at least the
larger of 2% of the formally admitted maximum candidate pool or $5 C$ examples,
with at least five per cell. A fixed numeric floor is not declared before the
registry is frozen.

#heading(level: 2, numbering: none)[External Adapter Record]

Every external evaluation adapter must record the source revision, retained and
excluded subsets, relation mapping, native answer contract, and any independent
entailment audit. External test rows are never used for SFT, prompt selection,
checkpoint selection, or judge calibration.
