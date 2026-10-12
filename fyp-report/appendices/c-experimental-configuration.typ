#pagebreak(weak: true)
#heading(level: 1, numbering: none)[Appendix C — Experimental Configuration]

This appendix records the frozen learning-study configuration. Configuration
values are methods rather than model results.

#heading(level: 2, numbering: none)[Model and Training Matrix]

#figure(
  table(
    columns: (1fr, 1fr),
    table.header([*Field*], [*Status and value*]),
    [Model], [`Qwen/Qwen3.5-4B`],
    [Training seeds], [`42`, `43`],
    [Epochs], [Two fixed epochs],
    [Core arms], [Answer-only; verified Natural],
    [Optional extension], [Compact Chain-of-Symbol-like target, only after corpus-wide token reduction is demonstrated],
    [Training allocation], [8,000 bases; nested 4,000-base subset retained as a data artifact],
    [Training depth], [Controlled 3/4 anchors plus CORE/compositions],
    [Sequence envelope], [12,288 training; 16,384 evaluation],
    [Benchmark groups], [CORE: 1,300; COMP: 1,200; SHIFT: 1,800],
    [Depth test], [3–8; 3/4 matched anchors, 5–8 answer-only extrapolation],
  ),
  caption: [Frozen training configuration. Exact model/tokenizer revisions, cell allocation, and LoRA fields are recorded in the prepared manifests.],
) <appendix-training-matrix>

The executed artifact must record base-model and tokenizer revisions, adapter
rank and targets, quantization, sequence length, batch and accumulation sizes,
learning rate and schedule, checkpoint-selection metric, exact data paths,
split fingerprints, prompt template, and generation settings. Configuration
files alone are not proof that a run completed.

#heading(level: 2, numbering: none)[Task and Split Record]

The release manifest contains the requested and accepted counts for every
frozen benchmark cell, query/capability marginals, continuous difficulty distributions, generation provenance,
structural signatures, rejection reasons, answer-position distributions,
context-token distributions, and train/development/test fingerprints. All arms
for one base problem remain in one split.

#heading(level: 2, numbering: none)[Quality-Review Rubric]

The source-blinded screen runs `DeepSeek-V4-Pro` and `GPT-5.5` sequentially in
isolated terminal-agent contexts over the same anonymized prompt and verified
Natural target. A human sees only items rejected by both judges and decides an
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
