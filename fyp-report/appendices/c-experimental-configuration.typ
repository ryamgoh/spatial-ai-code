#pagebreak(weak: true)
#heading(level: 1, numbering: none)[Appendix C — Experimental Configuration]

This appendix is the frozen configuration record for the eventual learning
study. Until the experiment executes, every value below is a predeclared method,
not a result.

#heading(level: 2, numbering: none)[Model and Training Matrix]

#figure(
  table(
    columns: (1fr, 1fr),
    table.header([*Field*], [*Predeclared value*]),
    [Models], [`Qwen/Qwen3.5-2B`; `Qwen/Qwen3.5-4B`],
    [Training seeds], [`42`, `43`],
    [Epochs], [Two fixed epochs],
    [Core arms], [Answer-only; checked Natural; checked Symbolic],
    [4B mechanism arms], [Corrupted Symbolic; Symbolic local proof],
    [Size calibration], [$4"K" subset 8"K" subset 17"K"$],
    [Training depth], [1–4],
    [Depth test], [1–8; 5–8 marked extrapolative],
  ),
  caption: [Frozen high-level training matrix. Exact model/tokenizer revisions and LoRA fields must be copied from the executed run manifests.],
) <appendix-training-matrix>

The executed artifact must record base-model and tokenizer revisions, adapter
rank and targets, quantization, sequence length, batch and accumulation sizes,
learning rate and schedule, checkpoint-selection metric, exact data paths,
split fingerprints, prompt template, and generation settings. Configuration
files alone are not proof that a run completed.

#heading(level: 2, numbering: none)[Task and Split Record]

The release manifest must contain the requested and accepted counts for every
task bucket, continuous difficulty distributions, generation provenance,
structural signatures, rejection reasons, answer-position distributions,
context-token distributions, and train/development/test fingerprints. All arms
for one base problem remain in one split.

#heading(level: 2, numbering: none)[Quality-Review Rubric]

The source-blinded review asks two independent external LLM families and one
human reviewer to score the same anonymized prompt and checked Natural trace.

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

The executed review record must pin both judge model identifiers, temperature,
system prompt, output schema, sample IDs, randomized order, and human
instructions. It reports raw agreement, Fleiss' kappa, pairwise Spearman
correlations, and adjudicated disagreement categories.

#heading(level: 2, numbering: none)[External Adapter Record]

Every external evaluation adapter must record the source revision, retained and
excluded subsets, relation mapping, native answer contract, and any independent
entailment audit. External test rows are never used for SFT, prompt selection,
checkpoint selection, or judge calibration.
