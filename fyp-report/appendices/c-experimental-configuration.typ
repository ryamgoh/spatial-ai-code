#pagebreak(weak: true)
#heading(level: 1, numbering: none)[Appendix C: Configuration Freeze]

The settled study design uses two matched mixed-contract adapters, Natural
and Symbolic, with one shared initial training seed. Stage 1 allows 4,096
generated tokens; Stage 2 runs on every example using the same checkpoint.
Checkpoint selection uses only Stage-1 synthetic development answer accuracy.

The following implementation settings still require a recorded freeze after
feasibility checks and before training:

- supported family/contract/semantic-case matrix and evidence schema;
- dataset size, sampling proportions, candidate domains, menu coverage, and
  chain-length ranges;
- matched explanation templates and evidence selection rules;
- base-model and tokenizer revisions, adapter configuration, training seed,
  schedule, example order, and context envelope;
- exact style and contract prompts, letter-set serialization, final-answer
  parser, stop rules, decoding settings, and Stage-2 answer allowance;
- checkpoint evaluation schedule, development aggregation, and tie-break rule;
- accuracy-loss tolerance and paired analysis procedure;
- release receipt, manual audit records, and external contract-view versions.

Historical model choices, quotas, training seeds, and context lengths are not
automatically inherited. Configuration files and a successful compilation of
this report are not evidence that training or evaluation has completed.
