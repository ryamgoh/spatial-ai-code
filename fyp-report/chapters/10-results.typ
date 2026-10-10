#pagebreak(weak: true)
= Results and Planned Analysis

== Available Evidence

The current quantitative result is the conditional SpatialMap-TQA semantic
audit reported in Part I. Software tests exercise certificate replay, parsing,
menu resolution, provenance, splits, and context admission. Tokenizer probes
measure serialized lengths for particular generated examples. None of these
measure a trained model's spatial capability.

The bounded pilot preparation admitted 80 base problems (320 rows). Maximum
full training lengths were 4,032 tokens for checked/corrupted Symbolic, 1,863
for Natural, and 129 for answer-only; the maximum reserved evaluation length
was 4,218 tokens and the maximum rendered generation target was 3,910. The
separate 16-base premise-first holdout preparation also passed admission, with
maximum training length 965 and reserved evaluation length 4,265. These are
Qwen3.5-4B tokenizer diagnostics for the executed configuration. They do not
estimate average cost across a broader corpus or any model's accuracy.

== Main Model Results: Not Run

The four-arm SFT pilot and its matched untuned evaluations have not been run.
The final result table will report example count, training seed, strict accuracy,
per-cell accuracy, token use, and development-selected checkpoint for each arm.
No ranking, improvement percentage, confidence interval, or winning format can
be supplied before those artifacts exist.

== Generalisation and Transfer: Not Run

Predeclared structural holdouts will be reported separately from interpolation.
Signature-disjoint splits alone do not establish a held-out reasoning skill.
The analysis must state which rule compositions, depths, or generation modes
were excluded from training and how their premise and entity budgets compare.
Corrected-SpatialEval model evaluation and retention checks remain pending;
the repository's earlier V1 experiments are not measurements of this V2 pilot.

== Planned Error Analysis

Classify answer errors by semantic cell, ambiguity, option position, and query
structure. For Symbolic output, distinguish malformed records, invalid local
steps, unsupported conclusions, incorrect qualitative models, domain disagreement,
and a wrong final answer after valid evidence. Report missing Natural process
validation as a measurement limit. Examine paired fixes and regressions instead
of inferring a mechanism solely from aggregate accuracy.

Count scaling requires a separate study over contingent-candidate count and
certificate length. Admission rejection rates are necessary to show how context
limits change the sampled distribution. Human readability and causal faithfulness
of explanations likewise remain unmeasured research questions.
