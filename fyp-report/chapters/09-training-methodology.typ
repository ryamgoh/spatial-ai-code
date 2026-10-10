#pagebreak(weak: true)
= Training and Experimental Methodology

== Models and Comparisons

The configured pilot uses Qwen3.5-4B with four SFT conditions: answer-only,
checked Natural, checked Symbolic, and corrupted Symbolic. Each trained model
is compared with an untuned checkpoint using the corresponding inference
instruction. Base problems, splits, answer contracts, and model initialization
are paired. Checkpoint selection uses development results only; the final test
is reserved for the declared comparison.

Answer-only targets omit intermediate evidence and control for task exposure.
Corrupted Symbolic targets retain syntactically valid record ordering and the
gold answer but mutate semantic evidence so replay rejects it. This tests the
value of valid external evidence within the Symbolic package. It does not
isolate every surface property, and no analogous Natural corruption or Natural
process checker is assumed. Natural now narrates checked dependencies, but
corpus-wide rendered-information coverage and token lengths must still be
audited before treating the comparison as notation-only.

== Training and Resource Accounting

The runner materializes arm-specific configurations and data paths from the
validated workload. Context admission uses the actual chat template and
includes evaluation generation reserve. The same tokenizer-rendered evaluation
prompt is sent to inference, without retemplating or added special tokens.
Per-request generation limits are enforced at that boundary. Admission checks
that the training chat starts with the exact evaluation prompt and counts the
remaining completion, including termination tokens, against the generation
limit. Raw assistant-text length alone is insufficient.

Equal examples and equal target tokens answer different questions. A fixed
example comparison measures each supervision package at common task exposure;
matching token volume may require repeating shorter targets and changes the
number of example presentations. Manifest multipliers are planning statistics,
not proof of equal training compute. Report actual optimizer updates, target and
prompt tokens, runtime, peak memory, and repeat factors for every executed arm.
No matched-compute conclusion follows from character counts or configuration
files alone.

== Evaluation and Decision Rules

Primary evaluation uses strict exact-answer-set accuracy, reported per
predeclared cell and macro-averaged across cells. Secondary measures include
ambiguity recognition, invalid output rate, answer-position sensitivity, and
generated tokens. Symbolic process metrics separately report reasoning replay,
menu validity, agreement between reasoning and decision domains, and full trace
validity including the final answer. Their denominator is the applicable
Symbolic outputs; Natural and answer-only outputs have no process score.

Screening may use one seed, but reported training comparisons require repeated
seeds and paired uncertainty estimates. Structural generalisation claims name
the withheld cells and generation provenance explicitly. The corrected
SpatialEval set is external evaluation data and is not a training source.
Retention and broader transfer evaluation are deferred until their prompt and
scoring contracts are integrated. RL is outside this pilot and would require a
separate decision after supervised errors and reward variance are measured.

== Execution Status

This chapter specifies the implemented preparation path and planned study.
No SFT run, checkpoint comparison, or final model test is reported in this
draft. Local tests and tokenizer diagnostics validate software boundaries;
they do not estimate learning effects.
