#pagebreak(weak: true)
= SpatialEval Audit and Correction

== Audit Protocol

The audit command reads the untouched SpatialMap-TQA JSONL file, parses every
question through the SpatialEval adapter, solves it under `SINGLE`, and records
the published oracle, possible answers, entailed answers, classification, and a
validated coordinate witness for each possibility. It then reparses every
corrected prompt before writing the output. Existing artifacts are preserved
unless replacement is requested explicitly.

The original dataset is not modified. The audit and corrected dataset are
stored separately, and coordinate witnesses appear only in the audit artifact.
Two independent Z3 runs produced byte-identical audit and correction files.

== Current Audit Result

#figure(
  table(
    columns: (1fr, auto, auto),
    table.header(
      [*Query*],
      [*Exact oracle*],
      [*Possible but non-unique*],
    ),
    [Direction], [332], [168],
    [Which], [140], [360],
    [Count], [195], [305],
    [*Total*], [*667*], [*833*],
  ),
  caption: [SpatialMap-TQA audit under the declared `SINGLE` semantics.],
) <spatialeval-audit-summary>

All 1,500 questions were parsed and solved. The published answer is uniquely
entailed in 667 questions and remains possible but non-unique in 833. No
published oracle was contradicted under the current formalisation. The affected
rate differs substantially by query family, making a single aggregate score
insufficient for analysis.

These results do not establish that the hidden source map was incorrect. They
show that its selected answer is not always identifiable from the released
text. The result remains conditional on the declared ontology and candidate
scope; representative cases and sensitivity checks are therefore retained as
part of the audit.

== SpatialMap-TQA-Corr

SpatialMap-TQA-Corr preserves the released 1,500 questions and their four
ordinary options. It adds option E, _Cannot be determined_. Under `SINGLE`, the
667 uniquely entailed questions retain their ordinary answer and the 833
underdetermined questions select E.

The corrected dataset is derived deterministically from the audit. It does not
contain coordinates or expose one arbitrarily chosen satisfying map. The
original benchmark remains available unchanged for paired comparison.

== LLM Evaluation Plan

The practical effect of the correction will be measured on a compact LLM-only
panel containing general instruction-tuned, reasoning-oriented, and compatible
text-spatial fine-tuned models. Each checkpoint will be evaluated under matched
instructions and decoding on:

+ original four-option SpatialMap-TQA;
+ five-option SpatialMap-TQA-Corr;
+ the unchanged uniquely entailed subset; and
+ the ambiguous subset.

The analysis will report strict accuracy, _Cannot be determined_ precision and
recall, impossible-answer rate, invalid outputs, paired fixes and regressions,
and changes in model ordering. On ambiguous questions, responses will be split
between the published oracle, another solver-possible answer, an impossible
answer, and the undetermined option.

The position of option E will be permuted in a robustness condition to detect
letter-position shortcuts. The larger prompting study belongs to the later
learning experiments; this comparison uses one shared task prompt and keeps any
model-native reasoning mode separate.

== Remaining Work and Limitations

Representative unique and ambiguous cases still require manual inspection and
the assumption-sensitivity analysis is not complete. The LLM landscape results
are also pending. These checks determine whether the current audit should be
retained unchanged or revised before SpatialMap-TQA-Corr is treated as frozen.

The case study cannot recover the authors' original coordinates or inspect
their unpublished generator. Its conclusion is restricted to entailment from
the released textual questions under the stated formal semantics.
