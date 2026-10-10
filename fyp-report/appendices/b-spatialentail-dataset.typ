#pagebreak(weak: true)
#heading(level: 1, numbering: none)[Appendix B — SpatialEntail Dataset Records]

Each accepted base problem records its query and semantic cell, source-generation
mode, canonical structural signature, measured difficulty, and rejection history.
Rendered variants carry distinct row identifiers while sharing the base identity.
The workload manifest records requested variants, split membership, overlap
validation, answer-position distributions, and context-admission metadata.

The frozen dataset release should archive the exact matrix, tokenizer revision,
chat template, generation seed, software revision, split fingerprints, rejection
counts, and materialized arm paths. These artifacts are required to reproduce a
pilot; this draft does not claim that a final release has been frozen.


#heading(level: 2, numbering: none)[Executed Renderer Example]

The following excerpt was produced by the Natural Direction training renderer
from a direct two-object fixture: A is Northeast of B, queried from A to B,
with supporting coordinates A=(1,1) and B=(0,0). The fixture is deterministic
and has no generation seed. It illustrates rendered evidence rather than a
sample from the planned depth-controlled pilot. The coordinates are used to
check the model; the training output prints only qualitative orders.

#quote[
  P1: A is Northeast of B. \
  Q-DIR: Therefore A is Northeast of B, as established by P1. \
  X west to east: B < A. \
  Y south to north: B < A. \
  This supporting model satisfies every premise and makes the claim that A is Northeast of B true. \
  Therefore the unique direction is Northeast; all other exact directions are excluded because exact directions are mutually exclusive.
]
