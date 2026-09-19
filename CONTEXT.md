# Spatial Reasoning Experiments

This context defines the canonical language for generated spatial worlds,
questions, and diagnostic dimensions.

## Language

**World**:
One set of objectively asserted spatial premises interpreted together. A world
is either consistent or inconsistent independently of the question asked.

**Question family**:
The requested output form: `direction`, `which`, or `count`.
_Avoid_: Cycle subtype

**Semantic subtype**:
One of the 12 base question semantics, such as `dir-1`, `which-2`, or
`count-omit`. A generated row retains it when its world is made inconsistent;
a prompt-only analysis may leave it unknown after global invalidation.
_Avoid_: `dir-cycle`, `which-cycle`, `count-cycle`, base semantic subtype

**World consistency**:
Whether both strict-order graphs are acyclic. Any cycle on either axis makes
the complete world inconsistent.
_Avoid_: Cycle question type

**Cycle specification**:
The diagnostic description of an injected contradiction: axis, topology,
length, and placement. It is orthogonal to question family and semantic
subtype.

**Matched open-chain control**:
A consistent world with the same question family, semantic subtype, cycle
placement intent, relation count, and entity budget as a closed-cycle world,
but whose final edge does not close the cycle.
