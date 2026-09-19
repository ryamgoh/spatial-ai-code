# V13 spatial reasoning semantic contract

This document defines v13 behavior independently of earlier dataset versions.
V6 and v12 remain useful comparison suites, but they are not the authority for
v13 semantics.

## World model

Every premise is asserted as an objectively true statement about one spatial
world. The world is represented by two strict orders:

- X: West `<` East
- Y: South `<` North

A cardinal statement constrains one axis. A diagonal statement constrains both.
For example:

```text
A is East of B       → B < A on X
A is North of B      → B < A on Y
A is Northeast of B  → B < A on X and B < A on Y
```

Strict ordering is transitive. A relation is logically proven only when the
corresponding directed path exists in the stated graph. Coordinates used while
constructing consistent random worlds are not evidence and are never visible
to grading.

## Global consistency

A strict-order cycle on either axis makes the complete premise set
inconsistent. Because all premises claim objective truth, one cycle anywhere
invalidates the complete world, including a cycle disconnected from the queried
objects.

```text
any X or Y cycle
    → inconsistent world
    → Cannot be determined for every question family
```

Cycle axis, direct/indirect topology, length, and query-connected/disconnected
placement affect diagnostic difficulty only. They never change the gold rule.

## Consistent-world question semantics

### Type 0: direction of one object relative to another

- Both axes proven: select the unique matching compound direction.
- Exactly one axis proven: both compatible compound directions remain valid.
- Neither axis proven: select `Cannot be determined`.
- Exactly one of a two-direction answer pair is listed: select
  `Cannot be determined`; choosing the listed member would invent the missing
  axis.
- A proven unique direction absent from the menu: select
  `None of the Options`.

### Type 1: which listed objects satisfy a direction

This is a logical-entailment question. Select every listed object whose
relationship to the reference is proven on every required axis. Hidden
generator coordinates do not count. If no listed object is proven, select
`None of the Options`.

### Type 2: how many objects satisfy a direction

Count every object whose relationship to the reference is proven on every
required axis. Zero is a valid count. If the derived integer is absent from the
menu, select `None of the Options`.

For consistent Type 1 and Type 2 worlds, `Cannot be determined` is not a
fallback for an empty result; it is reserved for global inconsistency.

## Option meanings

- `Cannot be determined`: the Type-0 relation lacks sufficient information, an
  incomplete two-direction menu cannot identify a unique full direction, or
  the complete world is inconsistent.
- `None of the Options`: the world is consistent and the logically derived
  answer is absent from the listed content choices.

## Taxonomy

V13 has 12 consistent-world base semantic subtypes:

```text
dir-1, dir-2, dir-undetermined, dir-incomplete, dir-omit
which-1, which-2, which-3, which-4, which-0
count-1, count-omit
```

Cycle is not a base subtype. It is an orthogonal world property applied through
`CycleSpec`. After global invalidation, the solver reports `dir-cycle`,
`which-cycle`, or `count-cycle`, while the generated row retains the original
`base_semantic_subtype`.

## Source of truth

The rendered prompt is authoritative. The generator must reparse it through
`SpatialSolverV13.solve_and_analyze` and may emit a row only when the solver
confirms its requested semantics and structural constraints. Assistant traces
must use only solver-parsed entities, relations, proof paths, and cycle
witnesses.
