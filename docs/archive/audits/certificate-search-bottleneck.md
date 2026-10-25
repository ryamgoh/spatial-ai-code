# Certificate construction bottleneck: inert tautology pruning

Status: fixed, 2026-10-14. Found while building the possibility (set-valued)
generator for `SpatialEntail-ALL-POSSIBLE`. No sealed `SpatialEntail-SINGLE`
artifact is affected: certificates are only rebuilt when new rows are
generated.

## Symptom

Generating four multi-answer Direction samples with the new possibility
skeletons took about 1,180 seconds. At that rate a few thousand benchmark rows
would take days, which blocks scaling any new bench.

## Diagnosis

The solver was not the cost. For one 7-object problem, Z3 analysis took under
0.1 s while building the answer certificate took 4.8 s and made about 11,700
recursive case-split calls.

The cause is in `build_formula_refutation` (`spatial/v2/proofs.py`). To prove a
candidate answer impossible, the builder assumes the candidate and searches for
a contradiction. When saturation is not enough, `_derive_contradiction_by_cases`
splits on every accessible disjunction, closing each branch by contradiction or
by splitting further.

Generators pad problems to a fixed premise count with excluded-middle fillers:
`(A is to the North of B) OR (NOT (A is to the North of B))`. These are true in
every world and carry no answer information, but each one is a disjunction. The
case-split search tried them, and when a split failed it backtracked and tried
the next combination. With six fillers the search was effectively exponential in
the filler count.

SINGLE generation rarely hit this because its unique-answer templates usually
close refutations by saturation or a direct axis contradiction. Possibility
problems leave several answers open, so more candidates need case reasoning.
That is when the fillers dominate.

## Fix

`_inert_tautologies(problem)` finds top-level premises of the form `P OR NOT P`
whose atom concerns an object pair that no other premise mentions.
`_derive_contradiction_by_cases` skips those disjunctions.

Why this is sound: splitting on `P OR NOT P` yields one branch assuming `P` and
one assuming `NOT P`. If no other premise mentions the pair, neither assumption
can interact with any other fact, so a branch can close only if it would have
closed without the assumption. The split adds search cost without enabling any
proof. Tautologies about pairs that other premises do mention are still
considered, because their branches can matter.

The fix changes the proof search only. The checker, the certificate format, and
the replay rules are unchanged, so every certificate is still independently
replayed by `check_formula_refutation`.

## Evidence

| Measurement | Before | After |
|---|---:|---:|
| Case-split calls, one 7-object Direction problem | ~11,700 | 6–12 |
| Certificate time, same problem | 4.8 s | 0.03 s |
| Four multi-answer Direction samples | ~1,180 s | 0.8–1.3 s |
| Four samples per cell, 25 width-varied cells | not feasible | 0.8–2.9 s per cell |

Soundness check: across 1,248 refutation attempts on premise-first Boolean
problems (case split, disjunctive syllogism, modus ponens, nested case split),
the pruned and unpruned searches agreed on every claim (0 mismatches). The full
`spatial/v2` test suite passes (505 tests).

## Related proof-builder fixes in the same change

Two builder defects surfaced once multi-answer Boolean problems were generated.
Both were in proof construction, not in the checker.

1. **Enclosing-branch axis facts.** `_append_axis_contradiction` only looked at
   axis facts in the current branch or the global scope. Inside a nested case
   branch it missed facts derived in the enclosing refutation branch, so valid
   refutations failed with "missing a model for non-refutable candidate". It now
   uses `_accessible_axis_steps`, which applies the same ancestor-scope rule as
   the checker's `_scope_is_accessible`.
2. **Branches that close without their assumption.** With the wider visibility, a
   case branch could close using only enclosing facts, never citing its own
   assumption. Reachability pruning then dropped that assumption, and the
   checker rejected the certificate for depending on another branch. Such a
   split is unnecessary, so the builder now treats a branch that closes without
   its own assumption as failed, and tries another disjunction or a simpler
   proof.

Regression test:
`test_case_split_with_divergent_consequents_is_fully_certified` in
`spatial/v2/tests/test_answer_certificates.py`.
