# Entailment-Only Migration Checkpoint

## Completed

- The pivot and experiment entry point now identify the entailment-only amendment
  as active, with the previous possibility-based design retained as history.
- New interface `spatialentail-entailment-v1` and distinct contract identifiers
  prevent accidental interpretation of old prompts and labels as the new task.
- Direct full-domain entailment analysis retains a checked counterexample for
  every non-entailed value and does not canonicalize every object pair.
- Unique uses the user's sole-individually-entailed-candidate rule. Listed selects
  displayed entailments. Exhaustive requires complete entailed-menu coverage.
- The initial explanation implementation checks positive axis paths, fixed
  membership Count proofs, counterexamples, and inconsistency certificates.
  Numeric coordinate assignments remain audit metadata, not supervision text.
- Admission checks preserve premise/domain round trips, matched references,
  renaming, premise order, verified context removal, menu permutation, and
  pinned-tokenizer/context limits.
- Non-entailed options receive contingent/impossible audit categories; neither
  category is selected merely because a claim can hold.
- Versioned scoring supports both stages, rejects mixed semantic versions, and
  reports Listed-option inclusion/exclusion separately by negative category.
- A draft coverage matrix preserves approved base sizes and numeric origins,
  while explicitly reporting structurally inapplicable difference gates.

## Verification

The migration/shared-infrastructure regression selection passed **522 tests**.
After subsequent coverage-metadata additions, the final entailment-specific
selection passed **155 tests**. These selections overlap; do not sum them.
Ruff checks passed on the changed implementation and test files.

The recorded development workload consists of six queries from the two existing
CPU development scenes. It creates no fresh held-out families and is not a
production candidate. All six queries passed full admission, producing 33
contract/menu reference views per style.

The paired timing receipt `ENTAILMENT-DEVELOPMENT-v3.json` records:

| Path | Local elapsed time |
|---|---:|
| Historical possibility-based full admission | 20.44 seconds |
| Entailment-only full admission | 3.45 seconds |

The maximum reference lengths were 644 Natural tokens and 622 Symbolic tokens
under the pinned tokenizer. These maxima do not establish a general style
advantage. The two paths answer different tasks; the check compares their
underlying entailed sets, not identical golds or explanation text. Timings are
not production-wide estimates or Slurm measurements. Each receipt binds the
source snapshot used at measurement time, not a production software freeze.

The coverage draft `ENTAILMENT-COVERAGE-DRAFT-v1.json` contains 105 reviewable
rules and numerical provenance. It is deliberately not executable or frozen.
Listed and Exhaustive always coincide for entailment-only Direction and Count;
the six old split/query difference minima are marked structurally inapplicable,
not silently treated as satisfied.

## Remaining Work

1. Complete the supported-evidence audit beyond the checked fixtures. The initial
   positive renderer requires axis derivations; Count proofs currently require
   fixed memberships. Do not claim completeness for all solver-supported
   entailments or silently drop unsupported categories.
2. Validate source supply for the revised matrix, including sole-entailment-plus-
   contingent Selection and both negative categories. Review the proposed
   reused numeric negative minima before freezing the new policy.
3. Integrate entailment-only records, gates, quotas, progress, ordered workers,
   checkpoint identity, and early stopping into production execution. The old
   allocation and launcher still describe the historical task.
4. Version and verify external SpatialMap-TQA-Corr views with the same new
   semantics. Preserve Corr SINGLE and historical releases unchanged.
5. Generate the complete unsealed train/dev/in-range/extrapolation candidate,
   inspect its distribution and reference explanations, and report rejection
   counts and coverage shortfalls. A development receipt cannot satisfy this.
6. Obtain separate approval before sealing or model training.

The approved retained sizes remain 6,000 / 1,200 / 2,400 / 1,200.
The active goal is unfinished. No new production generation, Slurm submission,
seal, or model training was performed during this checkpoint.
