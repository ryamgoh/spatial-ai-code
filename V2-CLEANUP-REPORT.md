# Spatial V2: Remaining Code Assessment

Assessed 2026-11-02; updated through selector/CPU workflow retirement after `78ab068`.
Current authority: [UNIQUE-ONLY-v1.md](experiments/spatialentail-composition-v1/UNIQUE-ONLY-v1.md).

## Conclusion

**Not every remaining v2 file is needed by the current experiment.**
The obvious historical packages are gone. The remaining problem is mostly
obsolete workflows embedded inside modules that also contain useful functions,
plus test fixtures that depend on deleted artifacts.

Do not delete every file outside a runtime trace, or merge unrelated code just
to reduce the file count. The next pass should remove obsolete functions and
their dependencies, retain the scientific checks, and repair test ownership.

## Current Inventory

Tracked files only; caches and generated outputs are excluded.
Python line counts include comments and blank lines and describe the current
working tree, excluding deleted tracked paths. No replacement files were added.

| Location | Files | Python files | Python lines |
|---|---:|---:|---:|
| v2 top level | 84 | 75 | 28,160 |
| `study/` retained analysis | 9 | 8 | 2,090 |
| `tests/` | 77 | 77 | 18,120 |
| **Total** | **170** | **160** | **48,370** |

There are **83 implementation Python files, 77 test files, and 10 docs/configs**.
`legacy/` and `hitl/` no longer exist. `study/` contains seven analysis/audit
utilities, an initializer, and a README; it no longer orchestrates experiments.

The current entry points are `generate_and_prepare.py`, `prepare_evaluation.py`
under the composition experiment, `entailment_thinking_training.py`,
`entailment_evaluation.py`, and `finetune/finetune.py`.
Earlier import/runtime inventories predate the latest removals. They are not
current usage counts: an import is not proof that every function is needed,
and a sampled test run is not a complete dead-code detector.

## Completed In This Pass

Removed nine files, including eight under v2:

- `entailment_training_selection.py` and its exclusive tests: old menu MILP selector.
- `composition_execution.py` and its coordinator tests.
- `composition_execution_benchmark.py`: old timing driver.
- `entailment_coverage.py` and its coverage-draft tests.
- `test_composition_cpu_launcher.py`: tested an already-deleted launcher.
- Experiment helper `audit_unique_training_export.py`: auditor for the retired
  selection/export workflow, not the active controlled-training auditor.

The classifier now belongs to `entailment_replacement_allocation.py`, unchanged.
The existing serialization module owns `atomic_json`, preserving encoding,
flush/fsync, and replace behavior. The six development cases now live in their
only remaining consumer, `test_entailment_dataset.py`; all six match exactly.

Retained production tests use explicit test-only policies instead of deleted
configuration files or the old draft compiler. Admission, naming, depth, quota,
contract, and gold-free export checks remain. Cached-versus-uncached admission
equality is checked across the existing pinned-tokenizer test matrix.

This pass removes a net **1,999 Python lines** across v2 and the experiment.
No frozen data, audit receipts, adapters, or evaluation results were modified.

## Priority Findings

### 1. Related Missing Fixtures Remain

Composition-production and retained entailment-production tests now use the
self-contained [test policy factory](spatial/v2/tests/__init__.py). The old
coverage-draft and coordinator tests are removed. Missing-config references
remain in scene-sampler, targeted-sampler, and diversity tests.

**Implication:** successful collection of 4,582 tests is not a green full suite.
For live behavior, replace external historical fixtures with small explicit
test policies. For retired workflow behavior, remove its exclusive tests.
Do not replace every missing fixture with today's frozen config: that may
silently change what the historical test was asserting.

### 2. Naming Vocabulary Coupling Remains

**Remaining action:** the module still imports `ENTITY_NAMES` from the generic
`generation.py`. Give that fixed vocabulary a lightweight owner without copying
divergent lists or changing name ordering and seeded output.
See [module](spatial/v2/entailment_names.py).

### 3. Shared Evaluation Helpers Still Depend On The Historical Trainer

`entailment_evaluation.py` imports `score` and `adapter_hash` from
`composition_evaluation.py`. The obsolete runner and selection entry points
were removed in `78ab068`; the shared helpers remain.

**Remaining coupling:** shared helpers still import constants and hashing from
`composition_training.py`, which also contains historical preparation code.
This pass did not extract that trainer's helpers or duplicate scoring.
See [shared helpers](spatial/v2/composition_evaluation.py).

### 4. Prototype Drivers and Old CLIs Are Embedded in Live Construction Code

- `run_all_controlled_prototype.py` remains a tokenizer-state provider for current
  generation. Its `initialize`, `TOKENIZER`, and historical tokenizer `REVISION`
  are live; its prototype CLI is a separate concern.
- Current construction uses `Story.problems`, `story_key`, and `renamed_problem`
  from `composition_pilot.py`; that module also owns `run_pilot` and `write_pilot`.
- `composition_feasibility.py`, `composition_balance.py`, and
  `composition_selection_expansion.py` contain CLI paths referencing deleted
  `research-analysis` protocol files. Their reusable seed, topology, proof, and
  scene functions remain live.
- `composition_training.py` mixes live `messages`, `prompt_text`,
  `tokenize_example`, and `completion_suffix` with old preparation/sealing.
  These token helpers are actually used by the historical admission gate before
  native thinking re-tokenization; they cannot simply be removed.
- `entailment_retention.py` still combines the live `matches` helper with old
  reserve-selection machinery. `entailment_unique_study.py` also imports its
  validator, and `entailment_production.make_gate` imports its completion gate.
  Removing the separate one-menu selector did not remove this older layer.

**Action:** separate or trim obsolete CLI sections before moving reusable
functions. Keep generation output identical. The old tokenizer gate is not an
accidental choice of training model; changing it would change admission behavior.

### 5. Fixed Benchmark Reconstruction Still Needs Historical Scheduling

`prepare_evaluation.py` reconstructs the unchanged 4,800-base inventory using
the historical source schedule. `composition_production.py` is therefore not
obsolete as a whole. Current generation also calls its allocation functions.

**Action:** retain schedule/allocation behavior, even if the old production CLI
is retired. Any extraction must preserve all fixed query fingerprints, seed
ordering, quotas, and source identities. This is more consequential than moving
an inference helper.

## Other Retained Utilities

These are not all current-study entry points. Retention depends on their
callers and scientific purpose, not merely on whether a test imports them.

| Modules | Current purpose and recommendation |
|---|---|
| `entailment_external.py` | External adaptation and independent verification. Keep; current inference consumes its outputs rather than calling it. |
| `audit_spatialeval.py`, `spatialeval_adapter.py` | Source-data audit and parsing; adapter also used by solver tests. Keep until source-audit requirements are settled. |
| `evaluation_chat.py` | Shared by historical analysis and `eval/eval_comparators.py`. Keep. |
| `context_budget.py`, `workload_manifest.py`, `generate_all.py` | Generic atomic workload tools and shared-task support. Not the current dataset launcher; decide whether to retain that public capability as a unit. |
| `consistency.py` | Standalone consistency-pair generation; now only detected Python callers are its tests. Candidate for retirement, not a dependency of the shared solver. |
| `entailment_positions.py` | Old menu-position rerenderer; exercised by position and scene-protocol tests, with fixtures reused by auditors. Move useful fixtures and protect menu semantics before removal. |
| `entailment_controlled_coverage.py` | Reference recipes used by generator equivalence and semantic tests. Keep the oracle until equivalent independent fixtures exist. |
| `study/__init__.py` | Required package initializer, not dead because it has no explicit import statement. |
| `study/metrics.py`, `study/analyze_randomized.py`, `study/summarize_randomized.py`, `study/summarize_corr.py`, `study/summarize_transfer_diagnostic.py`, `study/gen_fig_randomized.py`, `study/training_token_audit.py` | Deliberately retained historical result analysis/auditing. Keep independently of dataset generation. Frozen-source checks may require original snapshots. |

## Retired Formula Language

`Implies` and `Iff` are removed from the solver and from proof construction,
model checking, text parsing/rendering, symbolic trace codecs, structural
fingerprints, and complexity measurement. The exclusive `MODUS_PONENS` and
`IFF_ELIMINATION` rules and antecedent-materialization helper are removed too.
Repository Python searches find no remaining references to those symbols.

This intentionally narrows the supported input language. Old implication and
equivalence text/trace payloads are rejected, not silently translated.
Historical result labels and frozen artifacts remain unchanged; replaying old
Boolean certificates requires their original source revision.

`And`, `Not`, and `Or` remain because current entailment/counterexample checks
and complete joint-count assignments use them internally. The current study
validator still requires flat conjunctions of exact spatial atoms as premises.
Removing all logical operations would break that verification.

This retirement was committed in `78ab068`. Current Count and branch-isolation
checks remain; the latest workflow cleanup did not change the solver.

## Tests, Configs, and Provenance

- The 77 test files contain real scientific checks as well as stale
  workflow-specific cases. Review assertions and fixtures, not filenames.
- Seven top-level benchmark YAML files remain as historical specifications.
  No executable filename references were found in searched Python/shell/YAML.
  They are provenance, not current generation inputs; remove only if their
  documentary value is no longer needed.
- The v2 README and context now identify the supported formula language and
  link to the current Unique-only specification. Historical vocabulary is
  explicitly separated from current capabilities.
- `generate_and_prepare.py` hashes every top-level v2 Python file and experiment
  Python helper. Old preparation verification is thus coupled to unrelated
  files. Preserve old receipts; a narrower source inventory needs a new version.
- The repository validator has 23 known layout/archive findings. These include
  treating CPU jobs as requiring GPU directives and archived launcher references.
  Do not add GPU resources to CPU jobs merely to satisfy outdated validation.
- `experiments/15-v2-ablation/jobs/common.bash` still provides credentials and
  environment helpers to current training. Its historical location does not
  make it removable.

## Recommended Next Pass

1. Replace the remaining missing historical fixtures for live scheduler/solver
   behavior with explicit test fixtures.
2. Detach prototype tokenizer state and naming constants from heavy runners.
3. Separate the position-rerenderer's reusable test fixtures before retiring it.
4. Reassess larger pilot, training, and production modules. Do not
   replace fixed benchmark scheduling merely to get a cleaner directory tree.

Completion criteria: no newly broken imports or launcher references; all
retained targeted tests pass; current generation produces identical reference
records; fixed benchmark fingerprints and native token masks remain identical;
scoring and dev selection remain unchanged. No retraining or dataset
regeneration is required just to conduct this code cleanup.

## Verification Scope

- Baselines: 46 affected tests, 11 retained production tests with explicit
  fixtures, and 16 new classifier/atomic-write safeguards passed before removal.
- Post-removal: **159 retained allocation, dataset, external, and production
  tests passed**, including the expanded solver-reuse assertions.
- A further **1,209 tests and 5 subtests passed** across active generation,
  native thinking serialization, evaluation jobs/batching, solver reuse, and
  historical analysis/chat helpers. Total: **1,368 distinct focused tests and
  5 subtests**, using Python 3.12 and Z3 4.16 within the declared version range.
- All **4,582 tests collect**. This is not a full-suite execution.
- AST comparison confirms the moved classifier and all existing allocation
  functions/classes are unchanged. All six relocated development jobs match
  their originals exactly.
- Searches found no executable references to the retired modules or auditor.
  Frozen manifests still name historical sources intentionally.
- Full changed-file Ruff checks and whitespace checks pass.

No GPU inference or full 4,800-base admission campaign was run. External
scripts outside this checkout are not audited. Old prepared packages must
still be verified under their original source revision. This report does not
claim a definitive minimal file count or that every remaining function is live.
