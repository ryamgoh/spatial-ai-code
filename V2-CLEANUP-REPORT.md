# Spatial V2: Remaining Code Assessment

Assessed 2026-11-02; updated through implication/equivalence retirement after `c1f4360`.
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
working tree. The fixture, naming/evaluator, and formula cleanup added no files.

| Location | Files | Python files | Python lines |
|---|---:|---:|---:|
| v2 top level | 88 | 79 | 29,367 |
| `study/` retained analysis | 9 | 8 | 2,090 |
| `tests/` | 81 | 81 | 18,722 |
| **Total** | **178** | **168** | **50,179** |

There are **87 implementation Python files, 81 test files, and 10 docs/configs**.
`legacy/` and `hitl/` no longer exist. `study/` contains seven analysis/audit
utilities, an initializer, and a README; it no longer orchestrates experiments.

The current entry points are `generate_and_prepare.py`, `prepare_evaluation.py`
under the composition experiment, `entailment_thinking_training.py`,
`entailment_evaluation.py`, and `finetune/finetune.py`.
The pre-cleanup assessment found a conservative static import closure of
**66 of the 87 v2 modules**; that closure was not remeasured after function removal.
That is potential dependency reachability, not proof that every function is used.

## Evidence Beyond Imports

Before the latest function cleanup, two read-only CPU probes recorded Python
calls while executing existing tests:

| Probe | Result | Loaded v2 modules | Modules with observed function/method calls |
|---|---|---:|---:|
| Controlled training generation through audits and native thinking tokenization | 1 integration test passed | 64 | 28 |
| External benchmark loading and scoring regression cases | 29 tests passed | 27 | 8 |

The generation probe selects at least 20 construction cases spanning query,
answer category, depth endpoints, and contingent Which cases. The evaluation
probe checks the existing external payload and scoring fixtures; it does not
perform GPU inference or simulate a complete production evaluation.

Call counts exclude class-body execution during imports, lambdas/comprehensions,
and tests themselves. Loaded-module counts include test imports. Class definitions,
constants, and unexercised branches can still be required even when no function
call was observed. These probes are evidence for focused inspection, not a
dead-code detector. Temporary instrumentation/results are outside the repository.

## Priority Findings

### 1. Two Confirmed Fixture Failures Fixed; Related Fixtures Remain

These two tests originally failed during fixture setup and are now fixed:

- `test_composition_production.py::test_exact_quota_totals_and_depth_weights`:
  missing `production-config-v1.json`.
- `test_entailment_coverage.py::test_preserve_sizes_and_numeric_origins_without_freezing`:
  missing `production-allocation-v1/execution-config-v1.json`.

Both test modules now use the self-contained
[test policy factory](spatial/v2/tests/__init__.py). Numerical settings and all
36 coverage rules were checked against the original policies in Git. The CLI
test writes its draft config to pytest's temporary directory. Neither old
experiment folders nor production settings were restored.

All **110 tests across these two modules pass**, and changed-file lint passes.
The same missing-config paths still appear in scene-sampler, targeted-sampler,
diversity, execution, and entailment-production fixtures; those separate modules
were not changed in this focused fix.

**Implication:** successful collection of 4,605 tests is not a green full suite.
For live behavior, replace external historical fixtures with small explicit
test policies. For retired workflow behavior, remove its exclusive tests.
Do not replace every missing fixture with today's frozen config: that may
silently change what the historical test was asserting.

### 2. Scoped Naming Migration Removed; Vocabulary Coupling Remains

The generation probe calls `story_name_map` and `rename_structure`.
Current non-test callers import those functions and `NAMES`.
The scoped-record `rename_record` and its helper `semantic_record` are now
removed, along with migration-only imports and tests. Deterministic naming and
recursive renaming function bodies are structurally unchanged. Multiword-name,
independent-auditor, and matched-evidence tests remain.

**Remaining action:** the module still imports `ENTITY_NAMES` from the generic
`generation.py`. Give that fixed vocabulary a lightweight owner without copying
divergent lists or changing name ordering and seeded output.
See [module](spatial/v2/entailment_names.py).

### 3. Obsolete Evaluator Entry Points Removed; Shared Helpers Retained

`entailment_evaluation.py` imports `score` and `adapter_hash` from
`composition_evaluation.py`. The old `run_gpu`, `verify_seal`, `select_epoch`,
JSONL reader, and CLI are now removed. No executable callers of the removed
entry points remain in the repository search.

Shared scoring, adapter identity, parsing, answer-contract, and replay helpers
are structurally unchanged. The Unique-study selection test now exercises
`entailment_evaluation.choose`; separate stage-accuracy and token-total assertions
remain. The current thinking evaluator itself was not modified.

**Remaining coupling:** shared helpers still import constants and hashing from
`composition_training.py`, which also contains historical preparation code.
This pass did not extract that trainer's helpers or duplicate scoring.
See [shared helpers](spatial/v2/composition_evaluation.py).

### 4. Training Generation Imports Old Selection Machinery for One Classifier

`build_controlled_training.py` and `entailment_replacement_allocation.py` import
`answer_category` from `entailment_training_selection.py`. The generation probe
calls only that function in this module. The rest implements a previous
candidate-menu MILP selection workflow.

This is not an immediately deletable module: the retained
`audit_unique_training_export.py` imports both `answer_category` and `validate`.
That auditor and the old selector's tests need an explicit retention decision.

**Action:** isolate the category rule with current contract/allocation logic;
retire the old selection workflow with its old export auditor if it is no longer
needed. Keep current exact quotas and no-refill checks unchanged.
See [classifier and selector](spatial/v2/entailment_training_selection.py#L11).

### 5. Prototype Drivers and Old CLIs Are Embedded in Live Construction Code

- `run_all_controlled_prototype.py` remains a tokenizer-state provider for current
  generation. Its `initialize`, `TOKENIZER`, and historical tokenizer `REVISION`
  are live; its prototype CLI is a separate concern.
- The generation probe uses `Story.problems`, `story_key`, and `renamed_problem`
  from `composition_pilot.py`; that module also owns `run_pilot` and `write_pilot`.
- `composition_feasibility.py`, `composition_balance.py`, and
  `composition_selection_expansion.py` contain CLI paths referencing deleted
  `research-analysis` protocol files. Their reusable seed, topology, proof, and
  scene functions remain live.
- `composition_training.py` mixes live `messages`, `prompt_text`,
  `tokenize_example`, and `completion_suffix` with old preparation/sealing.
  These token helpers are actually used by the historical admission gate before
  native thinking re-tokenization; they cannot simply be removed.

**Action:** separate or trim obsolete CLI sections before moving reusable
functions. Keep generation output identical. The old tokenizer gate is not an
accidental choice of training model; changing it would change admission behavior.

### 6. Fixed Benchmark Reconstruction Still Needs Historical Scheduling

`prepare_evaluation.py` reconstructs the unchanged 4,800-base inventory using
the historical source schedule. `composition_production.py` is therefore not
obsolete as a whole. The generation probe also calls its allocation functions.

**Action:** retain schedule/allocation behavior, even if the old production CLI
is retired. Any extraction must preserve all fixed query fingerprints, seed
ordering, quotas, and source identities. This is more consequential than moving
an inference helper.

## Modules Outside the Assessed Static Closure

This inventory accounts for the implementation modules not in the 66-module
closure at the pre-cleanup assessment. It does not label them all as disposable.

| Modules | Current purpose and recommendation |
|---|---|
| `entailment_external.py` | External adaptation and independent verification. Keep; current inference consumes its outputs rather than calling it. |
| `audit_spatialeval.py`, `spatialeval_adapter.py` | Source-data audit and parsing; adapter also used by solver tests. Keep until source-audit requirements are settled. |
| `evaluation_chat.py` | Shared by historical analysis and `eval/eval_comparators.py`. Keep. |
| `context_budget.py`, `workload_manifest.py`, `generate_all.py` | Generic atomic workload tools and shared-task support. Not the current dataset launcher; decide whether to retain that public capability as a unit. |
| `consistency.py` | Standalone consistency-pair generation; now only detected Python callers are its tests. Candidate for retirement, not a dependency of the shared solver. |
| `entailment_coverage.py` | Old review-only gate compiler; current detected callers are coverage/production tests. Candidate for retirement after separating tests of current gates. |
| `entailment_positions.py` | Old menu-position rerenderer; exercised by position and scene-protocol tests, with fixtures reused by auditors. Move useful fixtures and protect menu semantics before removal. |
| `entailment_controlled_coverage.py` | Reference recipes used by generator equivalence and semantic tests. Keep the oracle until equivalent independent fixtures exist. |
| `composition_execution_benchmark.py` | Old timing CLI also provides `development_jobs` to retained dataset tests. Its naming-test caller was removed. Move the small fixture builder into existing test support, then retire the timing runner. |
| `composition_execution.py` | Old CPU coordinator. External preparation still imports `atomic_json`; tests and the benchmark use more. Isolate the atomic-write helper and needed fixtures before retiring the coordinator. |
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

Tests exclusive to the retired rules were removed. Count-correlation, nested
case search, branch isolation, incomplete-case rejection, model truth checks,
and structural invariance assertions were retained with supported fixtures.
New tests reject retired text syntax, nested symbolic formula kinds, and rule
payloads. This pass changed eight implementation modules and eight test files,
removing a net 532 Python lines without introducing replacement modules.

## Tests, Configs, and Provenance

- The 81 test files contain real scientific checks as well as stale
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

1. Repair test ownership: retire obsolete coverage-draft tests; replace missing
   fixtures for live scheduler/solver behavior with explicit test fixtures.
2. Detach prototype tokenizer state and naming constants from heavy runners.
3. Isolate `atomic_json` and development fixtures, then remove the obsolete CPU
   coordinator/timing tool where no callers remain.
4. Only then reassess larger pilot, training, and production modules. Do not
   replace fixed benchmark scheduling merely to get a cleaner directory tree.

Completion criteria: no newly broken imports or launcher references; all
retained targeted tests pass; current generation produces identical reference
records; fixed benchmark fingerprints and native token masks remain identical;
scoring and dev selection remain unchanged. No retraining or dataset
regeneration is required just to conduct this code cleanup.

## Verification Scope

The retirement commit is `3e24dc2`: 107 changed files, 22,612 deleted lines, and
266 added lines. Prior retirement verification passed 248 focused tests and
18 subtests, and collected 4,694 v2 tests. Those were selected checks, not a
full-suite pass.

This deeper assessment added the two runtime-call probes above (30 passing
tests total) and explicitly confirmed two fixture-setup failures. It also
rechecked source imports, function callers, deleted-path references, and current
file counts. No GPU inference or full 4,800-base admission run was executed.
The subsequent fixture fix changed only test support and two test modules;
110 tests passed.

The preceding naming/evaluator cleanup changed two implementation modules and
three test files: 384 lines removed and 16 added, a net reduction of 368 lines.
Its baseline passed 83 tests; post-cleanup verification passed **92 focused
tests**, including generation, current evaluation, batching, and job contracts.
At that point **4,688 v2 tests collected**; changed-file lint passed. AST comparison confirmed
the retained naming and scoring/identity/contract helper bodies are unchanged,
and source searches found no executable references to the removed functions.
This is not a full-suite or GPU validation. Source hashing still means old
prepared packages must be verified under their original source revision.

Formula-retirement verification:

- Baseline: 834 solver/certificate tests passed. Final core run: **751 passed**
  with Z3 4.16.0.0, within the project's declared `>=4.15,<5` range.
- Extended run: **2,236 tests and 5 subtests passed**, including core solver
  checks, generic/active generation, current contract/evidence/depth checks,
  native thinking tokenization, batching, evaluation jobs, and retained
  historical analysis. Together with the separate 213-test run below,
  **2,449 distinct focused tests and 5 subtests passed**. The extended run used
  cached Z3 5.3.3.0; the core was separately rechecked in the declared range.
- Current full collection: **4,605 tests**, not a full-suite execution.
- Before/after comparison against the original eight modules at HEAD:
  **53 cases byte-identical**, covering 13 paired Unique explanation views and
  40 proof/refutation cases across eight directions and depths 1, 2, 4, 8, 10.
  Comparison SHA256: `7d80786f2cc5d3f470fc6387b5afff56d56b68b32ce923ade395f51eb39d5977`.
- An additional 213 proof-consumer, evidence-audit, difficulty, and supervision
  regression tests passed. AST comparison found 170 unchanged functions in the
  eight implementation modules, 18 modified formula/rule handlers, and the one
  removed antecedent-materialization helper.
- Scoped syntax/import lint (`E4,E7,E9,F,I`) and whitespace checks pass.
  Existing `UP035`, `RUF007`, and `ISC004` warnings in the metrics/generation
  test files were reproduced at HEAD and remain outside this pass.

Runtime probes cover sampled paths, not all supported inputs. The assessed
static closure includes lazy imports and cannot distinguish constants from
functions by itself. External scripts outside this checkout are not audited.
Accordingly, the report identifies specific safe-to-investigate boundaries
rather than claiming a definitive minimal file count.
