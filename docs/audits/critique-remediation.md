# SpatialEntail critique remediation

Status: implementation goal complete; combined verification and independent review passed. This is the durable checklist for the second
four-reviewer critique. Passing implementation checks does not establish an SFT
benefit, human usability, or fidelity to an LLM's internal computation.

## Acceptance criteria and ownership

| Item | Required resolution | Owner | Status / evidence |
| --- | --- | --- | --- |
| 1. Count size | Reduce redundant exhaustive evidence; preserve joint dependencies; enforce exact context admission for every paired variant. | Certificates, supervision | Implemented: fixed memberships plus correlated residual coverage; Count v2 tampering tests and exact-token admission. |
| 2. Representation confound | State exactly which evidence each format exposes; remove representation-only claims unless dependencies match. | Certificates, report | Implemented and scoped: Natural exposes dependencies; report treats formats as supervision packages pending corpus-wide information/cost audit. |
| 3. Experiment integration | Current view names, controls, split settings; train/dev/test separation; final test never selects checkpoints. | Experiments, parent | Implemented: current views, four SFT arms and three untuned baselines; dev-only checkpoint selection, separate final test. |
| 4. Token budgets | Actual tokenizer and chat template, prompt plus target and evaluation reserve, rejection accounting, paired admission. | Supervision, experiments | Implemented: pinned Qwen chat template, full training/evaluation/generation budgets, bound source/prompt fingerprints and durable rejection reports. |
| 5. Option shortcuts | Reproducibly shuffle special and ordinary options together. | Generation | Implemented: all options shuffled together; paired menus generated once; answer-set and marginal-position reports. |
| 6. Premise-first | Implement visible-premise proposals with solver classification and checked evidence; label provenance honestly. | Generation | Implemented: independently sampled visible formulas, then query and solver/certificate checks; every Boolean syntax family tested. |
| 7. Holdouts | Canonical structural identity with stated limits, three-way overlap validation, explicit predeclared cell holdouts. | Generation, parent, experiments | Implemented: exact/coarse source canonicalization, three-way integrity checks and executed premise-first IFF/implication cell holdouts. |
| 8. Controls | Distinguish absent evidence from invalid evidence; replace misleading line shuffle or narrowly name what it tests. | Supervision | Implemented: deleted shuffled-trace; Symbolic semantic corruption preserves schema/decision but fails replay; answer-only process absent. |
| 9. Difficulty | Include exclusion support or stop calling unmeasured premises distractors. | Generation | Implemented: cumulative semantic deletion includes exclusions/count constraints; jointly sufficient support with timeout-completeness flag. |
| 10. Dead steps | Generated proofs contain only relevant evidence; checked acceptance contract explicitly distinguishes validity/relevance. | Certificates | Implemented: unused proof steps rejected, including unrelated extra inputs smuggled into formula rules; adversarial regressions. |
| 11. Consistency | Direction training includes consistency evidence or explicitly carries a checked consistency contract. | Certificates | Implemented: unique Direction v2 carries required checked witness after the Natural derivation; witness tampering rejected. |
| 12. Audit maps | All audit query kinds provide coordinate-bearing witnesses consistently; training remains qualitative. | Certificates | Implemented: uniform coordinate-bearing audit envelope for all query types; qualitative training witnesses. |
| 13. Natural readability | Concrete claim names, intelligible Boolean scopes and concise Count evidence. | Certificates | Implemented presentation: named claims, full scoped dependency narration and compressed Count evidence; human readability remains unmeasured. |
| 14. Natural validation | State what is checked for generated gold and for model output; do not claim arbitrary prose replay. | Certificates, report | Implemented honest validation: raw Symbolic model output replay and footer checks; Natural/answer-only process metrics unavailable, not invalid. |
| 15. Report integrity | Replace stale names/claims; draft methodology and dataset sections from actual implementation; explicitly mark unrun results. | Report | Implemented: stale names removed; abstract, methods, dataset, analysis and limits drafted; unrun learning results explicitly marked. |

## Verification and cleanup plan

1. Run the existing spatial suite with Z3 installed as a behavior baseline.
2. Add targeted regression cases for each changed correctness boundary before
   claiming the corresponding item resolved.
3. Cleanup pass after each workstream: delete obsolete paths first, consolidate
   duplicated behavior second, then clarify naming. Keep cleanup within touched
   files and rerun focused tests.
4. Run a separate reviewer pass over the integrated diff. Review soundness,
   leakage, context accounting and redundant abstractions independently of authors.
5. Run the full spatial and experiment test suites, lint/format checks, Typst
   compilation and an actual small generation/preparation run with the selected
   tokenizer. Record outputs and limits here before closing the goal.

No remote training or empirical human/XAI study is part of this implementation
goal. Existing uncommitted work is preserved; no commit is requested.

## Verification log

- Baseline: 453 spatial tests passed with Z3 (38.35 seconds); two Typer deprecation warnings.
- Split repairs: 15 focused manifest tests pass, including new development leakage,
  three-way structural clustering, explicit holdouts, missing control detection,
  answer-position counts and metadata-overwrite rejection.
- Evaluation: five loader/scorer tests pass. Full Symbolic output reaches replay;
  a correct final letter with broken evidence fails process validation; a valid
  certificate with a wrong answer footer fails full validity. Natural process
  checking is explicitly unavailable, not counted as an invalid proof.
- Cleanup pass 1 plan: replace duplicated two-way split validation with one
  three-way loop; remove obsolete subset-sum split selection in favor of bounded
  whole-cluster selection; keep legacy task helpers untouched while adding V2
  raw-response scoring. Scoped lint/format checks follow focused regression tests.
- Inference boundary: per-request generation budgets, sampling settings and stop
  strings now reach vLLM; prompt tokens plus the generation reserve are checked
  before generation. The pilot must set `add_special_tokens: false` for its
  pre-rendered prompts. Three GPU-free boundary tests pass, including driver
  forwarding of the optional bootstrap setting.
- The evaluation loader rejects rows without `metadata.evaluation_prompt`.
  This field must come from the admission tokenizer, with chat templating disabled
  in the evaluation configuration. Nine evaluation-loader/process tests pass,
  covering all three query kinds, ambiguity, prefilling of the opening think tag,
  absent reasoning and an incorrect final answer after a valid proof.
- Fresh before-change Qwen3.5-4B diagnostic, one seed-918 item per cell:

  | Query / cell | Natural full chat | Symbolic full chat |
  | --- | ---: | ---: |
  | Direction depth 2 | 290 | 861 |
  | Direction ambiguity 2 | 700 | 3,306 |
  | Which depth 2 | 988 | 3,060 |
  | Which ambiguity 2 | 827 | 2,011 |
  | Count depth 2 | 5,216 | 28,138 |
  | Count ambiguity 2 | 2,419 | 17,058 |

  These are tokenizer diagnostics, not aggregate benchmark results. Runtime:
  Transformers 5.16.1, Tokenizers 0.23.1. Snapshot:
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
  Full counts use `apply_chat_template(tokenize=False)` followed by
  `encode(add_special_tokens=False)`. Calling `len` on a returned tokenization
  mapping would incorrectly count fields instead of tokens.
- After certificate changes, the same seed/policy diagnostic full chats are:

  | Query / cell | Natural full chat | Symbolic full chat |
  | --- | ---: | ---: |
  | Direction depth 2 | 522 | 937 |
  | Direction ambiguity 2 | 1,492 | 3,306 |
  | Which depth 2 | 1,488 | 3,060 |
  | Which ambiguity 2 | 1,173 | 2,011 |
  | Count depth 2 | 1,742 | 3,234 |
  | Count ambiguity 2 | 1,391 | 2,488 |

  Count stores fixed membership evidence once and covers only residual joint
  assignments; complexity can still be exponential in the contingent candidates.
  Natural grew in some cells because it now renders all proof dependencies and
  the unique-Direction consistency witness. These observations do not establish
  performance or readability benefits; every row still needs context admission.
- Certificates: 107 focused tests passed. Context/control modules: 22 focused
  tests passed plus actual cached Qwen chat-template admission. Cross-author reviews then identified and fixed admission order/integrity and proof dependency edge cases.
- Paired workload validation also rejects duplicate variants, differing semantic
  identities within a base ID, or different prompts/gold answers within a menu.
  Nineteen focused manifest tests passed at that stage; the final combined verification below supersedes these interim counts.

## Final combined verification

- **591 tests passed in 33.97 seconds**, covering `spatial`, the actual V2
  experiment runner, task scoring, and both-stage inference preparation. Two
  non-failing Typer deprecation warnings remain.
- Ruff lint passed; all 53 scoped Python files pass formatting. Vulture at
  90% confidence reported no unused-code findings in `spatial/v2`.
- `git diff --check` and Typst compilation passed. The chapter and executed
  example appendix were visually inspected; final PDF is
  `/tmp/spatialentail-report.pdf`.
- Fresh pinned-tokenizer pilot: 80 base problems, 320 rows, 192/64/64
  train/dev/test rows. Both semantic strata have three entities and two premises.
  Maximum full training sequence 4,032 tokens; maximum generated suffix 3,910;
  maximum evaluation prompt plus reserve 4,218. No requested stratum was dropped.
- Fresh premise-first holdout: 16 bases, 64 rows, 32/16/16 train/dev/test rows.
  Atomic training, independently sampled IFF development, independently sampled
  implication test. Maximum training sequence 965 tokens.
- **All 384 rows** passed the real evaluation loader and expected replay/control
  outcomes: checked Symbolic evidence valid, corrupted Symbolic reasoning invalid
  with decision preserved, Natural/answer-only process not applicable, all final
  authored answers correct. This is artifact verification, not model evaluation.
- Final generated artifacts and replay evidence:
  `/tmp/spatial-v2-pilot-repair/final-pilot_*`, `final-holdout_*`,
  `replay-verification.json`, and seven arm configurations in `runs/final-smoke`.
- Independent reviews closed split integrity, reporting metadata, canonical gold
  and footers, admission source binding, target order/multiplicity, generation
  provenance, source canonicalization, and negative/count support. The final optional second-stage driver and ignored-extra-proof-input fixes were independently approved after 65 focused checks. No blocker remains in the reviewed implementation scope.

## Cleanup performed

Removed the legacy line-shuffle control, redundant fixed-membership Count conflict
objects/branches, order-sensitive proof-skeleton signatures, lossy Natural summary
wrappers, stale Delta views and AxisDecomposition names, and AST-extraction test
scaffolding. Shared request preparation now applies the same token-budget rule to
both inference stages. No compatibility fallback or new production dependency was
introduced.

## Research boundaries

This goal repairs implementation and reporting. Training benefit, human readability,
causal/internal reasoning faithfulness, and a clean notation-only effect remain
unmeasured. Count coverage can be exponential in genuinely contingent candidates.
Large symmetric source structures use conservative coarse grouping, which can
merge distinct structures; signatures are not logical equivalence or proof
isomorphism. The pilot is deliberately bounded. No GPU job was submitted and no
commit was requested.
