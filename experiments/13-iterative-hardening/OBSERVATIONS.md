# Experiment 13 diagnostic observations

Status: preliminary transfer findings, before native V13 SFT.

These observations compare untuned `Qwen/Qwen3.5-4B` with the frozen V12
4B adapters trained on 1,500 and 6,000 examples. All models were evaluated on
the same 2,256-row V13 diagnostic suite. Stage-1 and stage-2 reports were run
on the remote GPU environment and copied to the original checkout as
`SUMMARY.md` and `SUMMARY-2.md`.

Those reports predate the schema-4 taxonomy cleanup. Old derived cycle labels
now mean the corresponding base subtype under
`world_consistency=inconsistent`; prompts, gold answers, and responses are
unchanged by that metadata cleanup.

## Headline

| model | stage 1 | stage 2 | stage-2 change |
|---|---:|---:|---:|
| untuned 4B | 42.0% | 71.1% | +29.1 pp |
| V12 4B 1.5K | 50.4% | 52.9% | +2.5 pp |
| V12 4B 6K | 44.0% | 43.7% | -0.3 pp |

The V12 adapters' low V13 accuracy is not primarily an answer-format problem.
Stage 2 strongly rescues the untuned model but barely changes either adapter.
The adapters' first-pass reasoning is therefore the main failure surface.

The V13 score is not directly comparable to the V12 2K headline. V12's test
is a matched holdout from the same generator and mixture as its training data.
V13 is a stress-weighted transfer suite containing new relation modes,
controlled proof structures, and global consistency tests.

## What V13 changed relative to V12

| dimension | V12 | V13 diagnostic |
|---|---|---|
| Relation language | Diagonal premises; each statement updates both axes | Diagonal, cardinal, and mixed premises |
| Proof shape | Random relatively dense worlds; shortest depth uncontrolled | Exact independent X/Y paths at depths 1–5 |
| Distractors | Incidental non-proof relations | Solver-classified disconnected and query-branch distractors |
| Contradictions | Direction-oriented reversal of an existing relation | Direct/indirect X, Y, or both-axis cycles; connected or disconnected; all question families |
| Consistency policy | Narrow V12 cycle cases | Any cycle anywhere invalidates the complete world |
| Evaluation distribution | Matched to V12 SFT | Intentionally weighted toward structural stress cells |
| Prompt | V12 three-rule instruction | V13 cardinal/diagonal and global-consistency contract |

The suite consists of 720 semantic controls, 696 controlled depth/distractor
rows, and 840 closed-cycle/open-control rows. Its aggregate is intentionally
dominated by diagnostic interventions rather than ordinary matched examples.

## The untuned base is already strong at clean direction chains

Stage-2 controlled-depth accuracy:

| proof depth | untuned 4B | V12 1.5K | V12 6K |
|---:|---:|---:|---:|
| 1 | 99.2% | 70.0% | 77.5% |
| 2 | 99.3% | 56.2% | 56.2% |
| 3 | 97.2% | 62.5% | 49.3% |
| 4 | 98.6% | 54.9% | 35.4% |
| 5 | 96.5% | 47.9% | 19.4% |

The depth-5 construction is controlled and clean, not arbitrary graph search.
It contains two monotonic five-edge chains, one per axis. Their internal nodes
are disjoint, the queried entities are endpoints, and no shorter proof is
allowed. There are no competing valid paths or direction changes within a
proof chain. Query-branch cells add three known distractor relations.

Depth 5 therefore measures preservation of clean transitive composition more
than the base model's reasoning limit. A later harder graph test could add
forks, merges, multiple paths, unequal depths, internal-node branches, and
orientation reversals.

## Stage 2 is a major base-model scaffold

At depth 5, the untuned model moves from 72.9% in stage 1 to 96.5% in stage 2.
Stage 2 supplies the original prompt, the complete first-pass reasoning, and a
constrained `Answer:` readout. The base often produces recoverable reasoning
without reliably committing to the correct answer set in one pass.

| model | depth 5 stage 1 | depth 5 stage 2 |
|---|---:|---:|
| untuned 4B | 72.9% | 96.5% |
| V12 1.5K | 50.0% | 47.9% |
| V12 6K | 19.4% | 19.4% |

Invalid-output rate is 0% for every model. The adapters are not merely failing
to format a useful trace; stage 2 cannot recover correct answers from them.

## V12 SFT shows negative structural transfer

Under stage 2, the untuned base beats both V12 adapters overall:

```text
untuned 4B   71.1%
V12 1.5K    52.9%
V12 6K      43.7%
```

The degradation grows with additional V12 SFT:

- consistent `dir-1`: 94.0% base, 58.6% 1.5K, 42.7% 6K;
- cardinal-only: 80.2% base, 52.1% 1.5K, 45.6% 6K;
- mixed: 71.2% base, 49.0% 1.5K, 41.9% 6K; and
- depth 5: 96.5% base, 47.9% 1.5K, 19.4% 6K.

The leading explanation is V12-specific policy specialization. V12 training
uses diagonal relations almost exclusively, so nearly every premise updates X
and Y together. V13 cardinal premises require updating exactly one axis, and
mixed worlds require keeping both graphs separate before composition. More V12
training may make the diagonal trace pattern and answer-policy templates more
rigid. This explanation fits the dose-dependent degradation but remains a
hypothesis until error traces directly confirm incorrect axis updates or other
V12-specific trace habits.

## What the base model is not good at

The base model's strength is concentrated in clean direction-chain reasoning.
On consistent stage-2 worlds:

| question family | untuned 4B |
|---|---:|
| direction | 86.7% |
| which-object | 39.5% |
| count | 24.6% |

### Multi-answer direction policy

The base reaches 94.0% on consistent `dir-1` but only 1.7% on `dir-2`. A
`dir-2` item has exactly one known axis and two valid compound directions. The
failure is applying the multi-letter option rule, not traversing a chain.

### Entity-set enumeration

Base accuracy falls as more correct entity options must be returned:

- `which-1`: 70.0%;
- consistent `which-2`: 32.0%;
- `which-3`: 30.0%; and
- `which-4`: 21.7%.

The model is much better at deciding one queried relation than exhaustively
enumerating all entities that satisfy a relation.

### Counting and omitted answers

The base scores 17.5% on consistent `count-1` and 48.3% on `count-omit`. It
struggles to convert the proven entity set into a count and apply the option
menu. V12 SFT helps some policies; V12 6K reaches 93.3% on `count-omit`.

### One-pass answer commitment

Overall base accuracy rises from 42.0% to 71.1% between stages 1 and 2 despite
zero invalid outputs. Its first pass often contains useful intermediate
reasoning without a reliable final letter set.

## Closed cycles versus open controls

A closed cycle is an inconsistent strict-order loop. Every question over it
must return `Cannot be determined`. A matched open control has the same family,
relation/entity budget, topology intent, and placement intent, but does not
close the chain and must be solved normally.

| model | closed cycle | open control |
|---|---:|---:|
| untuned 4B | 90.7% | 42.1% |
| V12 1.5K | 17.4% | 57.1% |
| V12 6K | 38.3% | 32.1% |

Interpret both columns together:

- high closed / low open suggests over-refusal or contradiction detection
  without reliable local solving;
- low closed / high open suggests local solving that ignores global
  inconsistency; and
- high closed / high open demonstrates genuine discrimination.

Closed cycles collapse all families to one answer and bypass the base model's
weak enumeration and counting skills:

| family | closed cycle | open control |
|---|---:|---:|
| direction | 90.0% | 82.1% |
| which | 87.1% | 32.9% |
| count | 95.0% | 11.4% |

Current pairs are budget-matched, not literal one-edge counterfactuals from the
same paragraph. Shared-world bundles remain an optional later diagnostic.

## What V12 SFT helps

The adapters teach task-specific answer policies that the base lacks:

- `count-omit`: 48.3% base, 71.7% 1.5K, 93.3% 6K;
- `dir-undetermined`: 78.3% base, 88.3% 1.5K, 93.3% 6K;
- `which-4`: 21.7% base, 78.3% 1.5K, 33.3% 6K; and
- consistent count: 24.6% base, 66.5% 1.5K, 56.9% 6K.

The SFT problem is not simply learning more reasoning. It is preserving the
base model's direction/chain capability while teaching multi-answer semantics,
enumeration, counting, and calibrated closed/open discrimination.

## Established findings versus hypotheses

Established:

- V13 is a structural transfer suite, not a matched continuation of V12.
- Stage 2 explains much of the base improvement and almost none of the adapter
  results.
- The base is extremely strong on the current clean depth construction under
  stage 2.
- V12 6K transfers worse than V12 1.5K overall and on deep direction proofs.
- The base is weak on multi-answer, which-object, count, and open-control rows.
- No model is simultaneously strong on closed cycles and open controls across
  all question families.

Likely but not yet directly proven:

- V12 SFT causes incorrect one-axis updates because its premises are almost all
  diagonal.
- The 6K adapter is more specialized to V12 trace and answer templates.
- Some base closed-cycle success is conservative `Cannot be determined`
  behavior rather than exact cycle discrimination.

## Required bridge tests before native V13 SFT

1. Evaluate the original V12 2K test with `STAGES=1` for both adapters. This
   separates evaluation-protocol effects from structural distribution shift.
2. Inspect paired cardinal/mixed error traces and count incorrect updates to
   the irrelevant axis.
3. Report semantic controls separately from depth and cycle/control
   interventions. Do not present the 2,256-row aggregate as an apples-to-apples
   replacement for the V12 headline.
4. Keep the diagnostic evaluation-only. Native V13 SFT must use different
   seeds and world fingerprints.
5. Treat preservation of base depth performance as a first-class SFT metric.

## Current training implication

Native V13 SFT should target the base model's actual gaps—multi-answer rules,
set enumeration, counting, cardinal/mixed parsing, and consistency
discrimination—while retaining its clean direction-chain reasoning. More data
from a narrow trace template is not automatically beneficial; the V12
1.5K-to-6K degradation is the warning this experiment must address.
