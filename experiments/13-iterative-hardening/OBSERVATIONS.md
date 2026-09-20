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
rows, and 840 closed-loop/open-chain rows. Its aggregate is intentionally
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
rigid.

### Raw-trace verification

The paired trace audit found 493 base-correct/1.5K-wrong prompts and 495
1.5K-correct/6K-wrong prompts. Its first automated report overstated
`missing_active_axis_update` and `unaligned_trace`: the parser initially
recognized only V13-style `X-Extraction` lines and one-step-per-sentence
traces. Manual inspection of the exported raw examples corrected that
interpretation.

The 12 exported base-correct/1.5K-wrong representatives were all depth-5
examples. Every 1.5K trace processed all 13 premises, but 10/12 wrote at least
one cardinal premise to the irrelevant axis. Across 150 cardinal premises, 89
spurious cross-axis updates were visible. A representative failure interpreted
`Museum is North of Bank` as both `Bank < Museum` on X and Y. The model could
then construct a fluent but false global graph and select the wrong direction.

The 12 exported 1.5K-correct/6K-wrong representatives were also depth-5
examples. The 6K model did not generally stop after five premises: it used five
semantic phases and, in all 12 examples, listed all 13 premises in its parse
phase. However, 9/12 already reversed at least one cardinal relation in that
parse, and 10/12 later claimed a required query axis was unknown. Representative
traces show correctly read facts being dropped or contaminated while merging
the X/Y chains.

These counts describe the analyzer's deliberately depth-ranked 12-example
samples, not unbiased estimates over all 493/495 regressions. They nevertheless
directly confirm both failure mechanisms exist. The analyzer now supports both
the 1.5K sentence-step format and the 6K compact-phase format so the next full
audit will count them without treating formatting differences as reasoning
errors.

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

## Closed-loop conditions versus open-chain controls

A closed-loop condition is an inconsistent strict-order loop. Every question
over it must return `Cannot be determined`. An open-chain control has the same
family, relation/entity budget, topology intent, and placement intent, but its
final edge ends at a fresh entity instead of returning to the start. It remains
consistent and must be solved normally. Both conditions use the same number of
relations.

| model | closed loop | open chain |
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

Closed loops collapse all families to one answer and bypass the base model's
weak enumeration and counting skills:

| family | closed loop | open chain |
|---|---:|---:|
| direction | 90.0% | 82.1% |
| which | 87.1% | 32.9% |
| count | 95.0% | 11.4% |

Current conditions are structurally distribution-matched and independently
sampled, not literal one-edge counterfactuals from the same paragraph. Shared-
world bundles remain an optional later diagnostic. Ordinary consistent examples
teach general task behavior; open-chain controls are separately tracked hard
negatives near the loop-closure decision boundary.

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
- The base is weak on multi-answer, which-object, count, and open-chain-control
  rows.
- Before native V13 training, no evaluated model was simultaneously strong on
  closed-loop conditions and open-chain controls; Probe v2 later closes that
  gap in aggregate.

Likely causal interpretation after the corrected full audit:

- Incorrect one-axis updates and graph-merge omissions explain a large share
  of the V13 transfer gap. The corrected audit found spurious cross-axis
  updates in 389/493 base-correct/1.5K-wrong traces and incorrect or missing
  active-axis updates in 244/495 and 201/495 1.5K-correct/6K-wrong traces.
- The 6K adapter is more specialized to V12 trace and answer templates.
- Some base closed-loop success is conservative `Cannot be determined`
  behavior rather than exact cycle discrimination.

## Completed bridge tests before native V13 SFT

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

The corresponding repository entry points are
`slurm/eval-v12-stage1-bridge-h200.sh` and
`scripts/analyze_v13_error_traces.py`. The bridge writes
`results/V12-STAGE1-BRIDGE.md`; the trace audit writes
`results/ERROR-TRACE-AUDIT.md` and `ERROR-TRACE-EXAMPLES.jsonl`.

## Current training implication

Native V13 SFT should target the base model's actual gaps—multi-answer rules,
set enumeration, counting, cardinal/mixed parsing, and consistency
discrimination—while retaining its clean direction-chain reasoning. More data
from a narrow trace template is not automatically beneficial; the V12
1.5K-to-6K degradation is the warning this experiment must address.

The first intervention is therefore a 400-row, one-epoch native V13 safety
probe. Its recipe and explicit go/no-go gates are documented in `README.md`; it
is not the final 1.5K/6K scaling experiment.

Probe v1 produced a partial success: stage-1 V13 rose from 42.0% to 58.2%,
`dir-2` from 1.7% to 63.3%, consistent which from 6.1% to 47.0%, consistent
count from 1.2% to 67.3%, and open-chain controls from 7.4% to 61.9%. One-pass
depth 5 was retained (72.9% to 72.2%). However, closed-loop conditions fell
from 59.8% to 22.6%, showing that 30 closed-loop examples were insufficient to
establish a global check-before-local-solve policy. Probe v2 therefore changes
only the curriculum: 75 closed-loop conditions and 75 open-chain controls
across broader structures, while keeping the same fresh base, optimizer,
learning rate, epoch count, and total 400 rows.

## Probe v2 result — curriculum repair succeeded

Probe v2 passed all six V13 gates. It kept the training strength fixed and
changed only the curriculum from 30 closed loops/30 open chains to 75 closed
loops/75 open chains across all three question families and broader structures.

| capability, stage 1 | base | Probe v1 | Probe v2 | v2 vs v1 |
|---|---:|---:|---:|---:|
| V13 overall | 42.0% | 58.2% | **68.7%** | **+10.5 pp** |
| `dir-2` | 1.7% | **63.3%** | 55.0% | -8.3 pp |
| consistent which | 6.1% | 47.0% | **51.6%** | +4.6 pp |
| consistent count | 1.2% | **67.3%** | 60.8% | -6.5 pp |
| depth 5 | 72.9% | **72.2%** | 68.1% | -4.1 pp |
| closed-loop conditions | 59.8% | 22.6% | **74.5%** | **+51.9 pp** |
| open-chain controls | 7.4% | **61.9%** | 61.4% | -0.5 pp |

The decisive result is the closed-loop/open-chain comparison. Closed-loop
accuracy rose by 51.9 points while open-chain accuracy remained essentially
unchanged. This
is evidence of learned inconsistency discrimination rather than blanket
`Cannot be determined` refusal. Probe v2 also improves the untouched base by
14.7 points on closed loops and 54.0 points on open-chain controls.

Probe v2 remains broadly useful rather than trading everything for the cycle
policy: stage-1 overall rises 26.7 points over base; `dir-2`, which and count
remain far above base; and one-pass depth 5 stays within the planned five-point
retention budget (72.9% to 68.1%). The modest v1-to-v2 losses on `dir-2`, count
and depth are the cost of reallocating 90 rows toward paired consistency
training, but all remained above their v2 gates.

Probe v2 is also self-consistent across decoding protocols: 68.7% at stage 1
and 68.6% at stage 2. Unlike the untouched base, it does not rely on a second
readout pass to rescue its answer. Stage-2 depth remains lower than the base's
96.5%, but the primary policy-retention measure is one-pass depth because the
native SFT model must answer correctly itself.

V12 compatibility falls from 17.9% in Probe v1 to 10.3% in Probe v2. This is
reported as distribution compatibility, not a V13 gate: both probes start from
the untouched base and learn V13's different cardinal/mixed semantics and trace
contract. It would become an optimization objective only if backward V12
compatibility were explicitly required.

The curriculum lesson is now established: 30 closed-loop examples were not
enough, but 75 closed-loop examples paired with 75 structurally diverse
open-chain controls teach the global check without producing indiscriminate
refusal. Probe v2 is the first recipe approved for a native V13 1.5K scaling
run.

## Native V13 1.5K result — healthy SFT scaling

The first scale run expanded the approved Probe v2 curriculum to 1,500 rows:
940 ordinary consistent examples, 280 closed-loop conditions, and 280
open-chain controls. It broadened the ordinary portion across all 12 semantic
subtypes, all relation modes, depths 1–4, unequal-axis depths, and both
distractor topologies. Training remained a fresh-base, one-epoch QLoRA run at
`5e-5`; only data scale and coverage changed.

The 1.5K model passed all six scaling gates.

| capability, stage 1 | base | Probe v2 | V13 1.5K | 1.5K vs v2 |
|---|---:|---:|---:|---:|
| V13 overall | 42.0% | 68.7% | **85.4%** | **+16.7 pp** |
| `dir-2` | 1.7% | 55.0% | **60.0%** | +5.0 pp |
| consistent which | 6.1% | 51.6% | **69.5%** | +17.9 pp |
| consistent count | 1.2% | 60.8% | **81.2%** | +20.4 pp |
| depth 5 | 72.9% | 68.1% | **94.4%** | **+26.3 pp** |
| closed-loop condition | 59.8% | 74.5% | **91.9%** | +17.4 pp |
| open-chain control | 7.4% | 61.4% | **82.6%** | +21.2 pp |

This is the first evidence of healthy native-V13 SFT scaling. More data did not
repeat V12's pattern of improving only its familiar distribution while harming
structural transfer. Instead, broader 1.5K coverage improved every reported V13
capability relative to Probe v2, including the held-out depth-5 slice. The
depth result is especially important: the 1.5K model was never trained on depth
5, yet rose from 68.1% to 94.4%, indicating generalization from broader depths
1–4 rather than memorization of the held-out depth.

Closed-loop and open-chain performance also rose together. Closed loops improve
from 74.5% to 91.9%, while open chains improve from 61.4% to 82.6%. This is a
stronger discrimination result than merely increasing `Cannot be determined`
responses: the model both rejects inconsistent loops and continues solving
structurally matched valid chains.

The model is already strong in one pass (85.4%) and gains another two points in
stage 2 (87.4%). Stage-2 results remain consistent with the stage-1 picture:
`dir-2` 66.7%, which 71.1%, count 81.5%, depth 5 93.8%, closed loops 97.9%,
and open chains 83.6%. The small overall stage gap suggests the supervised
reasoning and final answer are substantially aligned.

V12 compatibility rises from 10.3% for Probe v2 to 29.6% at 1.5K. This remains
informational because native V13 intentionally uses different relation and
trace semantics, but the increase is reassuring: broader V13 coverage did not
make legacy compatibility monotonically worse.

### Scaling interpretation

The move from 400 to 1,500 rows improved breadth, loop discrimination, and
unseen-depth generalization simultaneously. This validates the Probe v2
curriculum as a real scaling recipe rather than a small-data artifact. A 6K run
is now scientifically justified, but it must remain a fresh-base model and use
a frozen nested data design so the next question is clean:

> Does additional diverse V13 data continue improving capability, or does the
> specialization pattern seen in V12 return at larger scale?

The 6K comparison must preserve the 1.5K rows as a subset, keep the diagnostic
and depth 5 held out, and report paired fixes versus regressions—not only the
aggregate score.

## Native V13 nested 6K result — continued scaling and diagnostic saturation

The nested 6K run cleanly isolates data scale. Its training set contains the
exact frozen 1.5K rows plus 4,500 new rows, every generation-cell quota is
scaled by four, and both adapters start independently from untouched
`Qwen/Qwen3.5-4B`. Training strength otherwise remains fixed. The stage-1
result passed all six gates and improves every reported capability over 1.5K.

| capability, stage 1 | base | V13 1.5K | V13 6K | 6K vs 1.5K |
|---|---:|---:|---:|---:|
| V13 overall | 42.0% | 85.4% | **94.9%** | **+9.5 pp** |
| `dir-2` | 1.7% | 60.0% | **78.3%** | +18.3 pp |
| consistent which | 6.1% | 69.5% | **92.0%** | **+22.5 pp** |
| consistent count | 1.2% | 81.2% | **90.4%** | +9.2 pp |
| depth 5 | 72.9% | 94.4% | **100.0%** | +5.6 pp |
| closed-loop condition | 59.8% | 91.9% | **99.8%** | +7.9 pp |
| open-chain control | 7.4% | 82.6% | **95.0%** | +12.4 pp |

The paired comparison is decisive rather than a small aggregate fluctuation.
Across the same 2,256 prompts, the 6K model fixes 244 examples that 1.5K
missed and regresses on 29 that 1.5K answered correctly: 8.4 fixes per
regression and a net gain of 215 examples. Slice-level paired changes are:

| slice | fixed by 6K | regressed at 6K | net fixes |
|---|---:|---:|---:|
| overall | 244 | 29 | +215 |
| `dir-2` | 16 | 5 | +11 |
| consistent which | 111 | 12 | +99 |
| consistent count | 32 | 8 | +24 |
| depth 5 | 8 | 0 | +8 |
| closed-loop condition | 34 | 1 | +33 |
| open-chain control | 58 | 6 | +52 |

The largest capability gain is complete which-object enumeration, which rises
22.5 points and accounts for 99 net paired fixes. `dir-2` also improves
substantially but remains the weakest reported aggregate slice at 78.3%, so
partial-axis answer-set policy is still a useful target when constructing a
harder successor suite. Counting reaches 90.4% but still has 25 errors in its
260-row consistent slice.

The consistency curriculum continues to behave correctly at scale. Closed
loops rise to 99.8% while structurally matched open chains rise to 95.0%. Since
both improve together, this is evidence of stronger discrimination between a
globally inconsistent world and a valid loop-shaped hard negative, not a
blanket tendency to answer `Cannot be determined`. Only one of 420 closed-loop
rows remains wrong; the consistency rule itself is therefore close to solved
on the present construction.

Depth-5 accuracy reaches 100% (144/144) even though depth 5 is absent from
training. This confirms successful extrapolation from the broader depth-1–4
curriculum on this construction. It does not establish unlimited depth
generalization: the current depth-5 slice is now saturated and can no longer
measure meaningful progress.

### Decision and next implication

The 6K adapter replaces 1.5K as the strongest native V13 SFT checkpoint. The
29 paired regressions should remain available for qualitative inspection, but
they do not outweigh 244 paired fixes or indicate broad specialization damage.

Stage 2 is not a priority for this decision. Stage 1 directly measures whether
the trained model's own reasoning trace and final answer are aligned, and the
6K stage-1 result is already conclusive. V12 compatibility likewise remains
informational rather than a native-V13 gate. Neither missing result blocks the
6K conclusion.

The main experimental implication is that the frozen V13 diagnostic is now
approaching saturation. Simply expanding the same 6K distribution to another
larger SFT set would offer diminishing information. Before scaling again, keep
this suite as a retention benchmark and add a new controlled challenge suite
that targets remaining policy failures and genuine structural extrapolation.
