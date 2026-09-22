# Experiments 10–14: repository evidence audit

This note uses only evidence committed in this repository at `c79c996`. “Recorded
result” means a numeric result is stated in a committed experiment document; a
YAML, launcher, or empty run-log table establishes intent, not execution. This is
important because no `results/` artifacts for Experiments 10–14 are committed in
this checkout.

## Experiment 10 — Option E / TQA-Corr-Full

**Recorded facts.** The question was whether SFT could learn `E` (“None of these
is proven”) when no A–D option is provable, avoid it when A–D is provable, and
improve with more traces (`experiments/10-option-e-full/README.md:7-9`). The
1,500-row evaluation set retained the cleaned SpatialMap rows but added E to
every item: 369 E-gold rows and 1,131 A–D-gold rows
(`experiments/10-option-e-full/README.md:11-26`). Training used synthetic
five-option traces, nested 1.5K/5K/20K datasets, and a 0.8B/2B/4B × three-size
QLoRA matrix (rank 64, two epochs) (`experiments/10-option-e-full/README.md:42-69`).
The intended evaluation was the Full task for all nine cells, summarized into
`results/full/SUMMARY.md` (`experiments/10-option-e-full/README.md:90-99`,
`experiments/10-option-e-full/README.md:120-133`).

**What actually ran / numeric findings.** No result is recorded. The run-log
table is empty (`experiments/10-option-e-full/README.md:135-139`), and the
repository contains configs and launchers but no committed Exp 10 results. The
3×3 expansion itself was added as configuration in commit `a5f5625`; that commit
does not establish that the cells completed.

**Unresolved.** All central hypotheses—E recall, false-E rate, data scaling, and
model-size scaling—remain unreported in primary repository evidence.

## Experiment 11 — v6 synthetic SFT

**Recorded facts.** Exp 11 replaced SpatialMap-shaped four-answer gold with
solver-labeled, five-option `generate_all_v6` traces and asked how performance
scaled across 2B/4B and nested 1.5K/6K/20K training sizes. It specified a
separate 4K synthetic test (seed 43) plus a 1,500-row SpatialMap set regraded by
the same solver (`experiments/11-v6-synthetic/README.md:1-14`). All six cells
used rank-64 QLoRA for two epochs; each was to be evaluated on both datasets
(`experiments/11-v6-synthetic/README.md:25-42`).

**What actually ran / numeric findings.** Only the 4B-1.5K cell has a committed
result analysis (added in commit `03e661e`). It scored 93.875% strict / 94.10%
loose on 4,000 synthetic examples and 96.13% strict / 96.40% loose on SpatialMap
v6 (`experiments/11-v6-synthetic/RESULTS.md:19-31`). Of 245 synthetic misses,
188 (77%) were concentrated in `dir-cycle`, `dir-incomplete`, and `dir-2`; their
strict accuracies were 36.8%, 55.6%, and 86.5%, versus at least 96% for the other
reported buckets (`experiments/11-v6-synthetic/RESULTS.md:35-58`,
`experiments/11-v6-synthetic/RESULTS.md:62-82`). Stage 2 changed only nine items
and overwrote eight already-correct stage-1 answers, while parsing had zero
failures (`experiments/11-v6-synthetic/RESULTS.md:26-31`).

**Recorded interpretation and unresolved work.** The result document attributes
the high mean to easy-bucket mass and identifies missing/ambiguous prompt rules
for cycles, incomplete pairs, and two-letter partial-axis answers
(`experiments/11-v6-synthetic/RESULTS.md:193-211`). It explicitly says the 2B
rows were still to finish and rejects GRPO on the 94% aggregate because residual
errors were concentrated and likely lacked useful reward variance
(`experiments/11-v6-synthetic/RESULTS.md:215-224`). The proposed next step was
patched prompting plus reweighted SFT, with GRPO only afterward if cleanly
formatted answers remained wrong (`experiments/11-v6-synthetic/RESULTS.md:230-244`).
No committed evidence reports the other five planned cells.

## Experiment 12 — hierarchical-uniform v6 reweighting

**Recorded facts.** Exp 12 implemented the Exp 11 diagnosis: it made the three
Type-0 letter rules explicit, balanced the hierarchy (one third each direction,
which, count; equal subtypes within each family), shuffled examples, and selected
checkpoints on a hard validation slice (`experiments/12-v6-mix-reweight/README.md:1-27`).
It used one seed-52 21K draw split into a disjoint 2K test, 1K validation set, and
nested 1.5K/6K/18K train sets (`experiments/12-v6-mix-reweight/README.md:10-14`,
`experiments/12-v6-mix-reweight/README.md:29-40`). Planned evaluations were the
matched 2K headline, 1K validation/checkpoint selection, and unmatched SpatialMap
v6 transfer; zero/one/three-shot untuned baselines were also configured
(`experiments/12-v6-mix-reweight/README.md:42-65`,
`experiments/12-v6-mix-reweight/README.md:90-96`).

**What actually ran / numeric findings.** There is no committed Exp 12 results
file. The strongest primary trace is a later repository statement that the 1.5K
model was “around 97%” and the 6K model “around 96%” on the matched 2K test
(`experiments/13-iterative-hardening/README.md:605-611`), plus the Exp 13 transfer
diagnostic, which explicitly says both frozen V12 adapters were evaluated remotely
(`experiments/13-iterative-hardening/OBSERVATIONS.md:3-14`). On the harder 2,256-row
V13 suite, those adapters scored 50.4%/52.9% (1.5K, stage 1/2) and 44.0%/43.7%
(6K), versus 42.0%/71.1% for untuned 4B (`experiments/13-iterative-hardening/OBSERVATIONS.md:16-31`).
These V13 values are transfer diagnostics, not Exp 12’s matched headline. There
is no primary evidence here that the 18K, few-shot, or all declared evaluation
cells ran.

**Recorded interpretation and unresolved work.** More V12 data improved the
matched task but transferred progressively worse to cardinal/mixed V13 structure;
the documented leading explanation is specialization to diagonal trace and answer
templates (`experiments/13-iterative-hardening/OBSERVATIONS.md:88-110`). Thus Exp
12 largely closed the immediate v6 policy holes, but produced a saturated matched
benchmark and exposed poor structural transfer rather than a clean GRPO target.

## Experiment 13 — iterative structural hardening and trace ablation

**Purpose and progression.** Exp 13 deliberately did **not** start with GRPO. It
constructed a harder, solver-controlled SFT task with cardinal/mixed relations,
controlled proof depth, structured distractors, global consistency, disjoint
splits, stratified reporting, and frozen V12/SpatialMap checks
(`experiments/13-iterative-hardening/README.md:605-660`,
`experiments/13-iterative-hardening/README.md:662-678`). The sequence actually
documented is: V12 transfer diagnostic → 400-row probe v1 → curriculum-repaired
probe v2 → native 1.5K → nested 6K → evaluation-only V13.1 breakpoint → targeted
full-state 8K → matched delta-state 8K trace ablation. Commits `b572a3c`,
`f1128b3`, `f73185e`, `0b85c2b`, and `e28c4f2` record these successive results.

**Evaluations and numeric findings.** All figures below are stage-1 strict
accuracy unless stated otherwise.

- Probe v1 raised V13 overall 42.0%→58.2% but collapsed closed-loop accuracy
  59.8%→22.6%; open-chain controls rose 7.4%→61.9%
  (`experiments/13-iterative-hardening/OBSERVATIONS.md:292-300`).
- Probe v2 changed only curriculum balance and reached 68.7% overall, 74.5%
  closed-loop, and 61.4% open-chain; it passed all gates
  (`experiments/13-iterative-hardening/OBSERVATIONS.md:302-330`).
- Native 1.5K reached 85.4% overall, 94.4% on held-out depth 5, 91.9% closed
  loops, and 82.6% matched open chains; stage 2 was 87.4% overall
  (`experiments/13-iterative-hardening/OBSERVATIONS.md:350-389`).
- Nested 6K reached 94.9% overall, 100% depth 5, 99.8% closed loops, and 95.0%
  open chains. Against 1.5K it made 244 fixes and 29 regressions on the same
  2,256 prompts (`experiments/13-iterative-hardening/OBSERVATIONS.md:411-442`).
- The frozen 1,224-row V13.1 breakpoint exposed an interference/scale failure:
  6K scored 82.4% overall and 62.7% at scale 10; the targeted full-state 8K run
  improved V13.1 only to 85.6% and scale 10 to 63.2%
  (`experiments/13-iterative-hardening/OBSERVATIONS.md:483-507`).
- In the matched trace-only ablation, delta-state 8K beat full-state 8K by
  12.2 points on V13.1 (97.8% vs 85.6%) while also improving original V13
  (96.2% vs 95.5%). It produced 158 fixes and nine regressions; depth/length 10
  rose 63.2%→97.5% (`experiments/13-iterative-hardening/OBSERVATIONS.md:509-547`).

**Recorded interpretation and unresolved work.** The matched ablation identifies
repeated full-state serialization—not absent intermediate-depth examples—as the
main artificial bottleneck, so delta-state became the default representation
(`experiments/13-iterative-hardening/OBSERVATIONS.md:556-562`). Remaining caveats
are that this was one run per trace format, V13.1 was nearly saturated, and
consistent which-object accuracy regressed from 94.1% to 92.5%
(`experiments/13-iterative-hardening/OBSERVATIONS.md:549-570`). New static
difficulty, multi-seed confirmation, and explicit which retention remained open.

## Experiment 14 — V13 GRPO feasibility

**Recorded facts / planned test.** Exp 14 is the first RL step after delta-state
SFT. It asks whether exact-outcome GRPO can improve a frozen hard holdout without
materially damaging original V13 (`experiments/14-v13-grpo-feasibility/README.md:1-13`).
The design merges the delta-state 8K adapter into BF16, then trains a separate
rank-64 LoRA with a trainer GPU and a vLLM rollout GPU
(`experiments/14-v13-grpo-feasibility/README.md:15-26`). Its data are 680
prompt-only training rows and a disjoint 300-row frozen holdout at scales
10/12/14; reward is exact solver answer-set match (1.0) plus a small format
reward (`experiments/14-v13-grpo-feasibility/README.md:28-39`).

Before optimization, 48 prompts × four rollouts must pass explicit parseability,
pass-rate, and within-group-variance gates. The planned GRPO probe is four
generations per prompt, 40 optimizer steps, LR `5e-6`, KL 0.04, and clipping 0.2
(`experiments/14-v13-grpo-feasibility/README.md:41-70`). Success requires reward
variance, hard-holdout improvement, and no more than a two-point original-V13
drop (`experiments/14-v13-grpo-feasibility/README.md:76-86`).

**What actually ran / numeric findings.** None are committed: there is no
`CALIBRATION.json`, `results/SUMMARY.md`, or other Exp 14 result artifact in this
checkout. Commits `26ee9a4`, `933b012`, `80261d9`, and `194f58f` establish the
probe, retention launcher, and calibration changes, not a successful run. The
rough-thoughts file explicitly labels the work provisional and says not to infer
the proposed GRPO narrative before calibration and frozen-holdout comparison
(`experiments/14-v13-grpo-feasibility/ROUGH-RESEARCH-THOUGHTS.md:1-5`,
`experiments/14-v13-grpo-feasibility/ROUGH-RESEARCH-THOUGHTS.md:315-330`).

## Progression toward GRPO (synthesis, not an additional result)

The evidence supports a staged research progression: Exp 10 repaired the label
space; Exp 11 showed that a high average concealed three policy-specific errors;
Exp 12 made those rules explicit and rebalanced SFT, but its stronger matched
scores transferred poorly to new structure; Exp 13 created that structure, found
and repaired curriculum and representation bottlenecks, and produced the 96.2% /
97.8% delta-state SFT policy; Exp 14 then formulated a gated, prompt-only,
solver-rewarded GRPO test. The final step is still a protocol, not a result. The
repository therefore supports “progression toward a scientifically testable GRPO
experiment,” but not yet “GRPO improves V13.”

## Appendix: exact question taxonomy and trace semantics

This appendix separates **answer semantics** from **orthogonal structure**. V13
defines exactly 12 consistent-world semantic subtypes; cycle is deliberately not a
thirteenth subtype (`spatial/spatial_generation_v13.py:36-48`,
`spatial/spatial_solver_v13.py:770-777`). The solver, not hidden generation
coordinates, is authoritative: it reparses the rendered prompt, computes transitive
closure, and the generator rejects a row if the recovered subtype or requested
structure differs (`spatial/spatial_solver_v13.py:204-235`,
`spatial/spatial_generation_v13.py:1110-1215`).

### Shared calculus and special-option rule

On X, `A < B` means A is west of B; on Y it means A is south of B. A cardinal
premise adds one axis edge and a diagonal premise adds two; proofs are directed
paths in the transitive closure (`spatial/spatial_solver_v13.py:37-46`,
`spatial/spatial_solver_v13.py:119-131`). Thus a compact faithful trace prefix is:

```text
For each premise: emit X edge / Y edge (or “none” for the untouched axis).
Close each axis transitively. Check the complete X and Y graphs for a cycle.
If consistent, show the shortest required proof path(s), then mark every option In/Out.
Answer: <exact accepted letter set>
```

`Cannot be determined` and `None of the Options` are not interchangeable. In a
consistent world, the former is Type-0 insufficient information or an incomplete
two-direction menu; the latter means a derived answer is absent. For consistent
which/count questions, an empty listed result uses None, while Cannot is reserved
for inconsistency (`spatial/spatial_solver_v13.py:480-503`,
`spatial/spatial_solver_v13.py:600-620`). The accepted textual paraphrases are
enumerated—not guessed—at `spatial/spatial_solver_v13.py:21-35`.

### All consistent-world semantic subtypes

The examples below use option labels only illustratively; option order is shuffled,
so semantics attach to option text, not to a fixed letter. Each template is the
minimal final-deduction portion of the generated trace; the actual renderer also
emits per-premise extraction and state updates (`spatial/spatial_generation_v13.py:940-1025`).

| subtype | minimal example | faithful compact CoT/CoS rule |
|---|---|---|
| `dir-1` | `B < A` on X and Y; ask A relative to B; NE is listed | `X-proof: B < … < A -> East; Y-proof: B < … < A -> North; compose NE; NE In, all others Out.` Both axes yield one compound. |
| `dir-2` | only X proves `B < A`; NE and SE are both listed | `X-proof -> East; Y-proof: none; possible compounds = {NE,SE}; both listed -> both In.` |
| `dir-undetermined` | no X or Y path between A and B | `X-proof: none; Y-proof: none; neither axis yields a compound; Cannot be determined In.` |
| `dir-incomplete` | only X proves East; NE is listed but SE is absent | `possible = {NE,SE}; listed intersection is nonempty but incomplete; do not invent Y; Cannot be determined In, NE Out.` |
| `dir-omit` | X and Y prove NE, but NE is absent | `derive unique NE; no listed content equals NE; None of the Options In.` |
| `which-1` … `which-4` | ask “Which object is north of R?” and exactly N of the four named options have an R-to-object Y path | `For each listed object O: required Y-proof R < … < O? yes -> In, otherwise Out; return all and only the N In letters.` The same rule covers N=1,2,3,4. |
| `which-0` | ask northeast of R; none of four listed objects has both required paths | `For each O require X: R < … < O AND Y: R < … < O; none pass; None of the Options In.` An unlisted proven object does not change that menu result. |
| `count-1` | exactly three objects have every required path and option “3” is listed | `proven set = {…}; count = 3; numeric 3 In; other numbers and special Out.` Zero is an ordinary integer answer if listed. |
| `count-omit` | derived count is 3, but 3 is absent | `proven set = {…}; count = 3; no numeric option matches; None of the Options In.` |

These are distinct option-set rules, not merely answer cardinalities. Type 0 first
constructs zero, one, or two compatible compounds and then applies the complete-menu
rule (`spatial/spatial_solver_v13.py:480-520`,
`spatial/spatial_solver_v13.py:640-646`). Which requires every axis named by the
queried direction for each candidate; count applies the identical predicate to all
objects and takes its cardinality (`spatial/spatial_solver_v13.py:523-599`). The
generator explicitly maps the 12 subtype names to question family and requested
answer count (`spatial/spatial_generation_v13.py:183-202`), constructs `which-0`
with four non-proven choices (`spatial/spatial_generation_v13.py:761-794`), and
constructs count omission by excluding the true integer (`spatial/spatial_generation_v13.py:797-820`).

### Orthogonal V13 structural conditions

None of the following changes the consistent-world rule above. They change what must
be parsed, selected, or composed before applying it.

| dimension | exact conditions | minimal example and trace obligation |
|---|---|---|
| relation mode | `diagonal`, `cardinal`, `mixed` | Diagonal: “A NE of B” emits `B<A` on both axes. Cardinal: “A east of B” emits X only and Y `none`. Mixed uses both premise kinds. The enum is exact at `spatial/spatial_generation_v13.py:30-34`; extraction behavior is at `spatial/spatial_generation_v13.py:963-978`. |
| proof depth | shortest X depth and shortest Y depth, with direct iff depth 1 | `B < C < A` has depth 2. Trace the shortest path, not hidden coordinates or every possible path. Depth/path/support metadata comes from BFS (`spatial/spatial_solver_v13.py:655-715`) and is recorded per axis (`spatial/spatial_solver_v13.py:907-929`). |
| axis support | shared or independent X/Y supporting statements | Independent example: X proof `B<X<A`, Y proof `B<Y<A`, with disjoint statement IDs. Trace both paths separately. `axes_independent` is true exactly when both exist and their support intersection is empty (`spatial/spatial_solver_v13.py:911-929`). |
| unequal axes | `(x_depth,y_depth)` may differ | Example X depth 2, Y depth 6: trace the complete two-edge X proof and six-edge Y proof, then compose once both finish. V13.1 freezes `(2,6),(6,2),(4,8),(8,4),(6,10),(10,6)` (`experiments/13-iterative-hardening/scripts/make_breakpoint_data.py:35-40`). Unequal depth is structural, still `dir-1`. |
| distractor policy | `none`, `disconnected`, `query-branch` | A disconnected relation lies outside the query proof component; a query-branch relation touches that component but belongs to neither shortest proof. Parse it into state, but exclude it from the final support path. Classification is by non-proof statement IDs and connectivity (`spatial/spatial_solver_v13.py:718-760`); construction is explicit at `spatial/spatial_generation_v13.py:527-551`. |
| trace representation | `full-state`, `delta-state` | Full-state reprints both complete axis states after every premise. Delta-state prints only affected components plus X/Y conflict status, then one complete final state. Final proof and option verdicts are identical (`spatial/spatial_generation_v13.py:128-131`, `spatial/spatial_generation_v13.py:979-1025`). This is a representation ablation, not a question subtype. |

Depth and distractor controls currently apply only to `dir-1`, require independent
axes, and disallow diagonal-only mode; distractors additionally require exact depths
(`spatial/spatial_generation_v13.py:267-300`). The original V13 diagnostic crosses
depths 1–5 with cardinal/mixed modes and none/disconnected/query-branch policies
where feasible (`spatial/generate_diagnostic_v13.py:38-43`,
`spatial/generate_diagnostic_v13.py:161-173`). V13.1 extends equal depths to 6/8/10
and adds the unequal-axis cells above; distractor count there equals the larger axis
depth (`experiments/13-iterative-hardening/scripts/make_breakpoint_data.py:89-118`).

### Closed loops and open-chain controls

V13 first checks **global** consistency. A directed strict-order cycle on X, Y, or
both makes the whole world inconsistent, whether it is query-connected or in a
disconnected component. The only valid answer for direction, which, and count is the
listed `Cannot be determined` option (`spatial/spatial_solver_v13.py:351-405`,
`spatial/spatial_solver_v13.py:407-449`). The compact trace is:

```text
Parse every premise -> close X/Y globally.
Cycle witness: A < B < ... < A on X and/or Y.
Conclusion: complete map inconsistent; skip local query semantics.
Cannot be determined In; every content/None option Out.
```

Cycle structure is independently parameterized by axes `x|y|both`, topology
`direct|indirect`, placement `query-connected|disconnected`, and length; a direct
cycle must have length 2 and an indirect one length at least 3
(`spatial/spatial_generation_v13.py:107-145`). Diagonal-only cycles can only target
both axes because every diagonal edge updates both
(`spatial/generate_diagnostic_v13.py:75-80`). Generated inconsistent rows retain
their solver-verified **base** subtype as metadata, but prompt-only analysis returns
`semantic_subtype=null`, since local semantics are overridden
(`spatial/spatial_generation_v13.py:1110-1151`).

The matched open-chain control replaces the closing destination with a fresh entity,
keeping relation and overall entity budgets matched. It has no cycle and therefore
uses its base semantic rule normally (`spatial/generate_diagnostic_v13.py:97-132`).
Its compact trace is the ordinary consistent template: `global cycle check: none;`
then show the local proofs and apply the subtype's option rule. Closed and open rows
are structurally matched but independently sampled, not literal one-edge rewrites of
one paragraph (`experiments/13-iterative-hardening/OBSERVATIONS.md:184-190`,
`experiments/13-iterative-hardening/OBSERVATIONS.md:216-220`).

### Legacy Exp 10, v6, and v12 differences

- **Exp 10 / Corr-Full:** its fixed E text is `None of these is proven`; E merges
  75 direction empty-oracle rows, 96 omitted-count rows, and 198 legacy `which-4`
  fallback rows. Remaining A–D gold keeps E as a distractor
  (`experiments/tasks/utils.py:81-103`). Thus Exp 10 E is a broad “no A–D proof”
  label and must not be retroactively read as V13's separate Cannot-versus-None
  policy. The original which fallback preserved every in-passage option when none
  was proven (`eval/clean_v5.py:327-349`); Corr-Full remapped that four-letter
  fallback to E.
- **v6 / Exp 11 and v12:** the 13 reported buckets were `dir-1`, `dir-2`,
  `dir-undetermined`, `dir-cycle`, `dir-incomplete`, `dir-omit`, `which-1..4`,
  `which-0`, `count`, and `count-omit`
  (`experiments/11-v6-synthetic/README.md:16-21`). Here `dir-cycle` is a semantic
  bucket: a conflict makes the affected queried axis unknown; it does **not** run a
  global pre-check over every family (`spatial/spatial_solver.py:1-10`,
  `spatial/spatial_solver.py:227-243`, `spatial/spatial_solver.py:278-353`). V12
  uses the same v6 generator/solver, but balances one third direction, one third
  which, one third count and equalizes subtypes within each family
  (`experiments/12-v6-mix-reweight/README.md:10-24`).
- **V13:** cardinal premises are first-class, `count` is renamed `count-1`, and
  cycle becomes orthogonal global consistency rather than a direction subtype. Its
  canonical base list therefore contains 12, not 13, semantic subtypes
  (`spatial/spatial_generation_v13.py:24-48`).
