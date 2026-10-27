# Professor brief research: a five-slide SpatialEntail story

Archive status: retired presentation narrative, superseded by the
[composition pivot](../../../experiments/spatialentail-composition-v1/pivot_20261026.md).
The recommendations below are historical. Report line-number citations refer
to the pre-composition chapters preserved in the
[report archive](../../../experiments/archive/precomposition-report-2372c13.tar.gz),
not the current rewritten draft.

## Recommended framing

Use exactly five content slides, one for each question below. The presenter does
not need to derive formulae or explain SAT/UNSAT notation. The spoken thread is:
**some spatial questions do not have one answer from the text; build only
questions whose answers can be checked; vary what must be reasoned about; train
on checked explanations; then test whether the learning survives controlled
changes.**

Do not call G1--G4 “difficulty levels.” `G` is the number of correct options in
the complete answer set. Reasoning-chain length (`L`) and removable distractors
(`X`) are separate controls. The active design states this separation directly
([randomized-v2 design, lines 10--18](../audits/randomized-proof-v2-design.md#L10-L18)).

## Slide 1 — The problem being created

**Title:** *Can the answer really be inferred from the text?*

**One-sentence claim:** Existing text-only spatial questions can preserve an
answer from a hidden map even when more than one answer remains possible from
the words shown to the model.

**Show:** one short counterexample: the released Recycle Center question allows
both Northeast and Northwest while the published answer selects Northeast
([problem formulation, lines 105--125](../../../fyp-report/chapters/03-problem-formulation.typ#L105-L125)).

**Say:** The project therefore creates **SpatialEntail**, a text-only framework
whose questions contain named objects, spatial statements and one of three
queries: find a Direction, select possible Entities, or determine possible
Counts ([problem formulation, lines 7--27](../../../fyp-report/chapters/03-problem-formulation.typ#L7-L27)).
An answer is accepted only according to all spatial worlds consistent with the
visible text, not according to one hidden drawing. This is motivated by the
current audit: of 1,500 released SpatialMap-TQA questions, 667 published answers
are uniquely implied and 833 are possible but non-unique under the declared
semantics ([results, lines 4--24](../../../fyp-report/chapters/06-results-discussion.typ#L4-L24)).

**Avoid:** coordinate algebra, solver notation, or saying the original hidden
maps are wrong. The repo explicitly limits the claim to whether the released
text uniquely identifies the answer ([problem formulation, lines 144--151](../../../fyp-report/chapters/03-problem-formulation.typ#L144-L151)).

## Slide 2 — How a checked question is built and randomized

**Title:** *Build the reasoning first; trust the checker last*

**Show as a simple left-to-right pipeline:**

`sample a reasoning pattern` → `write spatial premises and a query` →
`solve every possible answer` → `check a proof/counterexample` →
`render text` → `parse and solve it again`

This is faithful to the implemented pipeline and rejection policy: accepted
items are solved, matched to the requested cell, certificate-checked, rendered,
parsed back and re-solved; inconsistency or drift causes rejection
([method, lines 15--29](../../../fyp-report/chapters/04-spatialentail-method.typ#L15-L29),
[method, lines 81--100](../../../fyp-report/chapters/04-spatialentail-method.typ#L81-L100)).

**Building blocks, in plain language:**

- apply a stated rule (modus ponens);
- unpack an “if and only if” rule;
- eliminate one side of an either/or statement;
- remove a double negative; and
- chain spatial relations transitively.

These are the five rule tokens in the active implementation
([`random_proof.py`, lines 14--30](../../../spatial/v2/legacy/random_proof.py#L14-L30)). Case
splits are measured when they occur, but are not sampled as if they were one
more linear token because they create branches
([randomized-v2 design, lines 20--33](../audits/randomized-proof-v2-design.md#L20-L33)).

**What is randomized:** complete feasible rule paths are sampled uniformly via
completion counts; object pairs and directions are shuffled; premise order is
shuffled independently from proof order; answer options are also seeded and
shuffled. The checked certificate, rather than the intended template, decides
the realized chain length
([`random_proof.py`, lines 127--175](../../../spatial/v2/legacy/random_proof.py#L127-L175),
[randomized-v2 design, lines 20--25](../audits/randomized-proof-v2-design.md#L20-L25),
[method, lines 45--48](../../../fyp-report/chapters/04-spatialentail-method.typ#L45-L48)).

## Slide 3 — What G1--G4 scaling means

**Title:** *Scale the number of valid answers, separately from reasoning length*

`G` is the size of the gold answer set:

| Scale | Professor-facing meaning | Representative example |
|---|---|---|
| G1 | exactly one option is correct (`SINGLE`) | “A is Northeast of B”; ask A relative to B → Northeast. This exact two-object renderer fixture is in the appendix ([dataset appendix, lines 16--29](../../../fyp-report/appendices/b-spatialentail-dataset.typ#L16-L29)). |
| G2 | exactly two options remain possible (`ALL-POSSIBLE`) | Northeast **and** Northwest in the released Recycle Center ambiguity. This is a semantic analogue, not a randomized-v2 row ([problem formulation, lines 105--120](../../../fyp-report/chapters/03-problem-formulation.typ#L105-L120)). |
| G3 | exactly three options form the complete answer | Schematic Count example: three distinct totals remain possible, so all three option letters must be returned. |
| G4 | exactly four options form the complete answer | Schematic Selection example: four named candidates remain possible, so omitting any one is wrong. |

The implementation fixes a five-option menu, assigns G1 to `SINGLE`, and G2--G4
to `ALL-POSSIBLE` ([study manifest, lines 419--452](../../../spatial/v2/study/randomized_v2.py#L419-L452)).
The benchmark balances 900 rows at each of G1, G2, G3 and G4, across Direction,
Selection and Count ([benchmark audit, lines 11--31](../../../experiments/spatialentail-randomized-v2/benchmark-candidate-audit.json#L11-L31)).

**Important presenter line:** “More correct options changes answer completeness;
it does not automatically mean a longer proof.” The study independently varies
`L` (semantic dependency length), `X` (verified removable distractors), and
whether an adjacent rule transition was seen in training. Count has genuine
feasibility constraints--for example, complete G2/G3/G4 Count certificates have
minimum audited lengths L2/L3/L4--so impossible combinations are not fabricated
([randomized-v2 design, lines 47--58](../audits/randomized-proof-v2-design.md#L47-L58)).

## Slide 4 — Training approach

**Title:** *Does a checked worked solution teach more reliably than an answer?*

**Show two target cards for the same question:**

- **Answer-only:** the correct option letter(s).
- **Verified Natural:** a solver-rendered, readable worked solution ending in
  the same answer.

The checker’s replayable certificates remain offline audit artifacts; they are
not shown as ordinary training targets
([randomized-v2 design, lines 284--316](../audits/randomized-proof-v2-design.md#L284-L316)).

**Design:** start every adapter from the same pinned Qwen3.5-4B model. Train
`SINGLE` with Natural and answer-only supervision at seeds 42 and 43. Train
`MULTI` with Natural supervision at the same two seeds, giving six adapters.
Each contract has 8,000 training and 800 development bases; MULTI is balanced at
2,000 examples for each G1--G4 and shares its G1 core with SINGLE
([randomized-v2 design, lines 284--304](../audits/randomized-proof-v2-design.md#L284-L304)).
The prepared jobs use QLoRA on one H100-47 GPU
([experiment README, lines 73--83](../../../experiments/spatialentail-randomized-v2/README.md#L73-L83)).

The training audit reports 15,400 unique base structures, no benchmark overlap,
and no train/development overlap. Every complete training sequence fits the
common 4,096-token limit, with no truncation
([training audit, lines 10--23](../../../experiments/spatialentail-randomized-v2/training-candidate-audit.json#L10-L23),
[training audit, lines 25--67](../../../experiments/spatialentail-randomized-v2/training-candidate-audit.json#L25-L67)).

## Slide 5 — Experiments underway

**Title:** *Frozen data; training and evaluation are the remaining test*

Separate completed preparation from unverified live execution:

- **Completed in tracked artifacts:** a 3,600-row candidate passed its audit,
  with 3,600 unique structures, equal G1--G4 representation, and four test
  partitions: matched, longer reasoning, more distractors, and one held-out rule
  transition ([benchmark audit, lines 2--32](../../../experiments/spatialentail-randomized-v2/benchmark-candidate-audit.json#L2-L32)).
- **Marked running by the operational checklist:** six central SFT adapters--two
  `SINGLE` Natural, two `SINGLE` answer-only, and two `MULTI` Natural
  ([checklist, lines 55--76](../../../experiments/spatialentail-randomized-v2/CHECKLIST.md#L55-L76)).
- **Next required evaluation:** compare the untuned model and six adapters by
  contract, partition, query family, G, L and X; then run external transfer on
  SpatialMap-TQA-Corr ([checklist, lines 78--118](../../../experiments/spatialentail-randomized-v2/CHECKLIST.md#L78-L118)).
- **Prepared optional comparisons:** general instruction models, reasoning
  models, and five spatial-specialist models, with text-only preflights for the
  latter ([experiment README, lines 96--177](../../../experiments/spatialentail-randomized-v2/README.md#L96-L177)).

**Closing sentence:** “The dataset construction is checked and frozen enough to
run; the open question is whether verified explanations produce stable gains on
new answer-set sizes, longer chains, extra noise and unseen rule combinations.”

## Uncertainties and claims to avoid

1. **Live job state is not verifiable from this checkout.** The checklist labels
   all six jobs `[~]`, but every Slurm ID is still “fill in,” and the tracked
   experiment directory contains no adapter `COMPLETED` markers. Say “marked
   running in the repo checklist,” not “all six are currently running on the
   cluster.”
2. **The randomized-v2 model results do not yet appear in tracked result
   artifacts.** The checklist still leaves adapter evaluation, G/L/X analysis,
   external transfer and report closure incomplete
   ([checklist, lines 78--134](../../../experiments/spatialentail-randomized-v2/CHECKLIST.md#L78-L134)).
3. **Do not mix the completed older SINGLE-v1 results with randomized-v2.** The
   report contains completed two-seed SINGLE-v1 scores, while randomized-v2 is a
   new G/L/X-controlled study. They answer related but different questions
   ([results, lines 80--114](../../../fyp-report/chapters/06-results-discussion.typ#L80-L114),
   [randomized-v2 status, lines 3--8](../audits/randomized-proof-v2-design.md#L3-L8)).
4. **G3/G4 examples above are explanatory schematics.** No compact, professor-ready
   G3 or G4 question text is committed beside the audit summaries; do not present
   those schematics as quoted dataset rows.
5. **Do not claim unrestricted spatial intelligence.** The scope is controlled,
   qualitative, text-only 2D reasoning; it excludes visual grounding, metric
   geometry and embodied navigation
   ([introduction, lines 36--42](../../../fyp-report/chapters/01-introduction.typ#L36-L42)).

## Compression guidance for the presenter

Keep each slide to one claim, one visual and at most three spoken details. Expand
the acronyms once, then use ordinary words: “correct-answer count” for G,
“reasoning length” for L, and “irrelevant statements” for X. Put exact sample
counts in small footers rather than in the main narration. No Touying/Typst
source was needed for this content decision; the five-slide structure follows
the repository’s factual dependencies rather than a presentation-template rule.
