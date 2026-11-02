# SpatialEntail Final v1: Research Review

Date: 2026-11-02

## Implementation Follow-Up

The findings below describe the pre-fix generator. The accepted fixes have now
been implemented; the original probe evidence is preserved for context.

- Count menus: final-v1 opts into `synthetic-count-options-0-4-v1`, using four
  distinct values from 0-4 in every answer category. Full-domain solver
  semantics, quotas, scene seeds and historical defaults are unchanged.
  External menus are untouched. F remains omission of a sole answer.
- Review packet: `review-selection-v2` samples ten per query/case where
  available, supplements split/query/level coverage, contingent cases and Count
  values per split/level, and includes the longest native examples per split/style.
  A receipt lists selected IDs and reasons, explicitly `selected-not-reviewed`.
- Direction/variant coupling was left unchanged as a coverage limitation.

Validation: the final combined suite passed 775 tests, including real
generation over 601 representative planned questions. A fresh 176-synthetic /
3-external smoke run passed export and manifest checks. All 80 Direction/Which
references and five external data files were unchanged versus the previous
smoke; all 96 Count rows retained their labels with revised menus.

Replaying the revised menu sampler for all 2,000 planned train Count menu seeds
gives 300/2,000 (15%) for the formerly successful first-small-option rule and
600/2,000 (30%) for the earlier unique-small-option rule. Every menu now has
four values <=4, so these rules reduce to always A and always E, respectively.
This is a menu-only replay using checked representative semantics, not full
corpus construction/admission. An exhaustive sampling-law regression enumerates
all 120 ordered menus: with uniform sole values and letter positions, menu
probabilities match across A-D/E/F. Finite-seed artifacts and other nuisance
features still need the remaining shortcut review. Human review and training
readiness have not been asserted.

Scope: the current generator and experiment design, using Orchestra Research's
autoresearch evidence/synthesis workflow and ML reviewer criteria. This is a
bounded review, not a new experiment design, training campaign, or literature
novelty assessment. No production code, quotas, labels, or historical artifacts
were changed during the review itself; the later implementation is recorded above.

## Findings

Reassessment: the original review overstated the equivalence and urgency of
these findings. There is one confirmed menu-information concern, one incomplete
review artifact, and one construction-coverage limitation with unmeasured impact.
None is a demonstrated wrong-label or train/test leakage bug.

### Confirmed Research Concern: Count Menus Expose A Numeric-Range Cue

On a disjoint second sample of 400 train-only Count questions, the rule "choose
the first displayed option <=4, otherwise E" scored 188/400 (47%) without reading
the premises, versus 120/400 (30%) for always E. It answered 162/240 (67.5%) of
the A-D rows correctly. All 400 passed real generation and evidence audits.

An independent analytical check supports the mechanism without relying on that
sample. Consider the order-independent rule "choose uniformly among options
0-4; otherwise E". If the legal Count domain has N values, the number K of
additional small distractors in an A-D menu has probability
`P(K=k) = C(4,k) C(N-5,3-k) / C(N-1,3)`. Its conditional correctness is
`1/(k+1)`. On E cases the rule is correct when all four options exceed 4, with
probability `C(N-5,4) / C(N,4)`. On F it is never correct.

Averaging these probabilities over the actual train Count slot allocations and
generator scene sizes gives 46.35% expected accuracy versus the 30% no-information
majority baseline. These are expectations over the menu-sampling law, not measured
full-corpus scores. Domain sizes vary from 12 to 21 values across train levels;
the same rule has expected accuracy 35.58% at L1 and 51.55% at L8. Thus the cue
can also change with scene level, despite matched answer-category proportions.

The generator forces sole Count answers into 0-4 but samples distractors from
the much larger legal Count domain. Correct A-D position balancing does not
remove this value-range cue. See the probe method and limitations below.

This is information about the dataset prior, not a logically valid way to solve
an arbitrary spatial question. Some menu informativeness is normal in multiple
choice tasks; above-chance performance alone is not sufficient to declare a bug.
Here the concern is concrete because the extra information comes from a
controllable mismatch between true-answer and distractor ranges, and the stated
study aims to assess premise-based composition. The rule still fails over half
the questions. No evidence here shows that a trained LLM uses it, that either
supervision arm benefits more, or that it invalidates their eventual comparison.

Recommended correction: make synthetic Count menu-value distributions overlap
across answer cases, while retaining valid distinct options, 0-4 true-answer
targets, and omission-only F. Freeze the revised construction rule and validate
it with train-only menu baselines before generating the new benchmark.
Do not alter SpatialMap-TQA-Corr menus. This recommendation is not a claim that
any particular replacement menu policy has already been validated.

### Coverage Limitation: Direction and E-Recipe Variant Are Coupled

`spatial/v2/entailment_generation.py:327-345` chooses direction with `seed % 8`
and recipe variant with `seed % 2`. Variant is consequently fixed by direction
parity, including every retry. The 8,100 declared ambiguous/inconsistent
construction candidates occupy only eight of the sixteen direction/variant
combinations. There are zero off-parity combinations.

For contextual ambiguity, this ties cardinal directions to the shared-anchor
recipe and diagonal directions to the disconnected-anchor recipe. Inconsistent
recipes are coupled similarly. This does not invalidate the labels, but means
direction coverage is not independent coverage of the two construction mechanisms.
It prevents separating direction effects from recipe effects if that comparison
is attempted. However, the agreed experiment does not require all sixteen joint
cells, the coupling is shared across splits, and direction alone does not
determine A-D/E/F. No prediction experiment demonstrated harmful exploitation or
an aggregate arm-comparison bias from this coupling. Calling it a demonstrated
harmful confound or mandatory pre-generation fix was too strong.

Optional robustness improvement: use separately salted deterministic streams for
direction and variant, then audit joint coverage. Preserve the agreed answer and level
quotas; regenerate and re-audit any new candidate. There is no requirement here to
make ambiguity or inconsistency harder at higher levels.

### Confirmed Validation Gap: Review Packet Misses Most Levels

`spatial/v2/entailment_generation.py:980-989` takes the first ten questions in each
query/case cell. Build order starts with extrapolation.

Replaying this selection on the current complete slot schedule yields:

- 130 extrapolation questions; no train, dev, or in-range questions.
- 127 questions at construction level 9 and three at level 10.
- No coverage of levels 1-8 and no explicit longest-example selection.

This fails the plan's supplementary every-depth and longest-case review coverage.
It does not make the generated questions incorrect, but the packet cannot support
the intended readability and supervision review.

The workflow explicitly retains paired review in its pending checks and sets
`training_ready: false`. This is unfinished validation, not a false passing review
receipt. It must be completed before claiming the plan's review gate passed; it
does not prevent constructing candidate data.

Recommended correction: retain ten per query/case, supplement deterministically
across splits/levels and semantic subcases, and include actual longest examples
after token measurement. Record reviewed IDs and findings.

## Count Menu Probe

Hypothesis: true sole Count answers are restricted to 0-4 by the declared policy,
while distractors are sampled from the full legal range, 0 through the number of
nonreference objects. The menu may therefore expose a numeric-range cue.

Relevant code:

- `experiments/spatialentail-final-v1/generation_policy.py`: `i % 5` Count targets.
- `spatial/v2/composition_contracts.py:64-70`: complete legal Count answer domain.
- `spatial/v2/entailment_dataset.py:183-208`: uniform shuffled distractor pool and
  omission construction.

Exploratory probe 1 used 400 train-only Count slots: 240 A-D, 40 F, 90
no-entailment E, and 30 inconsistent E. Each case was sorted by
`stable_int("orchestra-probe/" + slot.identity(0))`, selecting its first target
number of rows. All were generated and independently audited using the current
workflow, alternative 0, with the pinned tokenizers.

The fixed rule was: if there are no options <=4, answer F; if exactly one, answer
its letter; otherwise answer E. It scored 132/400 (33%) against 120/400 (30%) for
always E. This first result is not strong evidence of shortcut performance.

Exploratory follow-up: before generating a second sample, changed the rule to
"choose the first displayed option <=4, otherwise E". Selected the next target
number of hash-ranked slots in each case, disjoint in slot identity from probe 1.
The second sample preserved the same 60/30/10 proportions; all 400 passed.
It scored 188/400 (47%) against 120/400 (30%) for the majority baseline.

| Case | Questions | Correct |
|---|---:|---:|
| A-D | 240 | 162 |
| F | 40 | 0 |
| No-entailment E | 90 | 22 |
| Inconsistent E | 30 | 4 |

This is a targeted follow-up prompted by the first result, not preregistered
confirmatory evidence or a reported test-benchmark score. The rule was fixed
before observing the second sample's results. It demonstrates exploitable menu
information on these train draws, not how strongly a trained LLM will exploit it.

No trained predictor, test-set tuning, or model evaluation was used. These are
diagnostic draws, not a released dataset or a family-grouped generalization
estimate. No canonical cross-row preflight was performed for these probe draws.

## What Remains Supported

- Current quotas and one-menu allocation match the agreed design.
- Entailment is checked over the complete domain; F is sole entailment omitted
  from the menu, not ambiguity.
- Independent evidence and selected-depth checks remain in place.
- Semantic cases require correct evidence and level coverage, not increasing
  ambiguity/contradiction difficulty. Their null-depth treatment is appropriate.
- Both supervision styles share questions, menus, labels, and evidence.
- External evaluation retains its own source-menu distribution, not synthetic
  quotas. The proposed synthetic menu investigation must not rebalance it.

Re-ran 103 contract, independent artifact-audit, and selected-explanation tests:
all passed. Prior 601-case generation coverage is useful but does not test whether
the resulting distribution permits prediction from menus alone.

## Limits And Next Steps

No new wrong-label counterexample was found in this review. A complete
10,800-question admission campaign and full overlap/coverage audit remain pending.
The findings concern sampling and research validation, not a reason to abandon
the experiment or reinstate a semantic-difficulty gate.

Address the Count cue through an explicit synthetic-menu design decision and
train-only validation before relying on benchmark accuracy as premise reasoning.
Complete representative review before passing that release gate. Treat independent
variant sampling as an optional robustness improvement, not a demonstrated
correctness repair. Retain the existing shortcut-review and training-readiness
gates. Passing semantic tests is evidence of correctness, not proof of unbiased
sampling; these findings also do not establish that the entire dataset is bad.
