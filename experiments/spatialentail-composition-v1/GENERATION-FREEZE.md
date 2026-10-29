# Candidate Generation Freeze

This freezes the bounded attempt policy before generating the larger candidate.
It does not declare a passed dataset release or authorize GPU training.

## Approved Direction

Random stories plus targeted cases; pinned Qwen3.5-4B QLoRA, Natural and
Symbolic only, shared training seed 42. Descriptive acceptable accuracy loss:
2 absolute percentage points for Symbolic relative to Natural, assessed
separately for Stage 1 and Stage 2. This is not an equivalence test.

## Fixed Attempt Budget

Exactly 200 story instances, generation seed 20261029:

- 100 random-world stories: sample 5-8 distinct points, then 2 through n+1
  atomic relations between randomly selected pairs. The hidden coordinates
  are discarded as constraints; only qualitative relations enter the solver.
- 50 random-atom stories: randomly select the same size range and relation
  pairs, independently sample each exact direction. Consistency is determined
  by the solver, not assumed.
- 50 targeted story instances from the pilot's supported constructions,
  including five matched consistent/inconsistent cycle pairs. Names and
  candidate ordering are randomized so designated construction roles do not
  have fixed visible object names.

Each story receives Direction, Selection, and Count queries: 600 attempts.
Selection/Count consider all declared nonreference candidates for random
stories, independently of A-D. Direction uses the full eight-direction domain.
No repeat-until-accepted loop, post-result size increase, or rejected-to-test move.
Final admitted counts and proportions are frozen in the release manifest,
not inflated to this attempt budget.

For each accepted base, retain one full-coverage menu plus one single-slot
partial-coverage variant when feasible. Otherwise retain one menu. Render all
three contracts for each menu with identical A-D permutations across styles.
Do not independently rebalance matched views. Report the resulting E/F and
substantive-letter frequencies rather than calling them balanced.

## Supported Matrix

All three contracts for each of these cases:

- Direction: unique, underdetermined, inconsistent.
- Selection (`which`): unique, no-match, multiple-certain-match,
  underdetermined, inconsistent.
- Count: unique (including zero), underdetermined, inconsistent.

This is 33 query/contract/semantic-case cells. Every released split must cover
all 33; a missing cell blocks sealing, rather than triggering automatic retries.

Evidence categories are assigned from semantic results and selected
certificate obligations, not recipe names:

1. Exact Direction derivation and exclusion proofs.
2. Direction ambiguity with differing valid maps.
3. Unique Selection with full candidate membership justification.
4. Empty Selection with exclusions.
5. Multiple-certain Selection.
6. Contingent Selection with witness/counterexample pairs.
7. Invariant zero Count.
8. Invariant nonzero Count.
9. Variable joint Count with whole-map witnesses and checked exclusions.
10. Strict-cycle inconsistency.
11. Forced-coincidence inconsistency under distinct points.
12. Consistent-cycle controls.
13. Missing-menu escapes with an unlisted witness or unique-answer proof.

Count exclusions are bounded to fixed memberships and pair incompatibilities,
with at most eight candidates. Unsupported proof rules, higher-order Count
correlations, or any unverifiable output are quarantined with their original
split assignment. No unsupported example is released with just a solver label.

## Evidence and Audit Policy

Versioned compact selection uses a positive checked proof for Unique Direction
when available, retaining all alternative exclusions in audit data. Other
contracts retain the existing evidence selection. Both styles receive exactly
the same selected evidence. All prior fixture behavior remains regression-tested.

All rows pass solver, witness, certificate, prompt round-trip and menu checks.
Every base receives a deterministic rename/order semantic check. Designated
distractors are removed individually and together and checked against full
semantics. Random stories claim zero *designated verified* distractors, not
that every sampled premise is necessary.

Before questions/views are admitted, union query-independent canonical
story groups with explicit source families. Assign approximately 60/20/20
families to train/dev/test using a fixed hash order within source-recipe strata.
Groups spanning recipes form their own joint stratum; fewer than three
families in a stratum stay in training. Renames/reorderings and
inverse-oriented duplicates cannot cross splits. Coarse grouping is
conservative; rotated or semantically equivalent graphs are not necessarily
deduplicated.

Pre-release amendment: the initial non-stratified hash allocation put no
Unique Selection cases into dev/test. The coverage gate blocked sealing.
Source-only stratification replaces it before regenerating the same 200
stories; no generated model results, seed search, retries, or added examples
inform the revision. Preserve the initial candidate as an unsealed audit record.

Audit at least one pair per populated cell and five distinct bases per
evidence category. Record reviewer identity accurately: agent inspection is
not independent human review. Any confirmed correctness issue blocks sealing.
Reference-backend crosschecks cover small Direction problems only; report
the count and shared domain/sign code.

## Training Preparation Boundary

Two epochs, one GPU, effective batch eight, learning rate 5e-5, QLoRA rank 64
and alpha 128 using the repository's existing Qwen target modules. No packing
or length-based reordering; equal examples/order/exposure across both arms.
Save both epoch checkpoints; select using only synthetic Stage-1 development
accuracy, macro-averaged equally over the three contracts, earliest epoch on
ties. No loss-based checkpoint selection and no early stopping.

Measure exact pinned-template/tokenizer lengths before selecting sequence
length. Stage 1: greedy decoding, up to 4,096 new tokens. Stage 2: all rows,
full original question and unmodified Stage-1 response, constrained A-F answer
completion, 16 new tokens, with context overflow rejected rather than truncated.
Freeze implementation details and measured admission results in the preparation
manifest. GPU loading, masking/runtime smoke checks and resource availability
remain launch gates, not things that can be certified on this Mac.
