# Reviewed study implementation receipt

Date: 2026-10-11. Uncommitted implementation; no GPU training launched.

Superseded scope note (2026-10-12): this receipt records an earlier proposed
study and is retained only as implementation provenance. The active FYP uses
four prepared answer-only/Natural jobs (two seeds each). Replayable Symbolic
certificates and corrupted variants are offline verifier fixtures, not SFT arms.
The production release is now sealed at 8,000 Train, 775 Dev, and 4,300 Bench
bases; later protocol and preparation receipts are authoritative.

## Implemented changes

- Three fully specified training compositions: DIR MP→TRC, SEL IFF→TRC,
  CNT NEG→TRC. Two same-provenance holdouts: DIR DJE→TRC, SEL CSR→TRC.
- Replay-valid ordered dependence and individual logical-support removal
  interventions. Direct-shortcut corruption is rejected. No global-minimal-proof
  assertion is made.
- Candidate corpus assembler with exact role quotas, all-or-nothing paired
  tokenizer admission, nested keyed ranks, cross-role contamination checks,
  separate instance/skeleton signatures, hashes, and per-arm token totals.
- Bounded immutable quality-review batches and aggregate coverage validation.
- Paired cell-stratified macro bootstrap and Holm adjustment helpers.
- Runnable thirteen-cell CORE engineering smoke configuration and explicit
  launcher `--config` option. No production benchmark is marked sealed.
- One primary 4B model with six central SFT runs; optional mechanisms and model
  scaling; separate matched/composition/depth outcomes; withdrawn two-point
  saturation rule and old 17K/5.8K-test allocation.

## Verification

- V2 plus launcher regression suite: 453 passed (37.54 seconds), before the
  final validation-only diagonal policy guard; focused checks cover that guard.
- Both composition registries: every listed cell probes ready with no failures.
- Real pinned Qwen3.5-4B tokenizer release smoke: six training bases, three
  development bases, two holdout bases; all three supervision views admitted;
  nested half-training pool contains three bases. No rejection in this smoke.
- Depth-eight answer-only samples for DIR/SEL/CNT all admitted. Reserved
  prompt-plus-output lengths: 4,273 / 4,348 / 4,347 tokens under 8,192. Measured
  axis/member depth was eight. This is a representative check, not proof that
  all future depth-eight instances will admit.
- CORE engineering submission graph dry-runs; no jobs submitted.
- Typst report compiles and local Markdown links resolve.

## Remaining release gates

Candidate 8K training, 950 development, and 2,700 benchmark quotas have not been
met or approved. Structural diversity, marginal coverage, all-variant context
admission, final quality receipts, and sealing remain required. The tiny pilot
and smoke are not final test data. Paired statistics are conditional on trained
checkpoints and the exact frozen skeletons. Human acceptability is not claimed;
selective human appeals remain the authorized workflow.

## Structural diversity blocker found by independent review

The current three-object/two-premise DIR-ARC-UNI template has at most 243
syntactic structures before admission filters, and a 600-candidate sample
contained only 120 unique signatures. The candidate 400-base training quota
alone exceeds that bound. Controlled depth constructors also repeat a small
set of chains. Thus the current candidate production allocation is not
feasible under strict structural uniqueness. It is a target for design work,
not a release-ready configuration. Keep the uniqueness gate: add genuinely
different support/distractor topologies and re-pilot, or reduce the relevant
quotas before sealing. Engineering smoke remains executable and explicitly
does not establish structural generalization.

### Bounded follow-up repair

Natural and Symbolic exposure totals now use format plus supervision-arm keys.
The depth gate rejects a declared maximum above four in train/dev and checks
achieved difficulty before admission. The statistics helper also produces a
paired macro randomization p-value for Holm adjustment. Unused declarations
and redundant constructor checks were deleted. Production DIR-ARC policy was
expanded to six objects/seven premises; depth fillers now vary support-independent
edge subsets. Thirty Direction depth-three samples had thirty signatures and
retained measured X/Y depth three. Quota feasibility and all-variant long-depth
admission still require a full pilot; the earlier hard bound applies to the
original minimal policies rather than the expanded candidate configuration.

The final regression run passed 456 tests in 39.75 seconds. Expanded anchor
checks showed that full Symbolic SEL/CNT targets require more than the original
4K envelope. The production candidate uses 12K train/output and 16K evaluation
limits; this compute choice is explicitly unsealed. No training is launched
until budget, full quotas, and release gates pass.

## Independent anti-slop verdict

The final reviewer-only follow-up passed the engineering implementation after
package-token accounting and achieved-depth gates were fixed. It found no new
blocking replay, nesting, intervention, or batch-coverage regression. Remaining
release gates are full unique quotas, all-variant context admission, production
config alignment with the candidate budget, coverage/cost/quality audits, and
sealing. The 12K trace budget and depth preference remain unsealed; no costly
training run is authorized by this engineering pass. No commit was made.

## Stronger hybrid extension

The three-cell minimum is superseded by four seen skeletons crossed with all
DIR/SEL/CNT contracts (12 cells): MP→TRC, IFF→TRC, TRC→MP, DJE→TRC. Three
reserved interleavings produce nine cells: MP→TRC→MP, MP→DJE→TRC,
TRC→MP→TRC. Ordered dependencies require distinct steps, with each an ancestor
of the next; individual support-removal interventions still reject shortcuts.
All component rules appear in training. The holdout tests the three named
interleavings, not arbitrary reasoning or unseen operators.

The proof engine now recomposes checked X/Y paths into exact spatial atoms
needed by logical antecedents. This allows spatial→logical and repeated
logical/spatial transitions. The new step is checker-validated and rejected by
an axis-corruption regression. Automatic spatial antecedent generation is
unscoped; arbitrary nested spatial case reasoning remains outside the claim.

All 21 registry cells probe. Full regression before the last corruption test:
479 passed in 50.94 seconds. Real pinned-tokenizer smoke admitted 24 train,
12 dev, 9 holdout bases with all three packages; the nested half pool has 12
train bases. Engineering smoke permits structural repeats and therefore is not
a final diversity receipt. Candidate quotas: 8K train, 775 dev, 4,300 benchmark.
No production quotas, benchmark seal, new model-judge quality run, or SFT were
executed. Changes remain uncommitted.

Final stronger-hybrid regression: 480 tests passed in 48.18 seconds; report
compiled, composition lint passed, and local Markdown links resolved. The
previous independent anti-slop pass predates this stronger-hybrid extension;
it is not evidence of an independent review of the new bridge rule.
