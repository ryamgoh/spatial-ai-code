# Generator Pilot Inspection

Date: 2026-10-28

Status: initial agent spot-check only, **not a passed release audit**.

## Scope

The generator pilot constructs new parameterized stories without importing the
fourteen hand-authored fixtures. It is bounded to five samples for each of nine
recipes, plus five matched inconsistent-cycle variants: 50 attempted stories,
150 query bases, and 678 rendered contract/menu views in the default run.
These views are not independent examples or a recommended training size.

See [generated report](../generator-pilot-v1/report.md) and
[audit sample bundle](../generator-pilot-v1/audit-samples.md). The report records
family grouping, split allocation, solver/reference checks, and rejections.
The receipt records source hashes and output hashes. Both the receipt and
sample bundle deliberately retain a pending manual-audit status.

## Inspected Pairs

Both Natural and Symbolic renderings and their displayed premises/options were
inspected for these default-run view IDs:

| View ID | Inspection |
|---|---|
| `empty-0-base-which-m0-single` | Every candidate has an exclusion; E means no match, not ambiguity. |
| `singleton-0-base-which-m0-single` | One guaranteed member and exclusions for all others; letter matches menu. |
| `multiple-0-base-which-m0-single` | Multiple certain members justify E without fabricated uncertainty. |
| `contingent-0-base-which-m0-single` | Valid qualifying/nonqualifying maps for the lone possible member; E is justified. |
| `joint-0-base-count-m0-single` | Two complete maps realize counts zero and one; no invariant count. |
| `joint-0-base-count-m0-visible-possible` | Pair incompatibilities exclude counts two and three; maps realize zero and one. |
| `cycle-0-inconsistent-count-m0-single` | Conflict disconnected from the queried candidates still forces E. |
| `fork-0-base-direction-m0-single` | Shared-axis ordering allows opposite directions in different maps. |
| `coincidence-0-base-direction-m0-single` | Both coordinate equalities are established before invoking distinct points. |
| `chain-0-base-direction-m1-all-possible` | An unlisted Northeast witness justifies F without claiming full-set recovery. |
| `multiple-1-base-which-m0-all-possible` | Five eligible matches are preserved; one unlisted match suffices for F. |

No semantic or option-mapping error was found in these inspected pairs.
Earlier small-run spot-checks also inspected one Direction pair per recipe;
those are exploratory checks, not additional default-run audit coverage.

## Findings Before Training

- Correct explanations are still unnecessarily verbose in some cases.
  Unique Direction currently renders exclusions for all other directions,
  although a checked entailed exact direction can suffice. Joint Count lists
  pair conflicts explicitly. Neither style has been measured with the model
  tokenizer; no context-fit or token-savings claim is supported.
- Sampling is diagnostic, not balanced. Full/partial menu pairing introduces
  many E/F answers; preserve matching and choose final proportions explicitly.
- Inspect per-split coverage in the generated report. Global coverage does not
  imply every required case is present in each split. In this default run,
  all 33 populated query/contract/semantic-case cells occur in each split;
  that does not prove every evidence subtype is represented. Do not reshuffle after
  inspecting model results or move unsupported rows into evaluation.
- Disconnected distractors are checked individually and jointly, but they are
  only one distractor construction. Their count is not a difficulty ranking.
- Supporting axis path lengths are measured only where applicable for Direction.
  No minimum-depth claim or Selection/Count chain-length descriptor is made.
- Canonical grouping is query-independent and conservative for names, order,
  and inverse orientation. It is not semantic equivalence or rotation deduplication.
- Count evidence remains limited to fixed memberships and pair conflicts,
  with at most eight candidates. Zero rejections here do not establish support
  for higher-order correlations or arbitrary generated maps.
- Reference-backend checks cover only Direction with at most six objects.
  An attempted Selection/Count crosscheck was stopped because exhaustive
  enumeration exceeded the practical pilot budget. These categories are not
  claimed to have an independent backend check.
- Observed maximum reference lengths were 10,368 Natural characters and 6,400
  Symbolic characters. These are not tokens and do not establish context fit,
  training cost, or generated-token savings.

## Outstanding Audit

The pivot requires inspection of every populated query-family/contract/
semantic-case cell and five distinct bases per admitted evidence category.
This spot-check does not satisfy that requirement. The generated audit bundle
selects cell representatives and five stories per construction recipe, but
recipe labels must still be mapped to an explicit evidence taxonomy.

Before any release: finish and record that audit, resolve per-split scope gaps,
freeze evidence selection/templates and sampling, measure actual tokenizer
lengths, and test the downstream style prompts and decoder integration.
No training was launched and no final dataset size was selected.

## Verification

- Broad regression run: 1,747 tests and 18 subtests passed before the final
  reporting/reference-scope refinements.
- Final focused pilot, composition evidence, and interface run: 276 tests passed.
- Pilot generator and new tests pass Ruff; tracked edits pass `git diff --check`.
- Human release review remains pending; these automated results do not replace it.
