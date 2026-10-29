# Readable Explanation Replay

## Scope

Replay the 282 accepted query bases (987 paired contract/menu views) in
`depth-feasibility-compact-v1` using `readable-paired-evidence-v2`.
This is an explanation feasibility check, not a new training set, benchmark,
sampling run, seal, or training launch. Preserve the eleven-pair v1 preview.

## Evidence

- Recheck complete solver semantics and the existing supported certificates.
- Build one shared explanation outline for both styles.
- Unique Direction uses composed coordinate paths and direction exclusions.
- Ambiguous Unique Direction/Count uses two complete valid maps with different outcomes.
- Listed Direction uses witnesses for displayed possible values and checked
  exclusions for every other displayed value. Exhaustive ordinary answers
  cover the entire direction domain. Reuse a coordinate path when it excludes
  several directions; otherwise show a temporary assumption and checked conflict.
- Selection includes the membership evidence needed by the contract, including
  no-match, multiple-certain-match, and contingent cases.
- Count retains jointly feasible whole-map witnesses and exclusions justified
  by fixed memberships and incompatible pairs. Do not count marginal possibilities.
- Missing-menu Exhaustive F uses one unlisted witness. Inconsistent E uses a
  checked strict cycle or forced-coincidence conflict.
- Preserve exact-direction semantics, distinct named points, open-world ordering,
  and the interpretation of numerical maps as examples, not added distances.

## Checks

Require identical source questions, menus, answers, semantic/escape reasons,
splits, depth measurements, graph metadata, and original audit certificates.
Store newly rebuilt certificates separately from the original certificates.
Verify every output pair against deterministic checked-outline rendering and
terminal answer parsing. This is template integrity, not independent semantic
parsing of arbitrary prose. Inspect actual text and test literal comparisons.

Use the source pilot's exact pinned Qwen3.5-4B tokenizer files, an 8192-token
context, the existing 4096-token Stage-1 response allowance, and existing
assistant-only masking/Stage-2 reservation checks. No cap expansion.
Admission is whole-base: all views and both styles must pass.

Record unsupported and failed bases explicitly without retaining historical
transcripts as purported readable outputs. Check source code and source-pilot
hashes before/after replay. Refuse output-directory overwrites.

## Boundaries

This does not broaden the underlying evidence fragment: exact atomic premises
and exact-direction queries; Count supports at most eight candidates and
fixed-membership/pairwise-incompatibility exclusions. Higher-order Count
correlations or shared certificate-construction failures remain fail-closed.
All-pilot success would establish coverage of this pilot, not arbitrary inputs,
human difficulty, stylistic fluency, or production sampling adequacy.

## Approval Checkpoint

Report acceptance/failures, token distributions and maxima, representative
paired explanations, preserved-input checks, and residual coverage limitations.
Do not generate or seal a new production dataset until separately approved.
