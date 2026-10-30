# All-Direction Controlled Selection Coverage

Development-only correctness check. No pool selection, production generation,
sealing, training or replacement of historical results.

## Construction

Extend the three existing unique-match Selection patterns with an explicit
query direction. Preserve the original default generator and 60-story schedule.
Cardinal constructions start from North. Diagonal constructions start from
Northeast, requiring both strict x and y inequalities for the match. Southward
chains exclude rivals; direct East/West relations exclude Northeast by equality
or the wrong sign on at least one axis. Rotate only by multiples of 90 degrees.
A 45-degree relabeling does not preserve qualitative axis constraints.

The three patterns keep their previous depth roles:

- Match chain: depth d for inclusion, direct rival exclusions.
- Rival chain: direct inclusion, depth d for one rival exclusion.
- Shared exclusion: direct inclusion, depth d for every rival exclusion.

## Fixed Check

Seed 20261101. Exactly 240 constructions: three patterns times eight directions
times depths 1-10, one sample per intersection. All are Selection questions.
Use the same construction choices across rotations. Rotations and depth variants
retain the same family ID: three families, not 240 independent constructions.
These remain existing controlled patterns, not a new structural-diversity claim.

For every construction:

1. Check consistency and possible = certain = {nominated target}.
2. Measure the requested depth, inclusion/exclusion support location and
   answer-changing premise ablation.
3. Run the existing full build_attempt admission pipeline: evidence
   certificates, name/order transformations, prompt round trip, all feasible
   displayed-possible counts, all three contracts, matched version-5 prose,
   independent text audit and pinned Qwen3.5-4B token limits.
4. Require a substantive view with printed requested-depth support under each
   contract; E/F views cannot fill those slots.
5. Save every attempt, including failures. No retries or replacements.

Publish the direction/pattern/depth matrix, tokens, failure reasons and six
Northeast examples (all patterns at depths 3 and 10). Verify output hashes and
run all composition/pivot regressions, including the unchanged legacy schedule
fingerprint.

This check establishes only this targeted Selection generator's coverage. It
does not establish production direction quotas, arbitrary graph support or
direction-balanced ambiguity, contradiction, Direction and Count datasets.
