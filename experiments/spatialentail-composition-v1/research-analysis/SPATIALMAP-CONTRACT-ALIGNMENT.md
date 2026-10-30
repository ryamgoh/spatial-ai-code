# SpatialMap-TQA and the Three Contracts

## Verdict

Partially aligned, not a drop-in match. The synthetic examples teach compatible
qualitative spatial operations, but the latest 240-case Selection pack neither
replicates SpatialMap-TQA's scene distribution nor tests distinctions between
the three contracts. Original SpatialMap-TQA is single-choice; our Unique,
Listed and Exhaustive rules define a separately specified task.

The recent 240/240 result establishes controlled directional and depth coverage.
It does not establish all-contract transfer to the external benchmark.

## 1. Native Answers Are Not Our Three Contracts

Released SpatialMap-TQA rows provide four options and one oracle answer. The
original answer does not by itself establish that this answer holds in every
map consistent with the text under our exact-axis, distinct-point semantics.
Listed and Exhaustive require new instructions and solver-derived labels.

| Our contract | Relation to SpatialMap-TQA |
|---|---|
| Unique | Same broad single-answer task shape, but stronger declared invariance semantics; original labels are not automatically valid |
| Listed | New possible-value, multi-answer task over displayed values, not the original benchmark contract |
| Exhaustive | New complete-domain task; particularly sensitive to omitted values and the declared candidate domain |

Do not score all three against the original single oracle letter. Keep the
historical Corr SINGLE evaluation separate from the new six-option views.

Primary-source research is documented in
[the companion source note](SPATIALMAP-PRIMARY-SOURCE-ALIGNMENT.md).
Local implementation: [current encoder](../../../spatial/v2/composition_contracts.py#L227)
and [external-view specification](../pivot_20261026.md).

Scoring and response order also differ. The original prompt asks for an answer
first, then an explanation. Its released evaluator heuristically extracts an
answer and tests lowercased reference substring membership, despite the README
calling the metric exact match. Our terminal `Answer:` convention and strict
letter-set scoring are a different operational protocol. See the
[official scoring code](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/evals/evaluation.py#L266-L298)
and companion note; do not present these accuracies as directly interchangeable.

## 2. The Latest Pack Does Not Discriminate Contracts

The actual direction-coverage-v2 payload contains:

- 240 Selection bases, all with exactly one guaranteed match.
- 480 menus, each expanded into three contract views.
- All 480 menus have the same gold under all three contracts: one substantive
  letter when the match is displayed, or F when it is omitted.

Thus 1,440 verified views are not 1,440 tests of contract discrimination.
These are useful targeted spatial exercises. The broader generator's ambiguity,
multi-possibility, incomplete-menu, no-match and inconsistent cases are needed
for the contract comparison. This finding concerns the latest targeted pack,
not a claim that the broader generator lacks those cases.

## 3. Actual Scene Structure Differs

A census of all 1,500 locally saved released rows found 500 questions of each
family. Every row has six objects, nine distinct pairwise constraints and a
connected undirected relation graph. All premise directions are diagonal.
Nine unique edges on six connected vertices gives four independent undirected
cycles; this describes graph structure, not logical inconsistency.

This is about **premise** directions, not every question: original Count
questions use all eight directions. The primary-source census found 262 cardinal
Count questions, all with native reference zero. Thus removing cardinals from
our curriculum would not follow from the diagonal-only premise vocabulary.

The latest controlled pack has 6-16 objects, 5-7 eligible candidates and tree
constructions. Half its queried directions are cardinal. It contains chain
support and rival exclusions, not a benchmark-matched mixture of connected
extra information and alternative paths.

Matching all eight direction labels does not establish matching the scene
distribution or relevance-selection demands. Nor does longer controlled depth
establish difficulty equivalence to the denser source scenes. The random-scene
component can help address this, but its final mixture and transfer remain
unmeasured.

Naming and prompt presentation also differ: the source uses multiword named
places; our controlled examples use Object identifiers, explicit formal rules,
an Eligible declaration, and A-F. These are deliberate interface choices, not
an exact reproduction of the original prompt.

## 4. Candidate Domains Are an Unresolved Adapter Decision

The historical adapter restricts Direction outcomes to the four diagonals:
[adapter lines 134-140](../../../spatial/v2/spatialeval_adapter.py#L134-L140).
The new interface requires all eight:
[validation lines 40-55](../../../spatial/v2/composition_contracts.py#L40-L55).

For the original Selection wording, the historical parser takes the candidates
from the four options:
[text parser lines 312-338](../../../spatial/v2/text.py#L312-L338).
All 500 parsed Selection rows have those four candidates. Count instead ranges
over all five nonreference scene objects.

Consequences:

- Preserving the four-candidate Selection domain, while requiring four distinct
  valid A-D values, means every menu covers the entire domain. Listed and
  Exhaustive coincide; a genuine partial menu cannot be constructed under
  those constraints.
- Expanding Selection eligibility to all five scene objects permits a partial
  menu, but is an explicit scope change. It can change Exhaustive and Unique
  labels. It must not happen implicitly when menus change.
- Extending Direction from four to eight values can introduce possible cardinal
  outcomes omitted from the original four diagonal options.
- Original multiword names also fail the current ASCII-identifier validation
  without a documented reversible representation mapping.

These are not changes to make silently in the frozen Corr SINGLE release.
The current composition modules contain no completed external three-contract
preparation pipeline. Existing synthetic checks do not substitute for that
adapter's semantic verification.

## 5. Concrete Source Questions

I ran the current solver and encoder on four source questions. I used a
bijective ASCII renaming, preserved the original A-D values, and used all eight
directions for Direction questions. The alternate Selection scope below is
explicitly a separate calculation.

| Source ID | Original gold | Current Unique | Listed | Exhaustive |
|---|---|---|---|---|
| 2000.0, Direction | A | A | A | A |
| 2003.0, Direction | C | E | CD | F |
| 2000.1, Selection, original four candidates | B | E | BD | BD |
| 2000.1, Selection, all five nonreference objects | B | E | BD | F |
| 2000.2, Count | D | D | D | D |

For 2003.0, North, Northeast and Northwest are possible under our rules.
North is absent from the original menu: Unique is E, Listed is CD, and
Exhaustive is F.

For 2000.1, Narwhal's Novelties and Police Supply Store are possible within
the four-option domain, with neither guaranteed. Expanding the domain adds
the unlisted Oz Oddities as a possible match, changing Exhaustive to F.

These are conditional semantic results, not proof that the original authors'
intended geometry or labeling algorithm is wrong. Original explicit
open-/closed-world rules and a full direction calculus are not established
by our prior source inspection.

The saved historical audit reported 667 exact oracle matches and 833 possible
but non-unique answers under its own SINGLE interpretation. Its input hash
matches the raw file examined here. Those are not fresh counts for the new
eight-direction, independently scoped contracts.

## Recommendation

Keep the verified controlled examples as one training cohort. Do not describe
them as a faithful SpatialMap-TQA reproduction or evidence that all external
contract views are ready.

Before production, settle one external adapter specification:

1. Preserve Corr SINGLE unchanged for historical comparability.
2. For new views, explicitly choose and freeze Direction and Selection domains
   before menu construction. A conservative Selection comparison keeps the
   original four eligible candidates and accepts that Listed/Exhaustive coincide;
   an all-scene comparison must be labeled as a separate scope intervention.
3. Apply the same declared spatial semantics, contract rules and serialization
   in synthetic training and new external evaluation, with newly checked labels.
4. Retain ambiguous and multi-possibility synthetic examples that actually
   distinguish contracts, and a random-scene component with connected irrelevant
   information. Do not substitute the latest unique-match pack for that mixture.

This calls for a clearly specified external adaptation, not another broad
generator redesign. Prediction transfer still requires experiments.

## Evidence and Limits

[Machine-readable measurements](SPATIALMAP-ALIGNMENT-MEASUREMENTS.json) record
the raw SHA-256, structural census, controlled-pack census and solver examples.
The raw file is [data/spatialeval_org.jsonl](../../../data/spatialeval_org.jsonl).
Historical counts come from
[the preserved correction summary](../../spatialentail-randomized-v2/prepared-corr-eval/correction-summary.json).

This analysis inspected all 1,500 rows structurally, all 240 latest controlled
records, current parser/encoder code and the external-view specification.
It did not generate a new 1,500-row relabeling, train a model, or measure
transfer accuracy. No source code was changed for this analysis.
