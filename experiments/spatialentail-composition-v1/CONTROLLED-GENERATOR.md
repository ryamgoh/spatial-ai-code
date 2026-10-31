# Controlled Generator

## Interface

New construction work uses `spatial.v2.entailment_generator.SceneRequest` and
`build_scene`. The construction logic is local to that module: it does not call
private construction functions in the earlier prototypes or sampler.

```python
from spatial.v2.entailment_generator import SceneRequest, build_scene
from spatial.v2.solver import Direction

request = SceneRequest(
    span=8,
    direction=Direction.NORTHEAST,
    kind="count",
    case="sole",
    count_value=2,
)
story, problem = build_scene(request, identity="scheduled-question-id", family="lineage-id")
```

The result is an **unverified scene**, not a training example. It must pass the
existing full-scene entailment, evidence, depth, reference and token checks.
The caller supplies identity and lineage; neither certifies canonical uniqueness.
Construction performs no file I/O, sampling loop, retries, allocation or split assignment.

## Rules

1. Choose the intended semantic case: `sole`, `no-entailed`, `multiple` (Which
   only), or `inconsistent`. There is no Listed contract. A-D versus F is decided
   later by whether the sole entailed value appears in the selected menu.
2. Construct direct or paired axis supports for positive cases. For Which and
   Count, support intermediates and all added objects participate in the query.
   Intermediate exclusions are necessary to preserve the intended answer.
3. For E, use the tested shared-anchor or disconnected-component ambiguity,
   or strict-cycle contradiction recipes. These are a small hand-designed
   collection, not an exhaustive distribution of spatial problems.
4. Add bounded, excluded-side context using the shared case-size budget, then
   randomize inverse wording, sentence order and ordinary place names.
5. Independently verify the resulting entire scene. Intended semantics are
   never sufficient to assign a gold answer.

## Supported Parameters

- `span`: integer 1-10. `request.requested_depth` is that number for positive
  cases, but `None` for no-entailed/inconsistent E. Scene size is not proof depth.
- `direction`: one of the eight `Direction` values.
- `kind`: `direction`, `which`, or `count`.
- `case`: the four semantic cases above.
- `count_value`: 0-4 for sole Count cases, including exact zero.
- `variant`: 0 or 1, selecting the preserved E mechanism. It does not add a new
  mechanism to positive cases.
- `size`: `contextual` (default), `compact`, or `compact-extra`. The compact
  recipes are supported only at span 1. These are existing tested size recipes,
  not a new general-purpose arbitrary-size control.

Which multiple-entailment cardinality still follows the tested `2 + span % 3`
recipe. Context edges remain restricted to the excluded side. Those design
restrictions were deliberately retained in this behavior-preserving refactor.
The generator is parameterized and documented, not free of custom logic.

## Replacement Sampling

`build_scene` now also accepts an optional `structure_seed`. The default retains
the exact previously tested scenes. With a structural seed, the v2 recipe varies
excluded-side relation directions as well as acyclic context attachments. It
does not edit the positive support. Auxiliary coordinates choose mutually
consistent context relations only; they never become extra constraints, answers
or supervision. The whole resulting scene still requires solver verification.

`contingent=True` is available only for sole Which cases. It introduces an
unordered object relative to the reference through a shared anchor, retaining
one entailed answer while another is merely possible. These examples belong to
the semantic cohort because the complete membership-depth descriptor is not
defined for a contingent alternative. Positive evidence is still checked.

The replacement allocation is in `entailment_replacement_allocation.py`; the
fail-closed build/export entry point is `build_controlled_training.py`. It
preserves numerical depth, Count-value, family and evidence targets while
removing the retired generation-source percentages. Candidate identity changes
are never counted as distinct questions without canonical checks.

## Preservation

`entailment_controlled_prototype.py`, `entailment_controlled_coverage.py`, their
runners, manifests and reports remain historical reproducibility artifacts.
They are not rewritten to import this module, because doing so would invalidate
their recorded source fingerprints. New construction callers should use this
module instead of extending the private prototype functions.

Regression checks compare all 968 fixed coverage requests against the preserved
implementation. Typed scenes and problems match exactly, including naming and
sentence ordering; only the `story.recipe` provenance tag changes.
Additional tests independently rebuild semantic evidence and check actual
positive depths through the new interface. This is a regression run over fixed
fixtures, not another sampled candidate batch.

## Remaining Work

The 1,200 A-D / 600 E / 200 F per-query target is unchanged. The replacement
workflow is implemented, but its actual completion and audit status must be
read from `CONTROLLED-TRAINING-BUILD-v2.json`, not inferred from this guide.
Historical commands and existing dataset payloads retain their original policy.

The `unique-readable-explanations-v2` renderer removed redundant sole-answer
rejections and internal-contract boilerplate, but left coordinate-heavy Natural
text and repeated ambiguity/membership explanations. The subsequent
`spatial-prose-shared-chains-v3` reference-only revision addresses those:

- Natural text uses compass-axis language without confusing one-axis comparisons
  with exact compass relations.
- Direct facts are explained directly.
- Count certificates share cited chains, including reusable prefixes, while
  accounting for every scene object.
- Ambiguous Direction/Count uses two saved valid arrangements with different
  outcomes rather than enumerating every rejected answer value.
- Which rejections are grouped by checked case, with every other object covered.
- Answer conclusions map the derived result to the menu without repeated
  contract boilerplate.

The renderer implementation is `entailment_readable_v3.py`, dispatched explicitly
through `entailment_evidence.py`. `rerender_controlled_training.py` permits only
reference text, reference-version and token-measurement changes. It preserves
questions, menus, labels, evidence, identities, split assignments and all depth
metadata. The status is recorded in `SPATIAL-PROSE-RERENDER-v3.json`. Old render
versions remain supported; neither a re-render nor its audit authorizes training.
