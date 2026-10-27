#pagebreak(weak: true)
= Experimental Design

== Matched Training

Train two adapters, Natural and Symbolic, on matched mixed-contract examples.
Underlying problems, contract/menu views, split membership, final answers,
base model, adapter setup, example order, and training schedule are held fixed.
Only reasoning representation and any necessary style instruction differ.
Both arms include Unique Answer, Listed Possibilities, and Exhaustive
Possibilities; three contracts do not require three separate models.

Use one shared training seed initially. There is no answer-only arm, retired
logical-proof baseline, or automatic model, seed, or data expansion. Equal
example exposure need not imply equal token compute, so actual target-token
counts and training cost are recorded for each arm.

After feasibility checks and before training, freeze the dataset size,
semantic and menu proportions, chain-length ranges, evidence selection,
rendering templates, model/tokenizer revisions, and training settings.
These values are not inherited from older experiments.

== Checkpoint Selection

Select each adapter's checkpoint by highest Stage-1 answer accuracy on the
held-out synthetic development set, with its matching style instruction.
Missing or invalid final answers are incorrect. Freeze the checkpoint
evaluation schedule, score aggregation, and deterministic tie-break rule
before training.

Use the same selected checkpoint for both test stages. Neither Stage-2 scores
nor synthetic test or SpatialMap results influence checkpoint selection.

== Two-Stage Evaluation

Evaluate four checkpoint/instruction combinations: Natural adapter with
Natural instructions, Symbolic adapter with Symbolic instructions, and the
same untuned base checkpoint with each of those styles. Each trained system
is compared with its style-matched untuned baseline.

Stage 1 allows up to 4,096 generated tokens for the initial reasoning and
answer, identically across all combinations. Preserve the raw response, final
answer if present, and finish or truncation status.

Stage 2 runs on every example, including valid and truncated Stage-1 outputs.
The same checkpoint receives the original question and complete Stage-1 output,
followed by: "Based on the question and your reasoning above, give only the
final answer." Structured decoding allows the contract-valid letter answers,
including E or F alone.

Stage 2 is constrained answer completion, not pure extraction or independent
verification. It may complete or revise a decision. Report Stage-1 and Stage-2
accuracy as separate, equally main results; do not replace Stage-1 failures
with Stage-2 successes.

Before execution, freeze style prompts, stop rules, Stage-1 parser, Stage-2
serialization and grammar, answer-token allowance, decoding settings, and
context capacity. The replayed question and full response must fit without
silent truncation.

== External SpatialMap Check

SpatialMap is external evaluation only. Keep the frozen Corr SINGLE release
separate from newly versioned Unique, Listed, and Exhaustive views. These
variants preserve each source question's family and identity.

External answer-accuracy evaluation may use verified questions, candidate
domains, menus, and labels without rendered reference explanations, provided
semantic evidence is retained and explanation availability is reported.
This exception does not relax synthetic release requirements.

== Measurements and Interpretation

Report accuracy by contract, query type, semantic case, and escape reason.
Report substantive-answer and E/F performance separately. Unique E alone
cannot distinguish uncertainty from inconsistency or Selection no-match and
multiple-match cases. Exhaustive F is not evidence that a model knows the
unlisted answer set.

Stage-1 generated tokens and total Stage-1 plus Stage-2 generated tokens are
separate main cost outcomes, including incorrect and truncated examples.
Also record both passes' input tokens. Replayed Stage-1 text is input to
Stage 2, not newly generated output. Token savings are not runtime savings.

Report format compliance, Stage-2 rescues and regressions, answer changes,
and truncation-stratified results. Compare Natural and Symbolic on paired
base IDs; multiple views of one story are not independent observations.
Reasoning-validity measurements require explicit verifier coverage.

Freeze an acceptable accuracy-loss tolerance before viewing results. One seed
and small samples cannot establish seed robustness or statistical equivalence.
Symbolic notation may be shorter but less accurate, or neither adapter may
improve on untuned. Such outcomes do not invalidate the dataset contribution.

== Bounded Schedule

The first three days target evidence feasibility, matched renderers, and
release gates. Days four through seven target the two pilot arms and
development checks. Week two completes fixed evaluation and comparisons;
week three is reserved for writing, figures, limitations, and presentation.
Correctness gates remain mandatory if the schedule proves difficult.
