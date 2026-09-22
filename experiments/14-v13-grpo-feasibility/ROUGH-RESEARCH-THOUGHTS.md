# Experiment 14 rough research thoughts

> Status: provisional research notebook. These are candidate questions and
> decision rules, not completed findings. Run the feasibility probe before
> selecting a larger GRPO study.

## Why RL is interesting here

The strongest V13 result so far is not simply high accuracy. The matched trace
ablation showed that reasoning representation determined whether a 4B model
could generalize under long graph interference:

| trace | original V13 | V13.1 |
|---|---:|---:|
| repeated full state | 95.5% | 85.6% |
| compact delta state | 96.2% | 97.8% |

The worlds, answers, and training configuration were otherwise matched. This
suggests a useful RL question:

> After SFT teaches a scalable spatial state representation, can outcome-only
> RL reinforce successful reasoning strategies without supplying gold traces?

The V13 symbolic solver makes this unusually clean. It can assign exact answer
rewards without human preference data or a learned reward model.

SFT and GRPO provide different supervision:

```text
SFT:  imitate this one solver-generated reasoning trajectory
GRPO: sample several trajectories, reward correct outcomes, reinforce winners
```

Potentially useful successful strategies include tracking only a relevant
subgraph, detecting contradictions earlier, using a shorter valid proof,
enumerating entities in another reliable order, or correcting an earlier
mistake. RL is less interesting if it only makes the model imitate the same
delta trace more confidently.

## Immediate feasibility question

The current 40-step probe asks only whether GRPO is technically and
scientifically viable for this model-task pair:

1. Does the hard prompt pool produce within-prompt reward variance?
2. Does the two-GPU vLLM/training loop operate correctly?
3. Does a short GRPO run improve a frozen hard holdout?
4. Does it preserve the original V13 skills?

The probe is not intended to establish a final RL gain. A positive result only
justifies more controlled experiments.

### Why calibration matters

With four completions for one prompt, useful groups contain both successes and
failures:

```text
useful:       [1, 0, 1, 0]
uninformative: [1, 1, 1, 1]
uninformative: [0, 0, 0, 0]
```

Aggregate accuracy can be 50% while every group is internally uniform: half
the prompts are always solved and half are always missed. GRPO needs variation
inside groups, not merely a balanced dataset average.

The current calibration samples 48 prompts with four rollouts each and stops
before training if the signal is degenerate.

## Candidate experiment 1 — uniform versus frontier sampling

This is likely the most inherently GRPO-specific next comparison.

### Uniform sampling

Randomly sample all eligible prompts.

### Frontier sampling

Prefer prompts where calibration produces one to three correct rollouts out of
four. Downweight groups that are always correct or always wrong.

Control:

- total generated completion tokens;
- optimizer steps;
- model and prompt pool;
- reward function;
- learning rate and KL settings.

Primary question:

> Does within-group reward-variance-aware sampling improve learning per
> generated token relative to uniform GRPO?

Metrics:

- frozen holdout gain per optimizer step;
- holdout gain per generated token;
- proportion of zero-variance groups;
- movement from `0/4 -> mixed -> 4/4`;
- original V13 and V13.1 retention;
- gains at structural scales 10, 12, and 14.

## Candidate experiment 2 — outcome-only versus process-aware reward

### Outcome-only

```text
1.0 exact final answer-set match
0.0 otherwise
```

### Light process-aware reward

Possible additions that can be solver-checked robustly:

- correct global consistency decision;
- correct queried X/Y conclusion;
- parseable final answer format.

Avoid initially rewarding exact reproduction of every state line. That could
force the model back into one prescribed trace and eliminate the exploration
advantage of RL.

Primary question:

> Does sparse outcome supervision permit better strategy discovery than dense
> process supervision, or is light symbolic shaping required for efficient
> learning?

Failure risks:

- reward hacking through malformed partial answers;
- overinclusive multi-letter answers receiving accidental partial credit;
- rigid process imitation replacing genuine exploration;
- correct final answers supported by incoherent traces.

Exact correctness should remain the headline metric even if process rewards
are introduced.

## Candidate experiment 3 — reasoning efficiency

Compare exact-answer reward with exact answer plus a soft length penalty after
a generous budget.

Question:

> Can GRPO discover shorter correct spatial reasoning without collapsing into
> unsupported guessing?

Track:

- strict accuracy;
- median and p95 completion tokens;
- invalid-output rate;
- accuracy per generated token;
- visible proof/state correctness on a manually or automatically audited
  sample;
- early-answer and no-reasoning frequency.

A length reward must not pay the model merely for being short. Correctness
should dominate, and length penalties should activate only above a soft cap.

## Candidate experiment 4 — self-correction

SFT demonstrations are perfect trajectories. They do not demonstrate recovery
from an earlier error. Outcome-only RL can reward a rollout that makes a
mistake, explicitly revises it, and finishes correctly.

Possible categories:

- correct without visible correction;
- correct after explicit correction;
- incorrect after attempted correction;
- incorrect without correction.

Question:

> Does GRPO increase successful recovery from intermediate spatial-state
> errors on long problems?

This requires a trace analyzer rather than a new reward initially. The earlier
V13 error-trace audit provides a starting point for detecting reversed edges,
missing updates, and later corrections.

## Candidate experiment 5 — presentation-order robustness

Render the same world in multiple premise orders:

- proof edges first;
- proof edges last;
- distractors interleaved;
- random order;
- contradiction-closing edge early or late.

The answer and underlying world remain unchanged.

Question:

> Does GRPO strengthen reasoning over the graph itself rather than dependence
> on familiar statement order?

Possible world-level reward:

```text
per-permutation correctness
+ bonus when all permutations are answered correctly and consistently
```

This is a cheap, controlled robustness experiment because no new semantics are
required.

## Candidate experiment 6 — shared-world consistency

Use one premise paragraph to ask direction, which-object, and count questions.
Reward individual answers and optionally add an all-questions-correct bonus.

Question:

> Does RL encourage one reusable world representation whose implications are
> applied consistently across question formats?

In inconsistent worlds, every question family should respect the same global
inconsistency. Useful metrics include per-question accuracy, all-questions
world accuracy, and cross-question contradiction consistency.

## Candidate experiment 7 — number of generations

Compare `G=4`, `G=8`, and possibly `G=16` while controlling total generated
tokens.

Question:

> Is compute better spent exploring more trajectories for fewer prompts or
> fewer trajectories across more prompts?

Expected trade-off:

- `G=4` may miss rare successful trajectories on hard prompts;
- `G=8` may expose substantially more mixed groups;
- `G=16` may waste compute if the policy is already homogeneous.

Calibration should determine whether increasing `G` is justified.

## Generalization design

Same-pool improvement is not enough. A useful RL claim should include a
structural holdout. A simple design is:

```text
GRPO prompts: scales 10 and 12
evaluation:   disjoint scale 14
```

Interpretation:

| observation | interpretation |
|---|---|
| improves 10/12 and held-out 14 | evidence of a stronger reusable procedure |
| improves 10/12 only | matched-distribution policy optimization |
| improves seen prompts but not disjoint matched prompts | overfitting/memorization |
| improves hard prompts but hurts original V13 | specialization or reward imbalance |
| no gain despite mixed reward groups | optimization/reward recipe may be ineffective |

Premise-order and shared-world holdouts can provide additional evidence that
the learned behavior is not tied to one generator template.

## Longer-term dynamic-state environment

Static V13 is almost saturated after delta-state SFT. Ordered state updates may
provide a more scalable RL environment:

```text
initial spatial state
-> move/update event
-> irrelevant event
-> another update
-> query an intermediate or final state
```

The simulator can verify every state transition and final answer. Candidate
comparisons include final-answer-only versus transition rewards, training on
four updates and testing on eight, intermediate-time queries, undo/correction
operations, and transfer from static V13.

Dynamic state tracking requires a separate semantic contract and should not be
added until the current static GRPO mechanics are known to work.

## Recommended sequence

1. **Current feasibility probe**
   - Verify within-group reward variance.
   - Run 40 outcome-reward GRPO steps.
   - Check hard holdout gain and V13 retention.

2. **Uniform versus frontier sampling**
   - Same rollout/token budget.
   - Test whether variance-aware prompt selection improves efficiency.

3. **Reward and efficiency ablation**
   - Outcome-only versus light symbolic shaping.
   - Outcome-only versus soft length penalty.

4. **Generalization and strategy analysis**
   - Held-out structural scale.
   - Premise-order robustness.
   - Self-correction behavior.

5. **Shared-world or dynamic-state follow-up**
   - Choose shared-world consistency for a final static study.
   - Choose dynamic state tracking for a richer longer-term RL environment.

## Possible research narrative

One coherent story is:

1. Repeated full-state SFT creates an artificial failure under graph
   interference.
2. Delta-state SFT removes that representation bottleneck.
3. Solver-verifiable outcome-only GRPO improves hard spatial prompts without
   gold reasoning traces.
4. Sampling prompts with within-group reward variance is more efficient than
   uniform sampling.
5. RL improves extrapolation, robustness, efficiency, or self-correction rather
   than merely increasing matched-test accuracy.

The current feasibility result determines whether steps 3–5 are actually
viable. Do not infer them before calibration and the frozen-holdout comparison.
