# Bounded Depth Feasibility Protocol

Status: written before execution; exploratory engineering pilot, not a
preregistered statistical experiment or a production release.
No commit, training, final benchmark generation or seal is authorized.

## Fixed Schedule

- Seed: 20261101; solver timeout: 5,000 ms per solver call.
- 60 constructed stories: six at each requested depth 1-10.
- Each depth has three provisional roles and two samples per role. At 1-8
  these roles are train/dev/in-range-test; at 9-10 all are extrapolation-pilot.
- One sample in each pair has one fixed qualifying member. The other cycles
  through 0, 2, 3, 4, 5 fixed members using depth and role indices.
- Five eligible members; nominated candidate C0 has the long positive support,
  or a long negative support in the zero-member case. Other candidates have
  direct fixed memberships. These are feasibility scaffolds, not a claim of
  rich production graph diversity.
- Randomize the exact direction, compatible chain-edge directions, 0-3 attached
  leaf branches, 0-2 disconnected distractor atoms, labels, orientations and
  premise order. Depth can stay fixed while context structure varies.
- Add 12 random-world stories, 12 independently random atomic-relation stories,
  and the existing ten targeted stories from `construct_stories(seed, 1)`.
- Three queries per story: 94 stories / 282 query attempts. No retries,
  replacement seeds, automatic expansion or relaxing failed admission gates.
- Produce one primary menu per accepted base; add the one-slot omission menu
  only when available and the stable base-ID hash is divisible by four.
  Retain all three contracts and both styles for each menu.

## Role and Identity Rules

Deliberately related depth variants share a source family and provisional role.
Depths 9/10 use disjoint source families from 1-8. Provisional roles are fixed
before semantic checks. Canonical duplicate collisions between roles quarantine
every member of the affected source families; no reassignment after rejection.
Pilot families must be excluded from any future final benchmark.

## Checks and Outcomes

Reuse existing solver, proof builder, paired renderer, exact prompt parser and
tokenizer template. Do not modify frozen release sources.

For consistent unique Direction, measure shortest x/y supports and their max.
For fixed Selection/Count, measure all positive and supported negative
memberships; require a nominated maximum-depth support premise whose removal
changes the complete query semantics (Count possible counts, not identities).
Unsupported or inapplicable cases have null depth, not zero.

Constructed rows must match their requested depth; random and targeted rows
are semantic controls, not evidence that a depth quota was achieved.
Test deliberate shortcuts, equality/strict path behavior, unsupported depth,
Count dependence, deterministic generation, split conflicts, prompt mapping,
and solver error/timeout quarantine. Independently enumerate small graph paths
to cross-check the existing shortest-path implementation.

Run the pinned Qwen tokenizer for both styles, retaining length measurements
even for over-budget rows. Enforce the current 8,192 context / 4,096 Stage-1 /
16 Stage-2 limits at base level: a failure in either style or any matched view
quarantines the whole base. Report pre-admission and admitted depth coverage.

Keep full verified evidence and both traces for inspected examples. Automated
checks are not a completed human/agent release audit.

Report attempted/admitted counts by requested depth, query and role; null-depth
reasons; literal golds, cardinalities and semantic reasons; token distributions;
structural identities and degree profiles; rejected attempts; and a
training-role-selected contract-majority baseline on provisional evaluation
roles. These are pilot diagnostics, not final test results.

## Stop Condition

After this schedule and regression checks, report feasibility and proposed
production settings, including unresolved limitations. Do not create a training
export, final held-out benchmark, or seal. A failed depth or token category is
a result requiring further work, not permission to alter this run's cap.
