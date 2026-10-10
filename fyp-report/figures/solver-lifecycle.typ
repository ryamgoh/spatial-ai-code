#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "diagram-style.typ": diagram-border, diagram-panel, input-fill, contract-fill, core-fill, analysis-fill, answer-fill, explanation-fill, audit-fill

#let system-lifecycle = figure(
  align(center)[
    #diagram(
      spacing: (5mm, 5mm),
      edge-stroke: .65pt + diagram-border,
      label-size: 7.5pt,
      label-sep: 3pt,
      mark-scale: 65%,

      node((0, 0), diagram-panel(
        [Dataset input],
        [Released row \ reader adapters],
      ), name: <dataset>),
      node((2, 0), diagram-panel(
        [Synthetic input],
        [Policy settings \ problem builder],
      ), name: <synthetic>),

      node((1, 1), diagram-panel(
        [Shared problem contract],
        [locations \ propositional premises \ typed query],
        width: 50mm,
        fill: contract-fill,
      ), name: <contract>),
      edge(<dataset.south>, <contract.north-west>, "->", [parse and validate]),
      edge(<synthetic.south>, <contract.north-east>, "->", [construct]),

      node((1, 2), diagram-panel(
        [Data-agnostic reasoning core],
        [consistency and possibility \ entailment and witnesses],
        width: 50mm,
        fill: core-fill,
      ), name: <core>),
      edge(<contract.south>, <core.north>, "->"),

      node((1, 3), diagram-panel(
        [Semantic analysis],
        [possible directions, entities, or counts \ consistency and witness data],
        width: 50mm,
        fill: analysis-fill,
      ), name: <analysis>),
      edge(<core.south>, <analysis.north>, "->"),

      node((0, 4), diagram-panel(
        [Answer policy],
        [Apply answer mode \ map semantic values \ to menu letters],
        fill: answer-fill,
      ), name: <policy>),
      node((1, 4), diagram-panel(
        [Certificate layer],
        [Build and replay evidence \ render Natural or \ Symbolic traces],
        fill: explanation-fill,
      ), name: <certificate>),
      node((2, 4), diagram-panel(
        [Audit layer],
        [Validate witnesses \ record alternatives \ and provenance],
        fill: audit-fill,
      ), name: <audit>),
      edge(<analysis.south-west>, <policy.north>, "->"),
      edge(<analysis.south>, <certificate.north>, "->"),
      edge(<analysis.south-east>, <audit.north>, "->"),

      node((0, 5), diagram-panel(
        [Answer output],
        [Resolved values \ option letters \ response scores],
        fill: answer-fill,
      ), name: <answer>),
      node((1, 5), diagram-panel(
        [Training output],
        [Prompt and answer decision \ Natural or Symbolic trace],
        fill: explanation-fill,
      ), name: <training>),
      node((2, 5), diagram-panel(
        [Audit artifact],
        [Status and alternatives \ witnesses and provenance],
        fill: audit-fill,
      ), name: <artifact>),
      edge(<policy.south>, <answer.north>, "->"),
      edge(<policy.south-east>, <training.north-west>, "->"),
      edge(<certificate.south>, <training.north>, "->"),
      edge(<audit.south>, <artifact.north>, "->"),
    )
  ],
  caption: [End-to-end lifecycle and module boundaries. Dataset-specific input
    handling ends at `SpatialProblem`; menu policy, rendering, and auditing begin
    only after the data-agnostic solver returns a semantic analysis.],
)

#let solver-lifecycle = figure(
  align(center)[
    #diagram(
      spacing: (8mm, 6mm),
      edge-stroke: .65pt + diagram-border,
      label-size: 7.5pt,
      label-sep: 3pt,
      mark-scale: 65%,

      node((0, 0), diagram-panel(
        [Step 1 — Validate the problem],
        [Check locations and formulas \ query targets and candidate domains],
        width: 52mm,
        fill: input-fill,
      ), name: <validate>),
      node((1, 0), diagram-panel(
        [Step 2 — Propagate required facts],
        [Use only positive facts that must hold \ stop early if they contradict],
        width: 52mm,
        fill: contract-fill,
      ), name: <propagate>),
      node((1, 1), diagram-panel(
        [Step 3 — Prepare exact constraints],
        [Create symbolic X/Y variables \ add no-co-location, propagation, and full premises],
        width: 58mm,
        fill: contract-fill,
      ), name: <prepare>),
      node((0, 1), diagram-panel(
        [Step 4 — Check consistency],
        [No satisfying assignment: inconsistent \ assignment exists: continue],
        width: 52mm,
        fill: analysis-fill,
      ), name: <consistent>),
      node((0, 2), diagram-panel(
        [Step 5 — Test query candidates],
        [Direction: each relation \ Selection: each entity \ Count: each count],
        width: 52mm,
        fill: core-fill,
      ), name: <candidates>),
      node((1, 2), diagram-panel(
        [Step 6 — Interpret each check],
        [Model found: possible and save one witness \ no model: impossible or, for a negated claim, entailed],
        width: 58mm,
        fill: explanation-fill,
      ), name: <interpret>),
      node((1, 3), diagram-panel(
        [Step 7 — Return typed analysis],
        [Consistency and answer sets \ normalised witnesses],
        width: 52mm,
        fill: answer-fill,
      ), name: <result>),

      edge(<validate.east>, <propagate.west>, "->"),
      edge(<propagate.south>, <prepare.north>, "->"),
      edge(<prepare.west>, <consistent.east>, "->"),
      edge(<consistent.south>, <candidates.north>, "->"),
      edge(<candidates.east>, <interpret.west>, "->"),
      edge(<interpret.south>, <result.north>, "->"),
    )
  ],
  caption: [Lifecycle inside the reasoning core. Coordinates begin as symbolic
    variables and receive concrete values only when the backend finds a
    satisfying model. Different candidate checks may therefore return different
    witness maps.],
)
