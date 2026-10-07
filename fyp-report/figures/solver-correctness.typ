#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "diagram-style.typ": diagram-ink, diagram-muted, diagram-border, diagram-panel, input-fill, contract-fill, core-fill, analysis-fill, answer-fill, explanation-fill

#let semantic-correspondence = figure(
  align(center)[
    #diagram(
      spacing: (28mm, 8mm),
      edge-stroke: .65pt + diagram-border,
      label-size: 7.5pt,
      label-sep: 3pt,
      mark-scale: 65%,

      node((0, 0), diagram-panel(
        [Qualitative world $W$],
        [Finite locations $L$ \ $p: L arrow.r RR^2$ \ No co-location \ Eight sign-pair relations],
        width: 50mm,
        fill: input-fill,
      ), name: <world>),
      node((1, 0), diagram-panel(
        [Integer model $M$],
        [$x_a, y_a in ZZ$ \ $x_a != x_b$ or $y_a != y_b$ \ Atomic X/Y comparisons],
        width: 50mm,
        fill: contract-fill,
      ), name: <integer-model>),
      edge(<world.east>, <integer-model.west>, "->", [rank compression]),

      node((0, 1), diagram-panel(
        [Source semantics],
        [$W |= phi$],
        width: 50mm,
        fill: input-fill,
      ), name: <source-semantics>),
      node((1, 1), diagram-panel(
        [Encoded semantics],
        [$M |= E(phi)$],
        width: 50mm,
        fill: contract-fill,
      ), name: <encoded-semantics>),
      edge(<source-semantics.east>, <encoded-semantics.west>, "<->", [truth preservation]),

      node((0, 2), diagram-panel(
        [Model-theoretic query],
        [Possible$(P,A)$ \ Entailed$(P,A)$],
        width: 50mm,
        fill: answer-fill,
      ), name: <query-semantics>),
      node((1, 2), diagram-panel(
        [SMT decision],
        [SAT$(E(P) and E(A))$ \ UNSAT$(E(P) and not E(A))$],
        width: 50mm,
        fill: core-fill,
      ), name: <decision>),
      edge(<query-semantics.east>, <decision.west>, "<->", [exact reduction]),
    )
  ],
  caption: [Truth-preserving correspondence between qualitative spatial worlds
    and integer constraint models, modulo order-preserving coordinate
    transformations.],
)

#let assurance-case = figure(
  align(center)[
    #diagram(
      spacing: (12mm, 12mm),
      edge-stroke: .65pt + diagram-border,
      label-size: 7.5pt,
      label-sep: 3pt,
      mark-scale: 65%,

      node((0, 0), diagram-panel(
        [Deductive obligations],
        [Finite representation equivalence \ Truth preservation \ Encoding and query correctness \ Propagation safety \ Proof and model certificate soundness],
        width: 52mm,
        fill: contract-fill,
      ), name: <deduction>),
      node((1, 0), diagram-panel(
        [Implementation evidence],
        [Bounded world enumeration \ Reference--Z3 comparisons \ Witness revalidation \ Parser round trips \ Proof mutation and branch-isolation tests \ Fail-closed errors],
        width: 52mm,
        fill: explanation-fill,
      ), name: <evidence>),

      node((0, 1), diagram-panel(
        [Semantic guarantees],
        [Encoding sound and complete \ Proof and model certificates sound],
        width: 52mm,
        fill: core-fill,
      ), name: <encoding-theorem>),
      node((1, 1), diagram-panel(
        [Tested conformance],
        [Confidence in Python \ implementation on covered cases],
        width: 52mm,
        fill: analysis-fill,
      ), name: <conformance>),
      edge(<deduction.south>, <encoding-theorem.north>, "->", [deduction]),
      edge(<evidence.south>, <conformance.north>, "->", [testing]),

      node((0.5, 2), diagram-panel(
        [Qualified claim],
        [Solver-backed analysis and checked proofs \ within the supported domain \ Z3 remains a trusted decision procedure],
        width: 65mm,
        fill: answer-fill,
      ), name: <qualified>),
      edge(<encoding-theorem.south-east>, <qualified.north-west>, "->", [guarantee]),
      edge(<conformance.south-west>, <qualified.north-east>, "->", [evidence]),
    )
  ],
  caption: [Separate roles of mathematical proof and implementation evidence in
    the solver-correctness claim. Passing tests supports conformance but does
    not formally verify the Python implementation.],
)

#let map-cell(label: none, query: false) = box(
  width: 7.5mm,
  height: 7.5mm,
  stroke: .4pt + diagram-border,
  fill: if query { answer-fill } else { white },
  align(center + horizon)[
    #if label != none {
      text(
        size: 7.5pt,
        weight: if query { "bold" } else { "regular" },
        fill: diagram-ink,
        label,
      )
    }
  ],
)

#let witness-panel(title, body) = block(
  inset: 8pt,
  radius: 4pt,
  stroke: .7pt + diagram-border,
  fill: input-fill,
  align(center)[
    #text(size: 9.5pt, weight: "bold", fill: diagram-ink)[#title]
    #v(3pt)
    #text(size: 8pt, fill: diagram-muted)[North $arrow.t$ #h(1em) East $arrow.r$]
    #v(5pt)
    #body
  ],
)

#let witness-map-one = table(
  columns: 6,
  rows: 6,
  gutter: 0pt,
  inset: 0pt,
  stroke: none,
  // y = 5
  map-cell(), map-cell(label: [R], query: true), map-cell(), map-cell(), map-cell(), map-cell(),
  // y = 4
  map-cell(), map-cell(), map-cell(), map-cell(label: [A]), map-cell(), map-cell(),
  // y = 3
  map-cell(), map-cell(), map-cell(label: [U]), map-cell(), map-cell(), map-cell(),
  // y = 2
  map-cell(), map-cell(), map-cell(), map-cell(), map-cell(label: [S]), map-cell(),
  // y = 1
  map-cell(), map-cell(), map-cell(label: [T]), map-cell(), map-cell(), map-cell(),
  // y = 0
  map-cell(label: [N], query: true), map-cell(), map-cell(), map-cell(), map-cell(), map-cell(),
)

#let witness-map-two = table(
  columns: 6,
  rows: 6,
  gutter: 0pt,
  inset: 0pt,
  stroke: none,
  // y = 5
  map-cell(label: [R], query: true), map-cell(), map-cell(), map-cell(), map-cell(), map-cell(),
  // y = 4
  map-cell(), map-cell(), map-cell(), map-cell(), map-cell(label: [A]), map-cell(),
  // y = 3
  map-cell(), map-cell(), map-cell(), map-cell(label: [U]), map-cell(), map-cell(),
  // y = 2
  map-cell(), map-cell(), map-cell(), map-cell(), map-cell(), map-cell(label: [S]),
  // y = 1
  map-cell(), map-cell(), map-cell(label: [T]), map-cell(), map-cell(), map-cell(),
  // y = 0
  map-cell(), map-cell(label: [N], query: true), map-cell(), map-cell(), map-cell(), map-cell(),
)

#let paired-countermodels = figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8mm,
    align(center)[
      #witness-panel([Witness 1: Northeast], witness-map-one)
    ],
    align(center)[
      #witness-panel([Witness 2: Northwest], witness-map-two)
    ],
  ),
  kind: image,
  caption: [Two coordinate models for SpatialEval item
    `spatialmap.tqa.2003.0`. Both satisfy the released premises, but the queried
    Recycle Center (R) lies northeast or northwest of Nightingale Novelties (N).
    A = Andy's Autos, S = Sally's Salon, T = Trail Hiking Gear, and U = Unicorn
    Umbrellas.],
)
