#import "@preview/touying:0.6.1": *
#import themes.simple: *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let ink = rgb("#111111")
#let muted = rgb("#555555")
#let hairline = rgb("#B8B8B8")
#let wash = rgb("#F3F3F3")

#let fs = (
  slide: 27pt,
  display: 24pt,
  lead: 17pt,
  card: 15pt,
  section: 13pt,
  body: 11.5pt,
  caption: 10.5pt,
  label: 9.5pt,
)

#let generated-traces = json("docs/brief-traces.json")
#let trace-text(value) = text(
  size: fs.body,
  value.split("\n").map(line => text(line)).join(linebreak()),
)

#let panel(body, fill: white, stroke: hairline, inset: 12pt) = block(
  width: 100%,
  fill: fill,
  stroke: stroke,
  inset: inset,
  body,
)

#let label(body) = text(size: fs.label, weight: "bold", tracking: 0.6pt, fill: muted, body)

#let step(number, title, body) = block(
  width: 100%,
  stroke: (top: 1.2pt + ink),
  inset: (top: 10pt, right: 4pt),
  [
    #grid(
      columns: (27pt, 1fr),
      gutter: 5pt,
      align: (left, top),
      text(size: fs.body, weight: "bold", fill: muted, "0" + str(number)),
      [#text(size: fs.section, weight: "bold", fill: ink)[#title] #linebreak()
       #text(size: fs.body, fill: muted)[#body]],
    )
  ],
)

#let answer-chip(value, active: false) = box(
  width: 23pt,
  height: 23pt,
  radius: 999pt,
  fill: if active { ink } else { white },
  stroke: if active { ink } else { hairline },
  align(center + horizon, text(
    size: fs.label,
    weight: "bold",
    fill: if active { white } else { muted },
    value,
  )),
)

#let g-card(g) = block(
  width: 100%,
  stroke: (top: 1.2pt + ink),
  inset: (top: 11pt),
  [
    #align(center)[
      #text(size: fs.display, weight: "bold", fill: ink)[G#g]
      #linebreak()
      #text(size: fs.caption, fill: muted)[#g possible #if g == 1 {[answer]} else {[answers]}]
      #v(8pt)
      #grid(
        columns: (23pt,) * 5,
        gutter: 4pt,
        ..range(5).map(i => answer-chip(("A", "B", "C", "D", "E").at(i), active: i < g)),
      )
    ]
  ],
)

#let spatial-case(northwest: false, spacing: (13mm, 11mm), radius: 9pt) = diagram(
  spacing: spacing,
  node-stroke: .9pt + ink,
  node-fill: white,
  edge-stroke: 1pt + muted,
  label-size: fs.label,
  node((if northwest { 0 } else { 1 }, 0), [R], radius: radius, name: <recycle>),
  node((2, 1), [S], radius: radius, name: <sally>),
  node((if northwest { 1 } else { 0 }, 2), [N], radius: radius, name: <nightingale>),
  edge(<recycle.south-east>, <sally.north-west>, "--"),
  edge(<sally.south-west>, <nightingale.north-east>, "--"),
)

#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [SpatialEntail · professor brief],
  config-common(
    slide-level: 2,
    new-section-slide-fn: none,
  ),
  config-colors(primary: ink),
  config-page(
    fill: white,
    margin: (x: 30pt, top: 24pt, bottom: 27pt),
  ),
)

#set text(font: "Arial", size: fs.body, fill: ink)
#set par(leading: 0.78em)
#show heading.where(level: 2): it => block(
  below: 13pt,
  [
    #text(size: fs.slide, weight: "bold", fill: ink)[#it.body]
  ],
)

== 1. The problem: one text can allow several answers

#grid(
  columns: (1.08fr, 0.92fr),
  gutter: 14pt,
  panel[
    #label([RELEASED EXAMPLE · spatialmap.tqa.2003.0])
    #v(9pt)
    #text(size: fs.section, weight: "bold", fill: ink)[Two facts from the released question]
    #v(5pt)
    #text(size: fs.body, fill: muted)[
      1. Sally's Salon is southeast of Recycle Center. #linebreak()
      2. Nightingale Novelties is southwest of Sally's Salon.
    ]
    #v(9pt)
    #text(size: fs.section, weight: "bold", fill: ink)[Question]
    #v(4pt)
    #text(size: fs.body)[Where is Recycle Center relative to Nightingale Novelties?]
    #v(6pt)
    #text(size: fs.caption, fill: muted)[A Southeast · B Southwest · C Northeast · D Northwest]
    #v(9pt)
    #panel(fill: wash, stroke: ink, inset: 9pt)[
      #text(size: fs.body, weight: "bold")[Published label: C · Northeast]
    ]
    #v(12pt)
    #text(size: fs.body, fill: muted)[*833 / 1,500* released questions had no unique answer from their text.]
  ],
  panel[
    #text(size: fs.section, weight: "bold", fill: ink)[What follows from the text]
    #v(6pt)
    #text(size: fs.body, fill: muted)[Recycle Center must be *above* Nightingale Novelties, but it may lie on either side horizontally.]
    #v(4pt)
    #text(size: fs.label, fill: muted)[R = Recycle Center · S = Sally's Salon · N = Nightingale Novelties]
    #v(10pt)
    #grid(
      columns: (1fr, 1fr),
      gutter: 7pt,
      panel(stroke: ink, inset: 9pt)[
        #align(center)[
          #text(size: fs.caption, weight: "bold")[LEGAL CASE 1]
          #v(5pt)
          #spatial-case()
          #v(5pt)
          #text(size: fs.caption, weight: "bold")[C · Northeast]
        ]
      ],
      panel(stroke: ink, inset: 9pt)[
        #align(center)[
          #text(size: fs.caption, weight: "bold")[LEGAL CASE 2]
          #v(5pt)
          #spatial-case(northwest: true)
          #v(5pt)
          #text(size: fs.caption, weight: "bold")[D · Northwest]
        ]
      ],
    )
    #v(9pt)
    #panel(fill: wash, stroke: ink, inset: 9pt)[
      #align(center)[#text(size: fs.body, weight: "bold")[Possible displayed options: C and D]]
    ]
  ],
)

#speaker-note[
  Say: “These are two relevant facts excerpted from the released question, not its entire premise set. The released text forces Recycle Center above Nightingale Novelties but does not fix horizontal order. C and D are the possible displayed directional options under an all-possible instruction. Under SINGLE, an ambiguity fallback would instead be selected if available; the original four-option menu has no such fallback. Exact North may also fit, but it is absent from that menu. The published label comes from one hidden source map. This was not rare: 833 of 1,500 released questions were underdetermined.”

  Source: `data/spatialeval_org.jsonl`, item `spatialmap.tqa.2003.0`; `fyp-report/chapters/03-problem-formulation.typ`.
]

== 2. Why call this spatial entailment?

#align(center)[
  #text(size: fs.lead, weight: "bold", fill: ink)[Does a spatial claim follow from the words alone?]
]

#v(12pt)

#grid(
  columns: (0.93fr, 1.07fr),
  gutter: 13pt,
  panel(stroke: ink, inset: 12pt)[
    #label([NOTATION])
    #v(9pt)
    #text(size: fs.section)[*P* = the stated spatial facts]
    #v(5pt)
    #text(size: fs.section)[*S* = the candidate statement]
    #v(12pt)
    #align(center)[#text(size: fs.display, weight: "bold")[$P tack.rr S$]]
    #v(7pt)
    #text(size: fs.body, fill: muted)[Say: “P entails S.” It means *every* arrangement consistent with P also makes S true.]
  ],
  panel(stroke: ink, inset: 12pt)[
    #label([EXAMPLE])
    #v(9pt)
    #text(size: fs.body)[
      *Two facts from P:* Sally's Salon is southeast of Recycle Center; Nightingale Novelties is southwest of Sally's Salon.
    ]
    #v(10pt)
    #text(size: fs.body)[
      *S:* Recycle Center is *above* Nightingale Novelties.
    ]
    #v(10pt)
    #text(size: fs.body, weight: "bold")[N is below Sally; Sally is below R. Therefore R is above N in every legal arrangement.]
  ],
)

#v(12pt)

#panel(fill: wash, stroke: ink, inset: 11pt)[
  #text(size: fs.section, weight: "bold")[But “possible” is weaker than “entailed.”]
  #v(5pt)
  #text(size: fs.body, fill: muted)[Northeast and Northwest each fit a legal arrangement. Neither exact direction is forced.]
]

#v(10pt)
#align(center)[#text(size: fs.body, weight: "bold")[One matching arrangement proves “could”; ruling out every counterexample proves “must.”]]

#speaker-note[
  Say: “The bar symbol is read ‘entails.’ P is only what the student can read; S is a proposed statement about positions. P entails S when there is no legal arrangement of the objects in which P holds and S fails. In our example the two downward steps force Recycle Center above Nightingale Novelties. But they leave horizontal order open, so Northeast and Northwest are each possible, and neither is entailed. ‘Above’ here means higher on the vertical axis, not the exact compass direction North; exact North may also fit, but it is not in the displayed answer menu. That is why this is spatial entailment: we check which spatial claims follow from visible facts, and which answers are only possible.”

  Source: `fyp-report/chapters/03-problem-formulation.typ`, sections “Directions as Axis Comparisons” and “Possibility, Entailment, and Ambiguity.”
]

== 3. Our spatial representation

#text(size: fs.lead, weight: "bold")[A and B name objects. “A is East of B” is a spatial atom.]
#v(13pt)
#table(
  columns: (0.65fr, 1.15fr, 1.2fr),
  inset: (x: 10pt, y: 11pt),
  stroke: (bottom: 0.6pt + hairline),
  table.header([*Building block*], [*Meaning in our system*], [*Example*]),
  [*Object*],
  [A named thing or place, represented by a 2D position.],
  [A, B, and C are object names, not atoms.],
  [*Spatial relation*],
  [How one object is positioned relative to another.],
  [“East of” means farther right, at the same height.],
  [*Spatial atom*],
  [One complete spatial claim about two objects.],
  [“A is East of B.” #linebreak() Two objects + one spatial relation form one atom.],
  [*Compound formula*],
  [Claims combined or negated using logical operators.],
  [“A is East of B” *and* “B is North of C.”],
)
#v(13pt)
#panel(fill: wash, stroke: ink, inset: 11pt)[
  #text(size: fs.body)[*Eight exact-direction types:* North, Northeast, East, Southeast, South, Southwest, West, Northwest.]
  #v(7pt)
  #text(size: fs.body)[*Logical operators:* and, or, not, if–then, if and only if. “Atomic” means the claim has no such operator at this level.]
]
#v(8pt)
#text(size: fs.caption, fill: muted)[A relation constraint can also allow several directions: “above” permits North, Northeast, or Northwest.]

#speaker-note[
  Say: “Start with objects. A and B are names for things or places, and each object has an integer X and Y position in a possible arrangement. Distinct objects cannot share a point. A relation describes their relative position. Applying that relation to two objects makes one spatial atom: A is East of B. It is not made from two atoms; it is a claim about two objects. Every spatial atom in our representation has a subject and a reference object. Atomic refers to the formula structure, not to the absence of objects or a relation inside the claim. Our exact-direction vocabulary has eight types, but changing the object pair produces different atoms. The implementation's RelationConstraint can also allow a set of exact directions, so above can be stored as North, Northeast, or Northwest without a Boolean Or node. Not, And, Or, Implies, and Iff then build compound formulas from spatial claims or other formulas. This is our chosen representation, not a universal definition of atoms in logic.”

  Source: `spatial/v2/solver.py`, `_DIRECTION_SIGNS`, `RelationConstraint`, `Not`, `And`, `Or`, `Implies`, and `Iff`.
]

== 4. Five reasoning rules used by the generator

#text(size: fs.body)[A, B, and C name *objects*. Each row is a separate example. Read the arrow as *“therefore.”*]
#v(10pt)
#block[
  #let rule-row(title, formal, premises) = grid(
    columns: (180pt, 1fr),
    gutter: 12pt,
    align: left + horizon,
    [
      #text(size: fs.body, weight: "bold", title)
      #linebreak()
      #text(size: fs.label, fill: muted, formal)
    ],
    diagram(
      spacing: 18pt,
      node-stroke: 0.6pt + hairline,
      node-inset: 7pt,
      edge-stroke: 1pt + ink,
      node((0, 0), text(size: fs.body, premises), width: 285pt, name: <premises>),
      node((1, 0), text(size: fs.body, weight: "bold")[A is East of C.],
        width: 130pt, fill: wash, name: <conclusion>),
      edge(<premises.east>, <conclusion.west>, "-|>"),
    ),
  )

  #rule-row([Apply an if–then rule], [Modus ponens · MP])[
    A is East of B. #linebreak()
    If A is East of B, then A is East of C.
  ]
  #v(7pt)
  #rule-row([Use an equivalence], [Equivalence elimination · IFF])[
    A is East of B. #linebreak()
    A is East of B if and only if A is East of C.
  ]
  #v(7pt)
  #rule-row([Eliminate an alternative], [Disjunctive syllogism · DJE])[
    A is East of B or A is East of C. #linebreak()
    A is not East of B.
  ]
  #v(7pt)
  #rule-row([Remove a double negation], [Double negation · DN])[
    It is not the case that A is not East of C.
  ]
  #v(7pt)
  #rule-row([Chain spatial comparisons], [Axis transitivity · TRC])[
    A is East of B. #linebreak()
    B is East of C.
  ]
]
#v(11pt)
#text(size: fs.body)[*To generate a question, work backwards:* choose a conclusion, then construct premises using these rules.]
#v(5pt)
#text(size: fs.caption, fill: muted)[Sample a complete allowed rule sequence uniformly, not each step independently. See the sampling appendix. These diagrams are teaching examples.]

#speaker-note[
  Say: “Read each row independently: the statements on the left justify the conclusion on the right. A, B, and C are object names throughout. Modus ponens applies an implication whose condition holds. Equivalence elimination uses one direction of an if-and-only-if statement; it also works in the reverse direction. Disjunctive syllogism uses an inclusive or plus the negation of one alternative to establish the other. Double negation removes two negations in our classical semantics. Axis transitivity chains comparisons through B; the East example chains increasing X positions and equal Y positions. Do not assume arbitrary mixed compass directions compose to one exact direction. The identifiers MP, IFF, DJE, DN, and TRC are the actual sampled codes; TRC maps to axis-transitivity in the proof labels. These are teaching examples of the production backward-step transformations, not renderer outputs. The generator samples a rule path and constructs its supporting premises backwards from a target. For ambiguous problems, a generated derivation can replace a required premise in a possibility skeleton. Fixed templates also support case splits, and Count evidence preserves joint assignments. The sampled path is linear, but complete evidence can branch. The five sampled rules are narrower than the full checker vocabulary, and a checked dependency path need not be the globally shortest proof.”

  Source: `spatial/v2/solver.py`, formula AST; `spatial/v2/random_proof.py`, rule alphabet; `spatial/v2/random_proof_generation.py`, backward transformations and premise lifting; `spatial/v2/proofs.py`, checker vocabulary.
]

== 5. Turn the structure into a checked question

#text(size: fs.lead, weight: "bold")[A question contains premises, a query, an answer instruction, and options.]
#v(10pt)
#grid(
  columns: (1.2fr, 0.8fr),
  gutter: 12pt,
  panel(stroke: ink, inset: 10pt)[
    #label([DIRECTION QUESTION · TEACHING EXAMPLE])
    #v(8pt)
    #text(size: fs.body)[*Premises:* A is East of B. B is East of C.]
    #v(8pt)
    #text(size: fs.body)[*Query:* In which direction is A relative to C?]
    #v(8pt)
    #text(size: fs.body)[*Instruction:* Select exactly one answer.]
    #v(8pt)
    #text(size: fs.body)[*Options:* A. North; B. Northeast; C. East; D. Southeast; E. South.]
    #v(8pt)
    #text(size: fs.body, weight: "bold")[Checked answer: C. East.]
    #v(5pt)
    #text(size: fs.caption, fill: muted)[Five options, matching the study format. Teaching example, not a quoted dataset row.]
  ],
  panel(stroke: ink, inset: 10pt)[
    #label([THREE QUERY TYPES])
    #v(8pt)
    #text(size: fs.body)[*Direction:* Where is A relative to C?]
    #v(8pt)
    #text(size: fs.body)[*Selection:* Which object is East of C?]
    #v(8pt)
    #text(size: fs.body)[*Count:* How many objects are East of C?]
    #v(8pt)
    #text(size: fs.caption, fill: muted)[Choose the query type during construction. Define which objects or directions are candidates.]
  ],
)
#v(10pt)
#grid(
  columns: (1fr,) * 3,
  gutter: 9pt,
  step(1, [Solve], [Find possible answers. Check the proofs, maps, and requested L, G, and X.]),
  step(2, [Write English], [Render the premises, query, instruction, and shuffled menu using templates.]),
  step(3, [Read it back], [Parse the text and solve again. Reject changes to the problem, options, or answers.]),
)
#v(7pt)
#text(size: fs.caption, fill: muted)[Use only stated facts, finite distinct objects in 2D, and the declared candidates. Distinct objects may share one axis.]

#speaker-note[
  Say: “The generator chooses a query family during construction, not after writing arbitrary prose. A structured problem stores objects, premises, and a query. Answer policy and menu supply the instruction and options. The teaching example continues the East-to-East transitivity row on slide 4; the answer is East, mapped to C in this illustrative five-option menu. The randomized study sets menu_option_count to five. For resolved SINGLE G1, the menu contains the correct direction and four impossible directions, without a Cannot be determined fallback. Full coverage means all possible answers are listed, not all eight direction types. This is not a saved generated row or trace. Direction asks for a relative direction, Selection for an object, and Count for a number; Selection and Count specify candidate objects and a relation to the reference. The solver determines the semantic answers independently of menu letters. Proof and model certificates are checked, and the realized L/G/X must match the requested controls. Fixed templates render the prompt. Parsing must recover the same structured problem and options, and solving again must recover the same semantic answer and selected letters. Inconsistent, unsupported, or changed examples are rejected. Query type and answer policy are separate: SINGLE requests one invariant answer, while ALL_POSSIBLE requests every possible value. The BENCH/SFT packaging distinction comes later. All reasoning remains conditional on visible premises, finite distinct objects in 2D, our relation vocabulary, and declared candidates.”

  Source: `spatial/v2/generation.py`, `_menu`, `_prompt`, and `_verify_round_trip`; `spatial/v2/study/randomized_v2.py`, `_randomized_policy`; `spatial/v2/solver.py`, query types.
]

== 6. Answer-set size is separate from difficulty

#align(center)[
  #text(size: fs.lead, weight: "bold")[G counts possible answers; L and X describe the reasoning task.]
]

#v(16pt)

#grid(
  columns: (1fr,) * 4,
  gutter: 9pt,
  g-card(1),
  g-card(2),
  g-card(3),
  g-card(4),
)

#v(18pt)

#grid(
  columns: (1fr,) * 3,
  gutter: 10pt,
  panel(stroke: ink, inset: 11pt)[
    #label([L · DEPENDENCY LENGTH])
    #v(7pt)
    #text(size: fs.body)[Longest dependency chain, counting semantic reasoning rules.]
    #v(5pt)
    #text(size: fs.caption, fill: muted)[Train L1–L6; test longer L7–L8.]
  ],
  panel(stroke: ink, inset: 11pt)[
    #label([X · DISTRACTORS])
    #v(7pt)
    #text(size: fs.body)[Facts removable together without changing the possible answers.]
    #v(5pt)
    #text(size: fs.caption, fill: muted)[Train X0–X6; test noisier X7–X8.]
  ],
  panel(stroke: ink, inset: 11pt)[
    #label([NEW RULE COMBINATIONS])
    #v(7pt)
    #text(size: fs.body)[One rule's conclusion feeds into the next rule.]
    #v(5pt)
    #text(size: fs.caption, fill: muted)[Test an order excluded from training. Each test path has exactly one of six held-out pairs.]
  ],
)

#v(13pt)

#panel(fill: wash, stroke: ink, inset: 10pt)[
  #align(center)[
    #text(size: fs.body, weight: "bold")[Under MULTI, G2 means select both possible answers. Missing either is wrong.]
    #v(4pt)
    #text(size: fs.caption, fill: muted)[Chips illustrate full-menu MULTI answers, not SINGLE fallback decisions. G is not the number of maps or menu choices.]
  ]
]

#speaker-note[
  Say: “G is the number of solver-possible semantic values: directions, entities, or counts. Possible counts one and three give G2. Full-menu MULTI selects G letters, but an ambiguous SINGLE question can have G greater than one and still select just the Cannot be determined fallback. The headline SINGLE training contract uses resolved G1 only. L counts semantic rule applications along the longest checked dependency path, not all low-level proof lines and not the globally shortest proof. X is measured by a greedy, support-preserving removal procedure, so it is not a globally maximum removable set. Requested L, G, and X must match measured values. Of 23 supported ordered rule transitions, 17 are seen and six held out; each composition-OOD path contains exactly one held-out transition. Each stress partition changes one factor at a time.”

  Source: `spatial/v2/proof_complexity.py`; `spatial/v2/difficulty.py`; `spatial/v2/grading.py`; `spatial/v2/random_proof.py`; `spatial/v2/study/randomized_v2.py`.
]

== 7. How we allocate examples

#align(center)[
  #text(size: fs.lead, weight: "bold")[Favor longer proofs and more distractors, while retaining the easier levels.]
]
#v(10pt)
#grid(
  columns: (1.05fr, 0.95fr),
  gutter: 12pt,
  panel(stroke: ink)[
    #label([EXPONENTIAL WEIGHTS · SEPARATE L AND X QUOTAS])
    #v(10pt)
    #align(center)[#text(size: fs.lead)[$p(k_i) = r^i / (sum_(j=0)^(m-1) r^j)$]]
    #v(9pt)
    #text(size: fs.body)[
      Order the m allowed levels from low to high; i starts at zero.
      Each next level gets *r times the previous weight*.
    ]
    #v(8pt)
    #text(size: fs.body)[Multiply by the row budget; round to exact integer quotas. Pair the L and X allocations with seeded shuffling.]
    #v(7pt)
    #text(size: fs.caption, fill: muted)[Dataset counts, not loss weights. Sample rule sequences uniformly at the required prefix length; see the appendix.]
  ],
  panel(stroke: ink)[
    #label([CONFIGURED RATIOS])
    #v(10pt)
    #grid(
      columns: (1.3fr, 1fr, 1fr),
      gutter: 9pt,
      [*Allocation*], [*L ratio*], [*X ratio*],
      [BENCH], [1.15], [1.10],
      [SFT train], [1.40], [1.25],
      [SFT dev], [1.00], [1.00],
    )
    #v(13pt)
    #text(size: fs.body)[*Example:* 2,000 MULTI G1 training rows over L1–L6:]
    #v(7pt)
    #text(size: fs.body, weight: "bold")[122 · 172 · 240 · 336 · 471 · 659]
    #v(6pt)
    #text(size: fs.caption, fill: muted)[G is balanced separately, not exponentially weighted.]
  ],
)
#v(9pt)
#panel(fill: wash, stroke: ink, inset: 11pt)[
  #label([HELD-OUT TESTS])
  #v(7pt)
  #text(size: fs.body)[
    *SFT:* L up to 6, X0–X6, seen rule transitions. #linebreak()
    *BENCH:* matched cases, longer L7–L8, noisier X7–X8, or a held-out rule pairing.
  ]
  #v(5pt)
    #text(size: fs.caption, fill: muted)[Test longer reasoning, more distractors, or new rule combinations separately. Minimum L depends on the query/G group.]
]

#speaker-note[
  Say: “This is the exponential weighting you remember. For an ordered list of allowed levels, the code gives index i weight r to the power i, normalizes, and turns the probabilities into exact row quotas using largest remainders. Benchmark ratios are 1.15 for L and 1.10 for X; training uses stronger tilts of 1.40 and 1.25. Development uses uniform conditional marginals. L and X lists are independently allocated and paired by seeded shuffling; this is not a full Cartesian grid or a loss-weighting schedule. G is balanced independently. The displayed 2,000-row example is the exact MULTI G1 L allocation. MULTI training uses L from G to six in each G group; SINGLE uses one to six. Benchmark Direction and Selection can start at L1 even for higher G, while ambiguous Count has higher minimum depth. Composition tests require at least L2, and ambiguous Count composition requires at least G plus two. Accepted rows must match the requested G, L, and X after checking. Feasible rule paths are sampled separately; their distribution is not this exponential.”

  Source: `spatial/v2/random_proof.py`, integer quota allocation; `spatial/v2/study/randomized_v2.py`, benchmark/training allocation and L/X ratio constants; `spatial/v2/generation.py`, realized-difficulty admission checks.
]

== 8. BENCH evaluates; SFT teaches

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  panel(stroke: ink)[
    #label([BENCH · HELD-OUT EVALUATION])
    #v(9pt)
    #text(size: fs.card, weight: "bold")[3,600 unique structures]
    #v(8pt)
    #text(size: fs.body)[
      900 SINGLE G1 + 2,700 ALL_POSSIBLE G2–G4. #linebreak()
      Direction, Selection, and Count: 1,200 each.
    ]
    #v(9pt)
    #text(size: fs.body)[
      *1,440* matched cases. #linebreak()
      *720 each:* length, noise, and composition stress tests.
    ]
    #v(8pt)
    #text(size: fs.caption, fill: muted)[Score model predictions against checked gold answers. These structures are excluded from SFT.]
  ],
  panel(stroke: ink)[
    #label([SFT · SUPERVISED FINE-TUNING])
    #v(9pt)
    #text(size: fs.card, weight: "bold")[8,000 train + 800 dev per package]
    #v(8pt)
    #text(size: fs.body)[
      *SINGLE:* resolved G1 questions only. #linebreak()
      *MULTI:* G1–G4 balanced, 2,000 train + 200 dev per G.
    ]
    #v(9pt)
    #text(size: fs.body)[
      MULTI shares 2,000 train + 200 dev G1 examples with SINGLE.
    ]
    #v(8pt)
    #text(size: fs.caption, fill: muted)[Training pairs prompts with assistant completions. Development is separate from training and BENCH.]
  ],
)
#v(13pt)
#panel(fill: wash, stroke: ink)[
  #label([ANSWER POLICIES])
  #v(8pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 15pt,
    [
      #text(size: fs.body)[*SINGLE:* ask for one invariant answer.]
      #v(5pt)
      #text(size: fs.caption)[If Northeast and Northwest are both possible, select “Cannot be determined.” Central SINGLE data exclude these ambiguous cases.]
    ],
    [
      #text(size: fs.body)[*ALL_POSSIBLE:* select every possible answer.]
      #v(5pt)
      #text(size: fs.caption)[For the same ambiguity, select Northeast and Northwest. The MULTI package combines SINGLE G1 with ALL_POSSIBLE G2–G4.]
    ],
  )
]
#v(9pt)
#text(size: fs.caption, fill: muted)[Recorded audits: no BENCH/SFT or train/dev structural overlap.]

#speaker-note[
  Say: “There are two separate distinctions. BENCH versus SFT is evaluation versus learning data. SINGLE versus ALL_POSSIBLE is the answer policy on a question. MULTI names a training package that mixes resolved SINGLE G1 rows with ALL_POSSIBLE G2 to G4 rows. BENCH has 3,600 unique structures: three query families, four G groups, and 300 examples per query/G group. Each such group has 120 matched and 60 for each of three stress tests. The SINGLE and MULTI SFT packages each contain 8,000 training and 800 development examples; MULTI's 2,000 training and 200 development G1 bases are shared verbatim with SINGLE. The SINGLE answer-only variant is derived from exactly the same SINGLE questions by dropping the Natural explanation. Counting shared rows only once gives 15,400 unique training/development bases. The recorded audits report no benchmark overlap and no training/development structural overlap. SINGLE semantics support an ambiguity fallback, but that is a separate diagnostic rather than part of central G1 training. We are leaving the visible-possible contract out of this presentation.”

  Source: `spatial/v2/study/randomized_v2.py`, benchmark allocation and training contract construction; `spatial/v2/grading.py`, answer resolution; `experiments/spatialentail-randomized-v2/benchmark-candidate-audit.json`; `experiments/spatialentail-randomized-v2/training-candidate-audit.json`.
]

== 9. Full Natural response: one direction

#label([G1 · CHECKED FIXTURE])
#v(9pt)

#panel(fill: wash, stroke: ink, inset: 11pt)[
  #text(size: fs.body)[
    *Question:* A is Northeast of B. Where is A relative to B? #linebreak()
    *Menu:* #generated-traces.g1.menu_text
  ]
]

#v(15pt)

#panel(stroke: ink, inset: 13pt)[
  #trace-text(generated-traces.g1.completion)
]

#v(13pt)
#text(size: fs.caption, fill: muted)[Loaded from saved renderer output: checked G1 fixture, reference solver backend. Not a historical training row.]

#speaker-note[
  Say: “This is the entire assistant completion produced by the production Natural renderer and answer-decision renderer for a small checked test fixture. Nothing inside the think block is omitted. The unique direction is proven directly from P1, the supporting model is summarized in relative X/Y order, and the answer policy maps Northeast to A. The menu shown above is the one used to render this response. It is a representative fixture, not a claim that this exact row was used in the six prepared runs.”

  Source: `spatial/v2/tests/test_answer_certificates.py`, `test_unique_answer_set_covers_all_eight_candidates`; `spatial/v2/answer_certificate_renderers.py`; `spatial/v2/generation.py`, `_render_answer_decision` and `GeneratedSpatialSample.as_sft_row`.
]

== 10. Full Natural response: two directions (1/2)

#label([G2 · CHECKED FIXTURE · REFERENCE SOLVER])
#v(8pt)

#panel(fill: wash, stroke: ink, inset: 10pt)[
  #text(size: fs.body)[
    *Question:* S is Southeast of R. N is Southwest of S. Where is R relative to N? #linebreak()
    *Menu:* #generated-traces.g2.menu_text #linebreak()
    *Policy:* Select every possible direction among NE, NW, SE, SW. South is an impossible menu distractor.
  ]
]

#v(12pt)
#panel(stroke: ink, inset: 11pt)[
  #trace-text(generated-traces.g2.parts.at(0))
]

#speaker-note[
  Say: “This and the next slide together contain the entire assistant completion generated by the Natural renderer for the checked three-object G2 fixture. It starts with a model supporting Northeast, then a rule-by-rule refutation of Southeast. Nothing has been elided. This example uses the displayed four-candidate query and a menu that maps Northeast and Northwest to C and D. The actual generator shuffles names and menu letters in other samples.”

  Source: `spatial/v2/tests/test_answer_certificates.py`, `test_ambiguous_ordinal_answer_set_has_two_models_and_two_refutations`; `spatial/v2/answer_certificate_renderers.py`; `spatial/v2/generation.py`, `_render_answer_decision`.
]

== 11. Full Natural response: two directions (2/2)

#label([G2 · CONTINUED VERBATIM FROM PREVIOUS SLIDE])
#v(11pt)

#panel(stroke: ink, inset: 12pt)[
  #trace-text(generated-traces.g2.parts.at(1))
]

#speaker-note[
  Say: “We continue from the previous slide without restarting the think block. The second refutation rules out Southwest, then a distinct model supports Northwest. The final decision requires all possible directions, and this example's menu maps them to C and D. This entire target is deterministic rendering of a checked certificate and menu decision, not generated by an LLM. We use a checked fixture because the prepared training files do not contain a saved Natural JSONL row in this workspace.”

  Source: `spatial/v2/tests/test_answer_certificates.py`, `test_ambiguous_ordinal_answer_set_has_two_models_and_two_refutations`; `spatial/v2/answer_certificate_renderers.py`; `spatial/v2/generation.py`, `_render_answer_decision`.
]

== 12. Training variants

#text(size: fs.lead, weight: "bold")[Qwen3.5-4B · QLoRA · seeds 42 and 43]

#v(15pt)

#table(
  columns: (0.8fr, 1.2fr, 1.2fr),
  inset: (x: 10pt, y: 12pt),
  stroke: (bottom: 0.6pt + hairline),
  table.header([*Variant*], [*Questions*], [*Assistant target*]),
  [1 · SINGLE], [G1 only], [Final answer only],
  [2 · SINGLE], [Same questions as variant 1], [Natural reasoning + answer],
  [3 · MULTI], [Balanced G1–G4], [Natural reasoning + answer set],
)

#v(15pt)

#panel(fill: wash, stroke: ink, inset: 12pt)[
  #text(size: fs.body)[*Data:* BENCH and SFT audited; complete training rows fit 4,096 tokens.]
  #v(7pt)
  #text(size: fs.body)[*Runs:* Six jobs prepared; completion evidence pending.]
  #v(7pt)
  #text(size: fs.body)[*Evaluation:* Exact answer sets and transfer to the corrected external benchmark.]
]

#v(18pt)
#align(center)[
  #text(size: fs.lead, weight: "bold")[Does checked Natural supervision help the model find *all* valid answers?]
]

#speaker-note[
  Say: “We compare three variants of the same 4-billion-parameter starting model. The first two train on the same G1 questions, changing only whether the target includes a checked Natural explanation. The third uses Natural explanations on G1 through G4, teaching complete answer sets. We use QLoRA for two epochs and repeat each variant with seeds 42 and 43; complete training rows fit the 4,096-token budget. The dataset and token audits passed. Six job configurations are prepared. The checklist marks running or awaiting artifacts, but job IDs and completion receipts are not recorded, so we do not establish live execution status. We will score exact answer sets, test the controlled difficulty changes described on slides 6–7, and measure transfer to a corrected external benchmark.”

  Source: `fyp-report/chapters/05-experimental-design.typ`; `experiments/spatialentail-randomized-v2/README.md`; `experiments/spatialentail-randomized-v2/CHECKLIST.md`; `fyp-report/chapters/06-results-discussion.typ`.
]

== Appendix · How Z3 finds an arrangement

#align(center)[
  #text(size: fs.lead, weight: "bold")[Write spatial rules, add a proposed answer, then ask whether any coordinates fit.]
]

#v(13pt)

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  panel(stroke: ink, inset: 13pt)[
    #label([1 · TRANSLATE THE FACTS])
    #v(10pt)
    #text(size: fs.body)[Give each object X (left/right) and Y (down/up) positions.]
    #v(10pt)
    #text(size: fs.body, weight: "bold")[S is southeast of R]
    #v(3pt)
    #text(size: fs.body, fill: muted)[$x_S > x_R$ and $y_S < y_R$]
    #v(10pt)
    #text(size: fs.body, weight: "bold")[N is southwest of S]
    #v(3pt)
    #text(size: fs.body, fill: muted)[$x_N < x_S$ and $y_N < y_S$]
    #v(10pt)
    #text(size: fs.caption, fill: muted)[All visible facts become constraints; distinct objects cannot share a point.]
  ],
  panel(stroke: ink, inset: 13pt)[
    #label([2 · ADD AN ANSWER AND SOLVE])
    #v(10pt)
    #text(size: fs.body, weight: "bold")[Could R be northeast of N?]
    #v(4pt)
    #text(size: fs.body, fill: muted)[Add $x_R > x_N$ and $y_R > y_N$. Our code prunes known constraints; Z3 checks whether all the rules can hold together.]
    #v(12pt)
    #text(size: fs.body, weight: "bold")[A checked example map, not raw Z3 output:]
    #v(5pt)
    #text(size: fs.body)[$N=(0,0)$   $R=(1,5)$   $S=(4,2)$]
    #v(7pt)
    #text(size: fs.body, fill: muted)[S is right and below R; N is left and below S; R is northeast of N. So the answer is *possible*.]
  ],
)

#v(13pt)

#panel(fill: wash, stroke: ink, inset: 12pt)[
  #label([3 · CHECK EACH CANDIDATE])
  #v(7pt)
  #text(size: fs.body)[
    *Could?* Solve facts + claim. A solution proves possible; no solution proves impossible. #linebreak()
    *Must?* Solve facts + NOT claim. No solution proves the claim is entailed.
  ]
  #v(6pt)
  #text(size: fs.caption, fill: muted)[First check that the facts are consistent. Each directional option supplies a claim; select by the answer policy, not by assuming “possible” means “necessary.”]
]

#speaker-note[
  Say: “Z3 is an SMT constraint solver, not a stored grid. Our code introduces integer X and Y variables for every named object, encodes every visible relation as comparisons, and forbids two objects from sharing a point. A Python path-consistency pass first prunes directions forced by the positive premises. Z3 then checks the full premise formula together with a proposed direction: R northeast of N. If satisfiable, it can return a model. For emitted solver witnesses, our code chooses a deterministic completion of pairwise directions and rank-compresses the model coordinates; these are not necessarily the raw integers Z3 first chose. The slide's numbers come from a separately checked six-object assignment in the report, with only R, S, and N shown. All nine premises hold in that assignment. For necessity we test the premises with R at or below N. But the facts force N below S below R, so the added condition is unsatisfiable. Since the premises are consistent, R must be above N. The numbers encode relative order only; they do not recover the unavailable source-map coordinates.”

  Source: `fyp-report/chapters/03-problem-formulation.typ`, “Directions as Axis Comparisons” and “Possibility, Entailment, and Ambiguity”; `fyp-report/figures/solver-correctness.typ`, first checked witness; `docs/spatialentail/semantics.md`, “World model” and “Inference.”
]

== Appendix · Generated contradiction

#label([COMPLETE SOUTHEAST REFUTATION FROM THE G2 RESPONSE])
#v(12pt)
#panel(stroke: ink, inset: 12pt)[
  #trace-text(generated-traces.g2.southeast_refutation)
]

#v(15pt)
#panel(fill: wash, stroke: ink, inset: 11pt)[
  #label([COMMENTARY · NOT PART OF THE TRACE])
  #v(7pt)
  #text(size: fs.body)[
    *A-REFUTE* assumes Southeast. *D-Y-1* derives N below R. *refutation-Y1* requires R below N. *refutation-C1* identifies the conflict.
  ]
  #v(6pt)
  #text(size: fs.body)[If S means “R is Southeast of N,” this proves $P tack.rr not S$: Southeast is impossible. It does *not* prove Northeast necessary.]
  #v(6pt)
  #text(size: fs.caption, fill: muted)[Saved production-renderer output for the checked three-object fixture, using the reference solver. Regenerate and compare with `docs/export_brief_traces.py --check`.]
]

#speaker-note[
  Say: “This complete refutation is loaded from the saved renderer output and also appears verbatim in the G2 response. It proves Southeast impossible, not Northeast necessary; Northeast and Northwest each have supporting models. The lower box is commentary, separate from the trace. The test fixture uses the reference backend because Z3 is not installed. It is not a historical training row or Z3's internal proof log. The exporter checks the certificates and exact output equality with --check.”

  Source: `docs/brief-traces.json`; `docs/export_brief_traces.py`; `spatial/v2/proof_renderers.py`, `render_direction_training_refutation`; `spatial/v2/tests/test_answer_certificates.py`.
]

== Appendix · Constructing the Natural target

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  panel(stroke: ink, inset: 12pt)[
    #label([1 · CLASSIFY THE ANSWERS])
    #v(10pt)
    #text(size: fs.body)[From the *visible facts*, the solver checks each direction.]
    #v(11pt)
    #text(size: fs.body, weight: "bold")[R relative to N:]
    #v(4pt)
    #text(size: fs.body, fill: muted)[Northeast and Northwest could hold. Southeast and Southwest cannot.]
  ],
  panel(stroke: ink, inset: 12pt)[
    #label([2 · BUILD CHECKED EVIDENCE])
    #v(10pt)
    #text(size: fs.body)[For each possible direction, keep a valid map. For each impossible direction, derive a contradiction.]
    #v(11pt)
    #text(size: fs.body, fill: muted)[A checker validates every map and replays every proof dependency.]
  ],
  panel(stroke: ink, inset: 12pt)[
    #label([3 · RENDER NATURAL])
    #v(10pt)
    #text(size: fs.body)[Fixed templates turn that evidence into premise steps, X/Y orderings, and an answer-policy conclusion.]
    #v(11pt)
    #text(size: fs.body, fill: muted)[The training text shows relative order, not the witness's raw coordinates.]
  ],
)

#v(15pt)

#panel(fill: wash, stroke: ink, inset: 13pt)[
  #label([4 · FORMAT THE RESPONSE])
  #v(9pt)
  #text(size: fs.body)[
    The assistant completion contains the rendered reasoning inside a `<think>` block, followed by an `Answer:` line with the selected menu letters.
  ]
  #v(9pt)
  #text(size: fs.caption, fill: muted)[Slides 9–11 load `docs/brief-traces.json` directly and are checked against regenerated output.]
]

#speaker-note[
  Say: “This is a second pipeline after the solver decision, not text generated by Z3. For the R/S/N example under an all-possible answer policy, Northeast and Northwest have supporting worlds; the two southern choices are impossible. From the solver's analyses, our certificate builder collects a checked coordinate model for each possible choice and constructs a rule-based refutation for each impossible choice. For a unique direction, it additionally constructs a positive proof. The certificate checker checks all declared candidates exactly once, validates every model against the premises, and replays the proof rules and their dependencies. Only then do deterministic Natural renderers turn the evidence into readable premise lines, relative X/Y orderings instead of raw coordinates, status lines, a conclusion, and an answer-policy sentence. The assistant training message wraps that trace in a think tag and adds the chosen letters from the shuffled menu. The answer-only control keeps the same prompt and final answer and omits the worked trace. This is constructed supervision, not an LLM's hidden reasoning. The saved fixture examples here were generated with the reference solver; they are not historical training rows.”

  Source: `spatial/v2/generation.py`, `_render_trace` and `GeneratedSpatialSample.as_sft_row`; `spatial/v2/certificate_generation.py`; `spatial/v2/answer_certificates.py`, `check_direction_answer_set`; `spatial/v2/answer_certificate_renderers.py`, `render_direction_training_trace`; `spatial/v2/model_certificate_renderers.py`.
]

== Appendix · What uniform rule sampling means

#text(size: fs.lead, weight: "bold")[At a fixed length, every complete allowed rule sequence has the same chance.]
#v(8pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  panel(stroke: ink, inset: 10pt)[
    #label([WHAT IS BEING SAMPLED?])
    #v(6pt)
    #text(size: fs.body)[Sample a prefix of n rule applications. Rules may repeat. For direct G1 construction, n = L.]
    #v(6pt)
    #text(size: fs.body)[Ambiguous Count already contributes depth G: n = L − G. Direction/Selection exclusion skeletons contribute zero.]
    #v(6pt)
    #text(size: fs.body, weight: "bold")[Among N prefixes allowed by the rule/pair/final-rule restrictions, each has probability 1/N.]
  ],
  panel(stroke: ink, inset: 10pt)[
    #label([WHY NOT PICK EACH RULE EQUALLY?])
    #v(6pt)
    #text(size: fs.body)[Some choices leave more allowed ways to finish the sequence than others.]
    #v(6pt)
    #text(size: fs.body)[The sampler counts these completions and chooses the next rule in proportion to that count.]
    #v(6pt)
    #text(size: fs.caption, fill: muted)[Equal complete-sequence probabilities do not mean equal probabilities for each rule at each position.]
  ],
)
#v(8pt)
#panel(fill: wash, stroke: ink, inset: 11pt)[
  #label([SMALL COUNTING EXAMPLE · ILLUSTRATIVE, NOT THE CONFIGURED PAIR LIST])
  #v(6pt)
  #text(size: fs.body)[Suppose only three length-two sequences are allowed: MP then DN; MP then TRC; IFF then DN.]
  #v(7pt)
  #text(size: fs.body)[Start with MP with probability 2/3, or IFF with 1/3. After MP, choose DN or TRC with probability 1/2 each.]
  #v(7pt)
  #text(size: fs.body, weight: "bold")[Each complete sequence gets 1/3: (2/3)(1/2), (2/3)(1/2), or (1/3)(1).]
]
#v(6pt)
#text(size: fs.caption, fill: muted)[Uniform proposals, not final questions: validation can reject them. Exponential weights allocate L/X levels, not rules.]

#speaker-note[
  Say: “Uniform means equal probability for each complete allowed sampled prefix at its fixed length n and policy, not independent one-in-five choices at each step or uniformity over the entire final proof. Direct G1 construction samples n = L. The configured ambiguous Count DJE skeleton contributes depth G, so its lifted prefix has n = L minus G. The configured Direction and Selection exclusion skeletons contribute zero. Without restrictions there would be five to the power n sequences. The sampler counts allowed suffixes and weights each next choice by its completion count. Final-rule rejection preserves conditional uniformity when a path is returned. The two-step example is a toy policy. Subsequent construction, solver, certificate, and realized L/G/X checks may reject problems, so accepted rows need not be uniform over prefixes or questions. Final L is the measured longest checked dependency depth. Exponential quotas separately allocate L/X levels. G counts possible answers, not proof branches.”

  Source: `spatial/v2/random_proof.py`, `_completion_counter`, `sample_rule_path`, `sample_policy_path`, and `exponential_quotas`; `spatial/v2/random_proof_generation.py`, construction and realized-depth checks; `spatial/v2/generation.py`, admission checks.
]
