#import "@preview/algorithmic:1.0.7" as algo
#import algo: algorithm-figure, style-algorithm

#set document(title: "SpatialEntail: High-Level Problem Generation")
#set page("a4", margin: 20mm, numbering: "1")
#set text(size: 10pt, lang: "en")
#set par(leading: 0.55em)
#show: style-algorithm.with(breakable: true)

#align(center)[
  #text(size: 16pt, weight: "bold")[SpatialEntail Problem Generation]
]

*Idea:* choose what the question should establish, work backward to build its
premises, then check what those premises actually imply.

*Inputs:* question type, desired number of possible answers, reasoning length,
and number of distractors.

#algorithm-figure(
  "Generate one verified spatial reasoning problem",
  inset: 0.1em,
  indent: 0.8em,
  vstroke: 0.4pt + luma(190),
  {
    import algo: *
    For([each attempt within the retry budget], {
      Line[Choose object names]
      Line[Build a target conclusion or a starting set of possible answers]
      Assign([path], [sample a valid rule sequence for the requested length])
      Assign([goal], [a statement needed to establish the target])
      For([rule in reverse(path)], {
        Line[Replace goal with the statement needed before applying this rule]
        Line[Add any supporting premise required by the rule]
      })
      Line[Add the final goal as a premise; add distractors and shuffle]
      Line[Solve the resulting question and check its proof]
      If([the question does not meet the requested constraints], {
        Line[Reject this candidate and try again]
      })
      Line[Render and validate the question, options, answer, and solution]
      Return([the verified example])
    })
    Return([generation failure])
  },
)

Valid rule sequences are sampled uniformly before candidate rejection; the
accepted examples need not be uniform. The solver determines the final answers,
rather than trusting the generator's intended result.
