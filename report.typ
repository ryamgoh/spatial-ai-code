#let project-title = "Large Language Models for World Models"
#let student-name = "GOH EN RUI RYANN"
#let department = "Department of Computer Science"
#let school = "School of Computing"
#let university = "National University of Singapore"
#let academic-year = "2025/2026 Semester 2"
#let submission-year = academic-year
#let project-number = "H036660"
#let advisor = "Professor LEE WEE SUN"
#let evaluator = "Professor LEONG TZE YUN"

#set document(
  title: project-title,
  author: student-name,
)

#set page(
  paper: "a4",
  margin: (
    top: 2.5cm,
    bottom: 2.5cm,
    left: 3cm,
    right: 2.5cm,
  ),
)

#set text(
  font: "Times New Roman",
  lang: "en",
  size: 12pt,
)

#set par(
  justify: true,
  leading: 1em,
)

#set heading(numbering: "1.1")

#let part(number, title) = {
  pagebreak(weak: true)
  heading(
    level: 1,
    numbering: none,
    [Part #number — #title],
  )
}

#let unnumbered-chapter(title) = {
  pagebreak(weak: true)
  heading(level: 1, numbering: none, title)
}

#set page(numbering: none)

#align(center + horizon)[
  #text(size: 14pt)[B.Comp Dissertation]

  #v(2.5em)

  #text(size: 20pt, weight: "bold")[#project-title]

  #v(3em)

  By \
  #student-name

  #v(3em)

  #department \
  #school \
  #university

  #v(3em)

  #academic-year
]

#pagebreak()

#align(center)[
  #text(size: 14pt)[B.Comp. Dissertation]
]

#v(1fr)

#align(center)[
  #text(size: 20pt, weight: "bold")[#project-title]
]

#v(1fr)

#align(center)[
  By \
  #student-name
]

#v(1fr)

#align(center)[
  #department \
  #school \
  #university

  #v(2em)

  #submission-year
]

#v(1fr)

#align(left)[
  *Project ID:* #project-number \
  *Advisor:* #advisor \
  *Evaluator:* #evaluator \

  #v(1.5em)

  *Deliverables:*

  #pad(left: 2em)[
    Report: 1 Volume \
    Software: 1
  ]
]

#pagebreak()
#set page(numbering: "i")
#counter(page).update(1)

#heading(numbering: none, outlined: false)[Abstract]

#v(1fr)

*Subject Descriptors:*

*Keywords:*

#pagebreak()

#[
  #set par(leading: 1.5em)
  #heading(numbering: none, outlined: false)[Acknowledgements]
  // Write acknowledgements inside this block to retain double spacing.
]

#pagebreak()

#outline(
  title: [Table of Contents],
  depth: 2,
)

// Add lists of figures, tables, or abbreviations here when needed.

#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)

// The main report runs from Introduction through Conclusion and is limited to 55 pages.

#include "chapters/01-introduction.typ"

#part("I", [From Spatial Reasoning Evaluation to Benchmark Correction])

#include "chapters/02-research-landscape.typ"
#include "chapters/03-benchmark-diagnosis.typ"
#include "chapters/04-solver-and-correction.typ"
#include "chapters/05-corrected-benchmark-evaluation.typ"

#part("II", [Training Spatial Reasoning with AxisDecomposition])

#include "chapters/06-axis-decomposition.typ"
#include "chapters/07-dataset.typ"
#include "chapters/08-training-methodology.typ"
#include "chapters/09-results.typ"
#include "chapters/10-discussion.typ"
#include "chapters/11-conclusion.typ"

#unnumbered-chapter[References]

#unnumbered-chapter[Appendices]

#include "appendices/appendix-a-solver.typ"
#include "appendices/appendix-b-corrections.typ"
#include "appendices/appendix-c-dataset.typ"
