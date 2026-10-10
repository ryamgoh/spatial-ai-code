#import "frontmatter/cover.typ": cover
#import "frontmatter/title-page.typ": title-page

#let project-title = "Formal Semantics for Evaluating and Improving Spatial Reasoning in Large Language Models"
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
#show figure.caption: set text(size: 10pt)

#set page(numbering: none)

#cover(
  project-title,
  student-name,
  department,
  school,
  university,
  academic-year,
)
#title-page(
  project-title,
  student-name,
  department,
  school,
  university,
  submission-year,
  project-number,
  advisor,
  evaluator,
)

#set page(numbering: "i")
#counter(page).update(1)

#include "frontmatter/abstract.typ"
#include "frontmatter/acknowledgements.typ"
#include "frontmatter/contents.typ"

#set page(numbering: "1")
#counter(page).update(1)

// The main report runs from Introduction through Conclusion and is limited to 55 pages.

#include "chapters/01-introduction.typ"
#include "chapters/02-related-work.typ"
#include "chapters/03-problem-formulation.typ"
#include "chapters/04-spatialentail-method.typ"
#include "chapters/05-experimental-design.typ"
#include "chapters/06-results-discussion.typ"
#include "chapters/07-conclusion.typ"

#pagebreak(weak: true)
#bibliography("references.bib", title: [References], style: "ieee")

#include "appendices/a-solver-correctness.typ"
#include "appendices/b-spatialentail-dataset.typ"
#include "appendices/c-experimental-configuration.typ"
#include "appendices/d-additional-results.typ"
