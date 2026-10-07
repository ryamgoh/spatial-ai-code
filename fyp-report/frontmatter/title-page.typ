#let title-page(
  project-title,
  student-name,
  department,
  school,
  university,
  submission-year,
  project-number,
  advisor,
  evaluator,
) = {
  align(center)[
    #text(size: 14pt)[B.Comp. Dissertation]
  ]

  v(1fr)

  align(center)[
    #text(size: 20pt, weight: "bold")[#project-title]
  ]

  v(1fr)

  align(center)[
    By \
    #student-name
  ]

  v(1fr)

  align(center)[
    #department \
    #school \
    #university

    #v(2em)

    #submission-year
  ]

  v(1fr)

  align(left)[
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

  pagebreak()
}
