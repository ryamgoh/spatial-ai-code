#let cover(
  project-title,
  student-name,
  department,
  school,
  university,
  academic-year,
) = {
  align(center + horizon)[
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

  pagebreak()
}
