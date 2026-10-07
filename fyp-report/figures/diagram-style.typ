#let diagram-ink = rgb("#1f2937")
#let diagram-muted = rgb("#475569")
#let diagram-border = rgb("#64748b")
#let input-fill = rgb("#f1f5f9")
#let contract-fill = rgb("#e0f2fe")
#let core-fill = rgb("#dbeafe")
#let analysis-fill = rgb("#eef2ff")
#let answer-fill = rgb("#ecfdf5")
#let explanation-fill = rgb("#fff7ed")
#let audit-fill = rgb("#fef2f2")

#let diagram-panel(
  title,
  body,
  width: 42mm,
  fill: input-fill,
  title-size: 9.5pt,
  body-size: 8.2pt,
) = block(
  width: width,
  inset: (x: 8pt, y: 7pt),
  radius: 4pt,
  stroke: .7pt + diagram-border,
  fill: fill,
  align(center)[
    #set par(justify: false)
    #text(size: title-size, weight: "bold", fill: diagram-ink)[#title]
    #v(3pt)
    #text(size: body-size, fill: diagram-ink)[#body]
  ],
)
