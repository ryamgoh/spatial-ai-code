#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "diagram-style.typ": diagram-border, diagram-panel, input-fill, contract-fill, answer-fill

#let modality-alignment = figure(
  align(center)[
    #diagram(
      spacing: (6mm, 6mm),
      edge-stroke: .65pt + diagram-border,
      label-size: 7.5pt,
      label-sep: 3pt,
      mark-scale: 65%,

      node((1, 0), diagram-panel(
        [Benchmark instance],
        [complete latent map $m^*$ \ shared question and oracle $a^*$],
        width: 50mm,
        fill: input-fill,
      ), name: <instance>),

      node((0, 1), diagram-panel(
        [TQA],
        [textual scene description \ textual question],
        fill: contract-fill,
      ), name: <tqa>),
      node((1, 1), diagram-panel(
        [VQA],
        [map image \ textual question],
        fill: contract-fill,
      ), name: <vqa>),
      node((2, 1), diagram-panel(
        [VTQA],
        [map image and description \ textual question],
        fill: contract-fill,
      ), name: <vtqa>),

      edge(<instance.south-west>, <tqa.north>, "->", [render text]),
      edge(<instance.south>, <vqa.north>, "->", [render image]),
      edge(<instance.south-east>, <vtqa.north>, "->", [render both]),

      node((0, 2), diagram-panel(
        [Open-world TQA audit],
        [consider every map satisfying \ the released text \ report all remaining \ answer possibilities],
        width: 46mm,
        fill: answer-fill,
      ), name: <open-audit>),
      edge(<tqa.south>, <open-audit.north>, "->", [analyse visible evidence]),
    )
  ],
  caption: [SpatialEval exposes aligned modality-specific views of each
    benchmark instance while retaining the same question and oracle. The TQA
    audit evaluates the set of maps admitted by the released text rather than
    assuming access to the latent map.],
)
