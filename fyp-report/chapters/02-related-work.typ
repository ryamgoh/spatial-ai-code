#pagebreak(weak: true)
= Related Work

== Textual Spatial Reasoning

StepGame provides controlled multi-hop directional reasoning over eight
relations @shi2022stepgame. It motivates studying chain length, but the
present study does not reproduce its train-short/test-long campaign.
SpartQA includes open-world spatial reasoning and several question types
@mirzaee2021spartqa. Open-world semantics is therefore not itself a novelty
claim. SpaRTUN and ReSQ further motivate distinguishing synthetic supervision
from human-language evaluation @mirzaee2022spartun.

SpatialEval evaluates textual, visual, and combined spatial reasoning
@wang2024spatialeval. This study uses SpatialMap as a fixed external check.
Its adapted labels are interpreted under SpatialEntail's declared semantics;
those assumptions are not attributed to the original authors.

== Spatial Explanations

SpaRP constructs deductively verified spatial paths, verbalizes them, and
uses them for model fine-tuning @rizvi2024sparp. This is direct prior art for
solver-derived spatial supervision. SpatialEntail's intended contribution
instead emphasizes controlled matched answer contracts and evidence coverage.

Chain-of-Symbol replaces natural-language reasoning with compact symbols in
few-shot prompting @hu2024chainofsymbol. Its results motivate a representation
comparison but do not establish the effect of matched Natural/Symbolic
supervised fine-tuning. In particular, savings in demonstration reasoning
are not measurements of this study's generated-token cost.

Graph-based StepGame training uses ordered relation triples
@zhou2024graphsynthetic, while Q-Chain incorporates formal spatial chains
into training constraints @premsri2025neurosymbolic. These precedents make
clear that symbolic spatial representations are established techniques,
not the contribution claimed here.

== Verification and Positioning

`interwhen` translates intermediate spatial claims into Z3 constraints and
uses detected violations to steer generation @bhat2026interwhen. Using Z3
or checking spatial statements is therefore not a broad novelty claim.

The proposed dataset combines atomic spatial composition, explicit
Unique/Listed/Exhaustive contracts, and checkable derivations, witnesses, and
conflicts. The supporting comparison holds the selected evidence and final
answers fixed across Natural and Symbolic supervision. Production evidence
coverage and empirical training results remain to be established.
