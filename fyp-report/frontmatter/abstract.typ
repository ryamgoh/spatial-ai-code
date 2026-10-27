#heading(numbering: none, outlined: false)[Abstract]

SpatialEntail studies atomic spatial relations and their composition under
explicit all-valid-map semantics. Its primary contribution is a planned
solver-verified dataset of Direction, Selection, and Count questions with three
matched answer contracts: Unique Answer, Listed Possibilities, and Exhaustive
Possibilities. Exact compass relations encode coordinate order, not distance
or equal-length movement. The dataset design includes uniquely determined,
underdetermined, and inconsistent premises, with checked derivations, witness
maps, and conflicting premise subsets.

A supporting experiment will compare two matched mixed-contract adapters
trained on Natural and Symbolic explanations rendered from the same verified
evidence. Accuracy and generated-token cost will be reported separately for
initial responses and constrained answer completion. SpatialMap is reserved for
external evaluation, with its existing corrected SINGLE release preserved.

This is a methodology draft. Executable semantic fixtures support the proposed
contracts, but the composition dataset has not passed production release gates
and the matched training comparison has not been run. No token savings,
accuracy preservation, or external improvement is claimed. The design is
post-hoc and informed by previously inspected SpatialMap failures.

#v(1fr)
*Subject Descriptors:* Artificial intelligence; knowledge representation and reasoning.

*Keywords:* spatial reasoning, spatial composition, answer contracts, verified datasets.

#pagebreak()
