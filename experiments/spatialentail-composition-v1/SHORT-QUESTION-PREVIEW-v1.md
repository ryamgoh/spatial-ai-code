# Short Unique Question Preview

Actual selected base: `production-b5-s19-which`. Independently replayed.
Only question wording and the shared system instructions changed.
Premises, options, labels, evidence, and reference prose are unchanged.

## Natural System Message

Solve the spatial entailment problem using the stated assumptions and answer contract. Explain entailed claims with deductions and non-entailed claims with checked counterexample relationships in ordinary English. Do not treat a claim as entailed merely because it could hold. Cite premises by their 1-based sentence positions when needed. End with exactly one line 'Answer: <letters>' using the required letter format.

Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists. For Which and Count, consider every object named in the premises except the reference object, independently of which options are displayed. Direction ranges over all eight exact directions. Count asks for an exact total, including zero; it is not the number of individually entailed members. Select its A-D option when exactly one answer value is entailed. Select F when that sole entailed value is not listed. Select E when no answer value is entailed, multiple answer values are entailed, or the premises are inconsistent. For Which, exactly one means one individually entailed object; another object that could qualify but is not guaranteed does not invalidate that sole entailment. When multiple objects are entailed, E indicates that there is no single answer, not uncertainty about those objects. End with exactly one line 'Answer: X', where X is one uppercase A-F letter.

## User Question

Consider a map with multiple objects:
Hospital is to the West of Police Station. Police Station is to the West of Restaurant. Clinic is to the West of Police Station. Pharmacy is to the East of Police Station. Supermarket is to the East of Police Station. Gallery is to the West of Supermarket. Clinic is to the West of Gallery. Police Station is to the West of Gallery.

Which object is west of Police Station?
A. Clinic
B. Pharmacy
C. Gallery
D. Hospital
E. Cannot be determined
F. None of the above

## Verified Answer

Clinic and Hospital are both individually entailed; the single-answer rule returns E.

## Existing Natural Reference

```text
Premises used (labels follow their order in the question):
P1: Hospital is to the West of Police Station.
P3: Clinic is to the West of Police Station.

Police Station has a greater x-coordinate than Clinic (premise sources: P3).
Clinic and Police Station share the same y-coordinate (premise sources: P3).
Therefore, Clinic is west of Police Station in every valid arrangement.

Police Station has a greater x-coordinate than Hospital (premise sources: P1).
Hospital and Police Station share the same y-coordinate (premise sources: P1).
Therefore, Hospital is west of Police Station in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E
```

The existing reference wording is shown verbatim, not presented as a prose rewrite.
