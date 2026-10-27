#pagebreak(weak: true)
#heading(level: 1, numbering: none)[Appendix B: Dataset Records]

The composition release must retain story and family identifiers, source
premise IDs, query family, reference and target or eligible candidates,
complete semantic answer, semantic case, and split assignment. Related story
variants remain in the same split.

Each contract/menu view records the answer contract, four substantive values,
their A-D permutation, fixed E/F labels, final letter answer, and precise
escape reason. Possible-answer count and returned-answer count are separate.
Natural and Symbolic renderings share the selected verified evidence and
final answer.

Audit metadata retains witnesses, counterexamples, conflicting subsets,
checked derivations, full-domain feasibility and exclusion outcomes,
chain lengths, verified distractor sets, and validation receipts. Raw audit
logs are not supervised reasoning targets.

The release receipt includes source revisions, content hashes, generation
seeds, counts of unique bases and rendered examples per arm, supported
evidence coverage, failures and exclusions, split checks, tokenizer settings,
manual audit sample IDs, and audit outcomes.

This appendix specifies required records, not a finalized production schema.
The schema and all admitted evidence categories must be frozen before bulk
generation.
