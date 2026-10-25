# SpatialEntail v2 propositional coverage

_Primary-source and implementation review, 17 October 2026._

## Bottom line

SpatialEntail v2 implements a **ground, quantifier-free classical Boolean
language over spatial relation atoms**, together with a **deliberately bounded
proof-certificate calculus**. Its formula syntax is broad: spatial atoms may be
nested under `NOT`, `AND`, `OR`, implication, and biconditional. Its proof rules
are much narrower: they include conjunction rules, modus ponens,
biconditional elimination, disjunction-by-cases, disjunctive syllogism,
double-negation elimination, explicit contradiction, restricted explosion,
and a special refutation discipline, but omit several standard introduction
and assumption-discharge rules.

It is therefore accurate to call the propositional part **bounded** or, in an
informal educational sense, **basic**. Calling it simply “basic propositional
logic” is underspecified and can mislead: it is not the standard complete
natural-deduction package, it contains derived shortcuts—including classical
double-negation elimination—normally presented as *additional* rules, and it is
extended by non-propositional spatial-order reasoning. For a logic-aware
audience, the concise description is:

> a bounded classical propositional proof fragment over ground spatial atoms,
> extended with typed spatial-order rules.

For the main FYP report, the less jargon-heavy equivalent is:

> SpatialEntail uses a bounded propositional fragment whose basic propositions
> are concrete pairwise spatial relations. It supplements the Boolean rules
> with checked operations for decomposing, inverting, composing, and
> recombining spatial directions.

## Formula syntax is not proof-rule coverage

The formula AST has a domain-specific atom, `RelationConstraint(subject,
reference, allowed)`, and recursively supports `Not`, n-ary nonempty `And` and
`Or`, `Implies`, and `Iff` ([`solver.py`, lines 104–141](../../../spatial/v2/solver.py#L104-L141),
[`solver.py`, lines 256–295](../../../spatial/v2/solver.py#L256-L295)). Both the
reference evaluator and Z3 translation give these connectives their ordinary
classical truth-functional meanings ([`solver.py`, lines 610–641](../../../spatial/v2/solver.py#L610-L641),
[`solver.py`, lines 859–896](../../../spatial/v2/solver.py#L859-L896)). This closely
matches the inductive TFL syntax in *forall x: Calgary*: sentence letters are
closed under negation, conjunction, disjunction, conditional, and
biconditional.[^syntax]

There are nevertheless important boundaries. SpatialEntail's atoms are ground
claims about finitely named objects, not freely interpreted propositional
letters; their models must also satisfy the spatial coordinate/order theory.
There are no quantifiers. `Contradiction` is a proof-object conclusion rather
than a `SpatialFormula`/falsum constant ([`proofs.py`, lines 92–97](../../../spatial/v2/proofs.py#L92-L97)).
Thus “full Boolean syntax over the available atoms” does **not** imply a full
propositional proof calculus.

## Inference-rule coverage

The table compares the checker—not merely the generator—with a conventional
classical natural-deduction presentation. *forall x* explicitly gives
introduction and elimination rules for each connective, with subproofs for
conditional introduction, biconditional introduction, disjunction elimination,
negation introduction, and indirect proof.[^basic]

| Family | SpatialEntail v2 checker | Relative to the conventional calculus |
|---|---|---|
| Premises and assumptions | Indexed premises; disjunct assumptions only when tied to an `Or`; one root refutation assumption in a refutation certificate | No unrestricted assumption/subproof mechanism |
| Conjunction | `AND_INTRODUCTION`, `AND_ELIMINATION` | Both directions covered; `And` is n-ary |
| Conditional | `MODUS_PONENS` (`→E`) | No conditional introduction (`→I`) |
| Biconditional | `IFF_ELIMINATION` in either direction | No biconditional introduction (`↔I`) |
| Disjunction | General finite `CASE_SPLIT` (`∨E`) and `DISJUNCTIVE_SYLLOGISM` | No disjunction introduction (`∨I`); DS is a derived shortcut |
| Negation and contradiction | `A, NOT A ⟹ Contradiction`; double-negation elimination; special claim-refutation certificates | No reusable negation-introduction rule deriving `NOT A`; no general indirect-proof rule deriving `A` from a closed `NOT A` subproof |
| Explosion | A contradiction may yield any formula only inside a branch | No unrestricted/global falsum elimination |
| Other shortcuts/conversions | None as general rules | No primitive LEM, modus tollens, contraposition, De Morgan conversion, Boolean normalization, or double-negation introduction |

The checker definitions and their side conditions are explicit in
[`proofs.py`, lines 304–528](../../../spatial/v2/proofs.py#L304-L528). In particular,
case split requires every disjunct to have a distinct scoped branch proving the
same result; explosion requires a branch; and assumptions cannot leak across
branches. A formula refutation is also a separate typed certificate: the root
claim assumption must lead to a dependent contradiction
([`proofs.py`, lines 684–743](../../../spatial/v2/proofs.py#L684-L743)). It is not a
general proof step that exports `NOT claim` for later reuse.

The distinction between *core* and *derived* rules matters. *forall x* presents
disjunctive syllogism and double-negation elimination as additional rules and
notes that the additions are not strictly necessary once the full basic system
is available.[^additional] Double-negation elimination is specifically a
classical principle; ordinary explosion does not require classical logic.[^classical]
SpatialEntail keeps those shortcuts while omitting several of the core
introduction rules. It is therefore not simply a small textbook
natural-deduction calculus obtained by stopping after the “basic rules” chapter.

Finally, `DIRECTION_DECOMPOSITION`, axis inversion/transitivity/contradiction,
and direction/atom recomposition are **domain rules**, not propositional rules
([`proofs.py`, lines 530–637](../../../spatial/v2/proofs.py#L530-L637)). They make
the overall certificate system richer in spatial reasoning without filling the
missing Boolean rule schemas.

## Checker coverage versus automatic proof search

The deterministic constructor is narrower than the checker interface. It
saturates selected eliminations and shortcuts, introduces conjunctions only
when demanded, and performs goal-directed finite case splits
([`proofs.py`, lines 996–1116](../../../spatial/v2/proofs.py#L996-L1116),
[`proofs.py`, lines 1368–1458](../../../spatial/v2/proofs.py#L1368-L1458)). The
repository's acceptance contract correspondingly says that semantically
decidable formulas can remain outside automatic proof-search coverage and
explicitly disclaims unrestricted proof-search completeness
([`certificates.md`, lines 3–8](../../spatialentail/certificates.md#L3-L8),
[`certificates.md`, lines 52–72](../../spatialentail/certificates.md#L52-L72)).

## Soundness and completeness: what the evidence supports

For a proof system, **soundness** is the direction from derivability to semantic
validity; **completeness** is the converse. *forall x* states these separately
and states that its TFL natural-deduction system is both sound and complete.[^metatheory]

The implemented Boolean checks mirror familiar classically truth-preserving
rule schemas, and the checker enforces scope and dependency conditions. That is
good implementation-level evidence for a soundness design. It is **not**, by
itself, a formal soundness theorem for every accepted SpatialEntail certificate:
such a result would also have to cover the spatial domain rules and connect all
typed certificate conclusions to the solver semantics.

Conversely, there is no basis here for claiming completeness with respect to
full classical propositional consequence. The system does not expose the same
general `Γ ⊢ A` relation as the comparison calculus, its positive and refutation
outputs are typed around spatial queries, and standard rule schemas are absent.
Those are concrete **coverage gaps** and justify rejecting a claim of “the
standard complete classical natural-deduction calculus.” They do not, without a
formal derivability definition and metatheoretic proof or counterexample,
establish a new theorem that the entire SpatialEntail system is formally
incomplete. The appropriately limited claim is: **classical semantic evaluation
is broad, while checked and automatically constructed proof coverage is
intentionally partial and has no completeness result.**

## Is “basic” fair?

Yes, if it means a finite, ground, quantifier-free Boolean fragment used to
exercise a selected set of familiar propositional transformations. No, if it
is intended as a precise claim that SpatialEntail implements a textbook's
“basic rules” or only trivial reasoning. The fragment supports nested case
analysis, contradiction-dependent refutation, and classical shortcuts, while
the full system also proves spatial order consequences. In technical writing,
“bounded propositional proof fragment” is both more accurate and more
informative than “basic propositional logic.”

## Authoritative sources

[^syntax]: Open Logic Project, [“Propositional Formula,” pinned source and formation clauses](https://github.com/OpenLogicProject/OpenLogic/blob/1e960beff9ed7835bf3e3f1335e21af3439cd107/content/propositional-logic/syntax-and-semantics/formulas.tex#L13-L32); P. D. Magnus et al., *forall x: Calgary*, [Chapter 6 §6.2, “Sentences of TFL”](https://forallx.openlogicproject.org/html/Ch6.html#S2), Fall 2025 edition, accessed 17 October 2026.
[^basic]: Open Logic Project, [official propositional natural-deduction rule schemata, pinned source](https://github.com/OpenLogicProject/OpenLogic/blob/1e960beff9ed7835bf3e3f1335e21af3439cd107/content/first-order-logic/natural-deduction/propositional-rules.tex#L16-L113); P. D. Magnus et al., *forall x: Calgary*, [Chapter 17, “Basic rules for TFL”](https://forallx.openlogicproject.org/html/Ch17.html), especially §§17.3–17.8, accessed 17 October 2026.
[^additional]: P. D. Magnus et al., *forall x: Calgary*, [Chapter 19, “Additional rules for TFL”](https://forallx.openlogicproject.org/html/Ch19.html), especially §§19.1 and 19.3–19.4; see also [Chapter 21, “Derived rules”](https://forallx.openlogicproject.org/html/Ch21.html), accessed 17 October 2026.
[^classical]: Open Logic Project, [natural deduction for intuitionistic logic, pinned source](https://github.com/OpenLogicProject/OpenLogic/blob/1e960beff9ed7835bf3e3f1335e21af3439cd107/content/intuitionistic-logic/introduction/natural-deduction.tex#L11-L17); Lean, [*Theorem Proving in Lean 4*, §3.5 “Classical Logic”](https://lean-lang.org/theorem_proving_in_lean4/Propositions-and-Proofs/#classical-logic), accessed 17 October 2026.
[^metatheory]: Open Logic Project, [definitions of soundness and completeness, pinned source](https://github.com/OpenLogicProject/OpenLogic/blob/1e960beff9ed7835bf3e3f1335e21af3439cd107/content/first-order-logic/introduction/soundness-completeness.tex#L25-L34), [propositional soundness theorem](https://github.com/OpenLogicProject/OpenLogic/blob/1e960beff9ed7835bf3e3f1335e21af3439cd107/content/propositional-logic/syntax-and-semantics/soundness.tex#L13-L17), and [propositional completeness corollary](https://github.com/OpenLogicProject/OpenLogic/blob/1e960beff9ed7835bf3e3f1335e21af3439cd107/content/propositional-logic/syntax-and-semantics/completeness.tex#L95-L97); P. D. Magnus et al., *forall x: Calgary*, [Chapter 22, “Soundness and completeness”](https://forallx.openlogicproject.org/html/Ch22.html), accessed 17 October 2026.
