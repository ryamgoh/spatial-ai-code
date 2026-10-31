"""Bounded, question-local reuse; canonical witnesses and audits are unchanged."""

from collections import OrderedDict
from copy import deepcopy

from spatial.v2.solver import (
    And,
    ClaimAnalysis,
    Not,
    SpatialSolverV2,
    WhichAnalysis,
    WhichQuery,
    membership_constraint,
)


class QuestionSolver(SpatialSolverV2):
    """Never share this solver across processes; clear it between admissions."""

    def __init__(self, *, timeout_ms=5000, cache_size=256):
        super().__init__(backend="z3", timeout_ms=timeout_ms)
        if type(cache_size) is not int or cache_size <= 0:
            raise ValueError("cache_size must be a positive integer")
        self._limit = cache_size
        self._results = OrderedDict()

    def clear(self):
        self._results.clear()

    def _remember(self, key, value):
        # Unknown/timeouts must remain errors, not reusable semantic conclusions.
        if value.error is None:
            self._store(key, value)
        return value

    def _store(self, key, value):
        self._results[key] = deepcopy(value)
        self._results.move_to_end(key)
        while len(self._results) > self._limit:
            self._results.popitem(last=False)

    def analyze(self, problem, *, include_witnesses=True):
        key = ("analysis", problem, include_witnesses)
        if key in self._results:
            return deepcopy(self._results[key])
        return self._remember(
            key, super().analyze(problem, include_witnesses=include_witnesses)
        )

    def analyze_entailment(self, problem):
        key = ("entailment-analysis", problem)
        if key in self._results:
            return deepcopy(self._results[key])
        return self._remember(key, super().analyze_entailment(problem))

    def negative_categories(self, problem):
        key = ("negative-categories", problem)
        if key in self._results:
            return deepcopy(self._results[key])
        # Exceptions propagate and are not cached.
        value = super().negative_categories(problem)
        self._store(key, value)
        return value

    def assess(self, problem, claim):
        key = ("claim", problem, claim)
        if key in self._results:
            return deepcopy(self._results[key])
        analysis = self._results.get(("analysis", problem, True))
        if (
            isinstance(problem.query, WhichQuery)
            and isinstance(analysis, WhichAnalysis)
            and analysis.consistent
            and claim
            in {
                membership_constraint(c, problem.query)
                for c in problem.query.candidates
            }
        ):
            # The cached positive map is the same lexicographic completion of
            # premise AND claim. Preserve object order; do not reuse arbitrary maps.
            candidate = claim.subject
            entailed = candidate in analysis.entailed_entities
            try:
                counterexample = (
                    None
                    if entailed
                    else self._engine.satisfy(
                        problem.objects, And((problem.premise, Not(claim)))
                    )
                )
            except RuntimeError as exc:
                return ClaimAnalysis(False, engine=self._engine.name, error=str(exc))
            value = ClaimAnalysis(
                True,
                possible=candidate in analysis.possible_entities,
                entailed=entailed,
                witness=deepcopy(analysis.witnesses.get(candidate)),
                counterexample=counterexample,
                engine=analysis.engine,
            )
            return self._remember(key, value)
        return self._remember(key, super().assess(problem, claim))
