# Coverage Selection Solver Revision

The v1 Z3 Optimize attempt timed out after its fixed 120-second optimization
limit with `unknown: sat.canceled`. Its result is preserved. No selection was
accepted from that attempt.

Repeat the identical selection policy on the identical saved pool with
SciPy/HiGHS binary MILP. No hard constraint, objective order, quota, source
example, or diversity stratum changes.

Solve the four objectives sequentially, each with a 120-second limit and zero
relative MIP gap. Fix each proven integer optimum as an equality before solving
the next objective. Return failure if any stage cannot prove its optimum.
Auxiliary binary variables encode exact OR for covered cells and topologies.
Recheck all hard constraints and integer objective values in Python.

This is a numerical optimization solver with floating-point tolerances, not a
formal rational proof checker. Integer solutions/objectives are independently
checked; report solver bounds and gaps. The original v1 limitations and
user approval checkpoint remain unchanged.
