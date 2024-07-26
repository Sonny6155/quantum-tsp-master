# Quantum TSP

This code is a minor rework of BOHRTECHNOLOGY's QUBO implementation, which
itself is based off Andrew Lucas' original Ising formulation. This means it
inherently supports asymmetric costs (and no-return paths with some minor
modifications), but I have not yet implemented settable start points, etc.

## Algorithm Breakdown
QUBO implements "soft" contraints via penalties and rewards based on what BVs
nodes were selected. Furthermore, this is restricted to checks of whether 1 or
2 nodes have been selected simulatenously. Unfortunately, hard constraints are
only possible by algorithm reformulations that smartly prune the problem space.

The full QUBO's Hamiltonian formulation is H_QUBO = H_A + H_B, on n^2 BVs in
the shape of a permutation matrix.

H_A corresponds to the Hamiltonian cycle/path problem, which adds penalty terms
to enforce two-way one-hot-ness of the permutation matrix:
- Time constraint (can't be multiple places at the same time)
    - Add postive penalty weights to horizontal rows
- Postion constraint (can't revisit the same node)
    - Add postive penalty weights to vertical rows

Large penalties are applied whenever an invalid pair of BVs are selected. This
also inherently caps it to a max of n selections. To enforce exactly n
selections, we can reward single variable selections we can give it substantial
reward. A simple option to balance these is a 2:1 penalty-reward.

H_B is the TSP's minimum cost objective, which is a straightforward mapping of
the input distance matrix to the corresponding QUBO edge. This version is
inherently asymmetric, as the QUBO edge (u, v+1) is fundamentally a different
edge to (u+1, v), obvious when laid out as permutation matrix. Cost mapping is
hence fairly straightforward.

In addition, this can be converted from a circuit to a path finder without a
needing a dummy node, since we can remove the terms that would wrap around from
the last row back to the first. With neither reward nor penalty, the algorithm
will just ignore the effect of return costs.

Decoding is then trivial by filtering invalid matrices then indexing 1s.

## Notable changes to BOHRTECHNOLOGY implementation
- Fixed solution decoder to reject invalid solutions.
    - Also now allows optionally filtering invalid distribution, too.
- One-hot enforcements now create a single unidirectional edge term. This is
equivalent to the original, but less duplicate QUBO terms is easier to trace.
- Single variable incentives were extracted into their own dedicated function
to help readbility and debugging.
- Added more config options to the class itself:
    - Added exact solving.
    - Added path solver.
- Removed solution storage responsibility from solver class, since exact
solving chews up RAM.
- Other refactoring for my own use case (var names etc).

## Possible improvements
- It should be possible to reduce circuit-mode to (n-1)^2 BVs by "affixing" the
first city. Presumably, we remove the first row and column, then add the
first city's cost to the subsequent row.

# Current Issues
Although it should handle 9 nodes, the dwave solver is giving bad results on
far less nodes. Since increasing it to 5k reads hasn't helped much, I suspect
it could be due to chain breaks. I'll probably need to set up hyperparam
optimisation across a variety of inputs to prove and fix this.

Pending questions:
- Is 800 penalty too high or low? How about cost scale factor?
- What is a good chain strength? Can we detect chain breaks manually?
- How many runs do we need?

Another problem to consider is if the max distance is too large, which would
make the normalised distances too small and thus fail to differentiate between
similar solutions. Depending on the expected inputs, high values may need to be
capped.

And finally, there are no proper unit tests. Need to get on that.

# Future Work
While this solution works and is currently still considered the
state-of-the-art, permutation ranking via Lehmer codes or similar could
theoretically provide optimal space encoding (around n log n, aka "practically"
linear). This would be a huge advancement for current annealers and removes the
need for large penalties/rewards that could affect chain stability.

A proposed method is the use of a permutation ranking algorithm (like Lehmer
coding), but the issue lies in find such a miraculous encoding that is
convertable into binary while allowing pairs of binary digits represent node
pair costs.