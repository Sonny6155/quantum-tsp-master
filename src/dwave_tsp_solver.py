import dimod
from dwave.system.samplers import DWaveSampler           # Library to interact with the QPU
from dwave.system.composites import EmbeddingComposite   # Library to embed our problem onto the QPU physical graph
import numpy as np

# import itertools
# import scipy.optimize


class DWaveTSPSolver(object):
    """
    Class for solving Travelling Salesman Problem using D-Wave via QUBO.

    Based on Andrew Lucas' Ising formulation and BOHRTECHNOLOGY's QUBO
    implementation. Supports asymmetric costs and paths, but not fixed
    start point (yet).
    """
    def __init__(
        self,
        distance_matrix: np.ndarray,
        sampler: str = "dwave",
        sapi_token: str | None = None,
        url: str | None = None,
        solver: str | None = None,
        circuit: bool = True,
        constraint_factor: int = 800,  # NOTE: I suspect this needs to be reduced, since normalised cost can get really small on matrices with large values, and constraints can get hude on long routes
        cost_factor: int = 10,  # NOTE: Similarly, this might need to increase depending on the actual solver specs
    ):
        """
        Inits DWaveTSPSolver with immediate QUBO formulation and underlying
        sampler management.

        Based on Andrew Lucas' Ising formulation and BOHRTECHNOLOGY's QUBO
        implementation. Supports asymmetric costs and paths, but not fixed
        start point (yet).

        TODO: Plan to allow a few more post-init adjustable hyperparams
        TODO: Plan to move qubo init into public function that can be called again post init with optional new data, allowing either hyperparam optimisation for the same problem or new problem with config reuse

        Parameters
        ----------
        distance_matrix : ndarray
            Asymmetric distance matrix to find a shortest route for.
        sampler : str, default="dwave"
            Whether to solve on. Accepts "dwave" and "exact".
            NOTE: No plans to add hybrid yet. Recursive cluster is also likely to go through an additional wrapper layer to simplify testing
        sapi_token : str or None, default=None
            The D-Wave Leap API token of format "DEV-abc...". Not required for
            local solvers like exact solving.
        url : str or None, default=None
            A URL targeting a server location for API calls. Takes the format
            "https://na-west-1.cloud.dwavesys.com/sapi/v2/". Not required for
            local solvers like exact solving.
        solver : str or None, default=None
            Name of the solver/underlying machine. Available machines can be
            found via Leap, taking the format "AdvantageX.Y". Not required for
            local solvers like exact solving.
        circuit : bool, default=True
            If True, disconnects last node from first to force path discovery.
        constraint_factor : int, default=800
            Scales QUBO weight for penalising constraint violations.
        cost_factor : int, default=10
            Scales QUBO weight for costs. Cannot exceed constraint factor.
        """
        # Fast fail validation
        if len(distance_matrix) >= 10:
            raise ValueError("D-Wave and exact solvers currently cannot handle more than 10 TSP nodes.")
            # NOTE: This holds for now, but not once we add hybrid solvers or our recursive clustering strat
        elif cost_factor < 1 or constraint_factor < 1:
            raise ValueError("Cost/constraint factors must be at least 1.")
        elif cost_factor > constraint_factor:
            raise ValueError("Cost factor cannot exceed constraint factor.")

        # Problem properties
        max_distance = np.max(np.array(distance_matrix))
        scaled_distance_matrix = distance_matrix / max_distance
        self._distance_matrix = scaled_distance_matrix  # Normalisation is required for constraints to work across all cost scales
        self._circuit = circuit

        # Sampler properties
        self._constraint_factor = constraint_factor
        self._cost_factor = cost_factor
        self._chain_strength = 800  # TODO: Check if chain breaks is a problem based on valid solution count and chain_break_frequency. If so, bump it to 1000
        self._num_runs = 5000  # Super high number just in case. Probably could be set to 1024 after testing is done.

        # Sampler/solver object setup
        if sampler == "dwave":
            if len(distance_matrix) >= 10:
                raise ValueError("D-Wave currently cannot handle more than 10 TSP nodes.")
            # NOTE: This holds for now, but not once we add hybrid solvers or our recursive clustering strat
            elif None in (sapi_token, url, solver):
                raise ValueError("API-based samplers must have sapi_token/url/solver values.")

            self._sampler = EmbeddingComposite(DWaveSampler(token=sapi_token, endpoint=url, solver=solver))

        elif sampler == "exact":
            if len(distance_matrix) >= 10:
                raise ValueError("Exact solvers currently cannot handle more than 4 TSP nodes.")
            # NOTE: Ditto note

            self._sampler = dimod.ExactSolver()

        # Problem setup
        n = len(self._distance_matrix)

        cost_terms = self._add_cost_objective(self._cost_factor, self._distance_matrix, self._circuit)  # The TSP-specific addition to constraints
        incentive_terms = self._add_selection_incentive(self._constraint_factor, n)
        time_times = self._add_time_constraints(self._constraint_factor, n)
        position_terms = self._add_position_constraints(self._constraint_factor, n)

        self._qubo_dict = cost_terms | incentive_terms | time_times | position_terms


    # Helper functions
    def _add_cost_objective(self, cost_factor: int, distance_matrix: np.ndarray, circuit: bool) -> dict[tuple[int, int], int]:
        """
        Encodes asymmetric distances as QUBO edges between sequential nodes.

        Runs in O(n^3), creating n^2(n-1) new edges. If path, n(n-1)^2.

        Parameters
        ----------
        cost_factor : int
            Scaling factor to adjust distance weighting.
        distance_matrix : ndarray
            Asymmetric distance matrix to find a shortest route for.
        circuit : bool
            If True, disconnects last node from first to force path discovery.

        Returns
        -------
        dict
            Collection of QUBO edge weights, indexed by 2-tuple of BV indices.
            Note that QUBO just sums (x, y) and (y, x) edges.
        """
        qubo_dict = {}
        n = len(distance_matrix)

        # If path, do not connect end back to start. Neither reward nor
        # penalise, acting like a dummy 0 cost node without the node wastage.
        max_row = n if circuit else n-1

        # Links each BV to the row below (represents travel cost to next node)
        for row in range(max_row):
            for col in range(n):
                qubit_a = row * n + col
                for col_iter in range(n):
                    if col != col_iter:  # Avoid one-hot violations
                        qubit_b = (row + 1)%n * n + col_iter
                        qubo_dict[(qubit_a, qubit_b)] = cost_factor * distance_matrix[col][col_iter]
        
        return qubo_dict


    def _add_selection_incentive(self, constraint_factor: int, n: int) -> dict[tuple[int, int], int]:
        """
        Encourages variable selection, effectively discouraging less than n
        selections.

        Runs in O(n^2), creating n^2 new edges (may as well be self-weights).

        Parameters
        ----------
        constraint_factor : int
            Scaling factor to adjust constraint weighting.
        n : int
            Number of nodes in a route.

        Returns
        -------
        dict
            Collection of QUBO edge weights, indexed by 2-tuple of BV indices.
            Note that QUBO just sums (x, y) and (y, x) edges.
        """
        qubo_dict = {}

        # Basically, reward every individual variable selection
        for row in range(n):
            for col in range(n):
                qubit_a = row * n + col
                qubo_dict[(qubit_a, qubit_a)] = -constraint_factor

        return qubo_dict


    def _add_time_constraints(self, constraint_factor: int, n: int) -> dict[tuple[int, int], int]:
        """
        Strongly penalises being in multiple places at the same time (aka
        horizontal one-hot violations). When paired with position constraints,
        this inherently discourages more than n selections.

        Runs in O(n^3), creating n(n-1)^2 new edges.

        Parameters
        ----------
        constraint_factor : int
            Scaling factor to adjust constraint weighting.
        n : int
            Number of nodes in a route.

        Returns
        -------
        dict
            Collection of QUBO edge weights, indexed by 2-tuple of BV indices.
            Note that QUBO just sums (x, y) and (y, x) edges.
        """
        qubo_dict = {}

        for row in range(n):
            for col in range(n-1):  # Link cols to their right-ward BVs
                qubit_a = row * n + col
                for col_offset in range(col+1, n):
                    qubit_b = row * n + col_offset
                    qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_factor
                    # Penalty must be strong enough to offset variable selection incentive, hence double
                    # TODO: Check if double is needed, since any exceedance inherently violates multiple times anyways

        return qubo_dict


    def _add_position_constraints(self, constraint_factor: int, n: int) -> dict[tuple[int, int], int]:
        """
        Strongly penalises revisiting the same node (aka vertical one-hot
        violations). When paired with time constraints, this inherently
        discourages more than n selections.

        Runs in O(n^3), creating n(n-1)^2 new edges.

        Parameters
        ----------
        constraint_factor : int
            Scaling factor to adjust constraint weighting.
        n : int
            Number of nodes in a route.

        Returns
        -------
        dict
            Collection of QUBO edge weights, indexed by 2-tuple of BV indices.
            Note that QUBO just sums (x, y) and (y, x) edges.
        """
        qubo_dict = {}

        # Same as time penalty, but downwards instead
        for row in range(n-1):
            for col in range(n):
                qubit_a = row * n + col
                for row_offset in range(row+1, n):
                    qubit_b = row_offset * n + col
                    qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_factor

        return qubo_dict


    def _validate_permutation(self, matrix: np.ndarray) -> bool:
        """
        Validate that permutation matrix is correctly one-hot encoded.

        Parameters
        ----------
        matrix : ndarray
            Permutation matrix. Assumes binary matrix.

        Returns
        -------
        bool
            True if one-hot encoded.
        """
        # Every row/col sums to 1
        return np.all(np.sum(matrix, axis=0) == 1) and np.all(np.sum(matrix, axis=1) == 1)


    def _decode_solution(self, response: dimod.SampleSet, valid_only: bool = True) -> tuple[list[int] | None, dict[tuple[int, ...], tuple[int, int]]]:
        """
        Decodes BV permutation matrix, returning 0-based route.

        Does not support circuit filitering yet (and would be hard).

        Parameters
        ----------
        response : SampleSet
            Response from D-Wave (QUBO-specific). Contains sampled info in a
            NumPy recarray format. See official docs: https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/sampleset.html
        valid_only : boolean, default=True
            Filters distribution to valid if True. Highly recommended to filter
            when exact solving.

        Returns
        -------
        best_solution : list of int or None
            The valid route with the lowest energy, if found.
        valid_distribution : dict
            Collection of energy and hit count, indexed by tupled route.
            If valid_only is disabled, invalid solutions will be added,
            but left in BV form as it is unable to be decoded.
            TODO: Consider a more *sensible* table format...
        """
        valid_distribution = {}
        min_energy = np.argmax(response.record.energy)
        best_solution = None
        node_count = int(np.sqrt(len(response.record[0].sample)))

        for entry in response.record:
            solution_matrix = np.reshape(entry.sample, (node_count, node_count))

            # If solution is valid, record and update best if lowest energy
            if self._validate_permutation(solution_matrix):
                solution = [np.where(row==1)[0][0] for row in solution_matrix]
                valid_distribution[tuple(solution)] = (entry.energy, entry.num_occurrences)
                if entry.energy <= min_energy:
                    min_energy = entry.energy
                    best_solution = solution

            # Even if invalid, user can choose to record, albeit in BV form
            elif not valid_only:
                valid_distribution[tuple(node for node in entry.sample)] = (entry.energy, entry.num_occurrences)

        return best_solution, valid_distribution

    # Public methods
    def solve_tsp(self, valid_only: bool = True) -> tuple[list[int] | None, dict[tuple[int, ...], tuple[int, int]]]:
        """
        Solve TSP via QUBO.

        Parameters
        ----------
        valid_only : boolean, default=True
            Filters distribution to valid if True. Highly recommended to filter
            when exact solving.

        Returns
        -------
        best_solution : list of int or None
            The valid route with the lowest energy, if found.
        valid_distribution : dict
            Collection of energy and hit count, indexed by tupled route.
            If valid_only is disabled, invalid solutions will be added,
            but left in BV form as it is unable to be decoded.
            TODO: Consider a more *sensible* table format... like dataframe
        """
        response = self._sampler.sample_qubo(self._qubo_dict, chain_strength=self._chain_strength, num_reads=self._num_runs)
        return self._decode_solution(response, valid_only)
        # NOTE: Confirmed that built-in TSP by dwave is symmetric only and not really configurable either
    

    @property
    def qubo_dict(self) -> dict[tuple[int, int], int]:
        return self._qubo_dict.copy()
