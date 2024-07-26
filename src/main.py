import numpy as np

import time
import os

import TSP_utilities
from dwave_tsp_solver import DWaveTSPSolver


def dw_test_case():
    asym_matrix = np.array(
        [
            [0, 50, 75, 82, 1],
            [1, 0, 40, 76, 32],
            [12, 23, 0, 1, 23],
            [99, 2, 94, 0, 62],
            [24, 67, 1, 75, 0],
        ]
    )
    # Path should find [0, 4, 2, 3], 1 + 1 + 1 + 1 = 5
    # Circuit should find some shift of [0, 4, 2, 3, 1], 1 + 1 + 1 + 1 + 2 = 6

    # For exact solver, since it otherwise takes ages
    asym_matrix_mini = np.array(
        [
            [0, 1, 3],
            [3, 0, 5],
            [2, 3, 0],
        ]
    )
    # Path should find [2, 0, 1], 2 + 1 = 3
    # Circuit should find some shift of [1, 2, 0], 5 + 2 + 1 = 8
    asym_matrix_med = np.array(
        [
            [0, 50, 1, 82],
            [1, 0, 40, 76],
            [12, 23, 0, 1],
            [99, 2, 94, 0],
        ]
    )
    # Path should find [1, 0, 2, 3], 1 + 1 + 1 = 3
    # Circuit should find some shift of [2, 3, 1, 0], 1 + 2 + 1 + 1 = 5

    # starting_node = 0  # Won't need this for now, as we can just choose the closest node to depot in custom code after?

    sampler_config = {
        "sampler": "exact",  # TODO: Currently debugging. Will exact is local, while dwave uses the token args
        "sapi_token": os.environ["DWAVE_TOKEN"],
        "url": os.environ["DWAVE_URL"],
        "solver": os.environ["DWAVE_SOLVER"],
    }

    problem_config = {
        "circuit": True,
        # Also need to attempt constraint/cost factor tuning
    }

    # Solve each test case matrix
    for tsp_matrix in [asym_matrix]:
        # Call and time solving process
        print("DWave solution")
        start_time = time.time()
        dwave_solver = DWaveTSPSolver(tsp_matrix, **sampler_config, **problem_config)
        dwave_solution, dwave_distribution = dwave_solver.solve_tsp()
        end_time = time.time()
        calculation_time = end_time - start_time
        print("Calculation time:", calculation_time)

        # Print solution and cost
        solution_cost = TSP_utilities.calculate_cost(
            tsp_matrix, dwave_solution, problem_config["circuit"]
        )
        print("Solution:", dwave_solution)
        print("Solution cost:", solution_cost)

        # Print all costs
        print("Route, cost, (energy, occurrences)):")
        costs = [
            (
                sol,
                TSP_utilities.calculate_cost(
                    tsp_matrix, sol, problem_config["circuit"]
                ),
                dwave_distribution[sol],
            )
            for sol in dwave_distribution
        ]
        for cost in costs:
            print(cost)

        # TSP_utilities.plot_solution('dwave_' + str(bf_start_time), nodes_array, dwave_solution)
        # TODO: Need a new plotting solution using networkx to handle asym matrices
        # TODO: would be good to be able to plot qubo too, though that might get messy for large ones


if __name__ == "__main__":
    dw_test_case()
