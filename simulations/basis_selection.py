import numpy as np
from heston import generate_heston_paths
from option_valuation import estimate_continuation_value
from math_apps import basis_expansions
from typing import Callable


def estimate_convergence_path(basis_expansion: Callable, basis_expansion_degree: int = 2, avg_horizon: int = 6, tol: float = 1e-5) -> tuple[list[int], list[float]] | None:
    change = np.inf
    current_avg_change = np.inf
    prev_continuation_value = 0
    num_paths = 0
    continuation_values = []
    change_vals = []
    num_paths_vals = []

    # Parameters -- all paths will use these
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    r = 0.04
    S0 = 100.
    tau = 1.0
    K = 90

    total_num_paths = 10000
    S_master = generate_heston_paths(tau, kappa, theta, sigma, rho, v0, S0, r, N = 1000, M = total_num_paths, num_parallel_procs=10) # Note that if we don't use pseudoinversion, we mandate that M >= N to avoid singular matrix

    # Find optimal M
    while current_avg_change > tol or num_paths <= total_num_paths:
        num_paths += 1

        S = S_master[:num_paths, :]
        current_continuation_value = estimate_continuation_value(S, K, r, "call", basis_expansion, N=basis_expansion_degree)

        change = abs(current_continuation_value - prev_continuation_value)
        continuation_values.append(current_continuation_value)
        change_vals.append(change)
        num_paths_vals.append(num_paths)
        current_avg_change = np.mean(change_vals[-min(avg_horizon, len(change_vals) - 1):])

        print(f"M = {num_paths} -> Continuation value: {current_continuation_value}, Change: {change}, Avg change: {current_avg_change}")

        prev_continuation_value = current_continuation_value
    
    if num_paths >= total_num_paths:
        print("Did not converge!")
        return
    else:
        print(f"Converged at {num_paths} paths")
        return num_paths_vals, change_vals


if __name__ == "__main__":
    estimate_convergence_path(basis_expansion=basis_expansions.polynomial_basis)