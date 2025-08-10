import numpy as np
import matplotlib.pyplot as plt
from heston import generate_heston_paths, generate_heston_hull_white_paths
from option_valuation import estimate_continuation_value
from math_apps import basis_expansions, finite_differencing
from typing import Callable
from multiprocessing import Pool, cpu_count


def estimate_convergence_path(basis_expansion: Callable, S_master: np.ndarray, K: float, r: float | np.ndarray, v: None | np.ndarray, tau: float, basis_expansion_degree: int = 2, avg_horizon: int = 6, tol: float = 1e-4) -> tuple[int, list[int], list[float]] | None:
    change = np.inf
    current_avg_change = np.inf
    prev_continuation_value = 0
    num_paths = 0
    continuation_values = []
    change_vals = []
    num_paths_vals = []

    total_num_paths = S_master.shape[0]

    # Find optimal M
    while current_avg_change > tol and num_paths <= total_num_paths:
        num_paths += 1

        S = S_master[:num_paths, :]
        current_continuation_value = estimate_continuation_value(S, K, r, v, tau, "call", basis_expansion, N=basis_expansion_degree)

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
        print(f"Converged to continuation value {continuation_values[-1]} at {num_paths} paths with variance {np.var(continuation_values)}.")
        return num_paths, num_paths_vals, change_vals


def estimate_optimal_basis_degree(basis_expansion: Callable, degree_range: range) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    # Parameters -- all paths will use these
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    r = 0.04
    S0 = 100.
    tau = 1.0
    K = 90.

    total_num_paths = 10000
    S, v = generate_heston_paths(tau, kappa, theta, sigma, rho, v0, S0, r, N = 1000, M = total_num_paths, num_parallel_procs=10)

    # Evaluate many degree expansions in parallel
    with Pool(min(len(degree_range), cpu_count())) as pool:
        results = pool.starmap(estimate_convergence_path, [(basis_expansion, S, K, r, v, tau, degree) for degree in degree_range])

    lower_limit = min(degree_range)
    stopping_points = [(i + lower_limit, result[0]) for i, result in enumerate(results) if result is not None]

    # Find best stopping point & expansion degree 
    return min(stopping_points, key = lambda x: x[1]), stopping_points # (degree, num_paths)


def estimate_optimal_time_steps(basis_expansion: Callable, basis_expansion_degree: int, time_step_range: range, M: int = 10000, tol: float = 1e-2) -> tuple[int, float, float, list[float]]:
    # Heston params
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7

    # Hull-White params
    a = 0.1
    b = 0.03
    eta = 0.01
    r0 = 0.03

    # Option params
    S0 = 100
    v0 = 0.04
    r0 = 0.03
    K = 90
    tau = 0.5
    S_max = 200
    v_max = 1.0
    r_max = 0.1
    N_S, N_v, N_r = 20, 10, 10
    M_time = 100

    target_value = finite_differencing.heston_hull_white_ADI("put", S0, v0, r0, K, S_max, v_max, r_max, N_S, N_v, N_r, M_time, tau, kappa, theta, sigma, rho, a, b, eta)
    values = []

    # Convergence: estimate lower bound of option continuation value
    current_val = np.inf
    for num_timesteps in time_step_range:
        S, v, r = generate_heston_hull_white_paths(tau, kappa, theta, sigma, rho, v0, S0, r0, a, b, eta, num_timesteps, M)
        current_val = estimate_continuation_value(S, K, r, v, tau, "put", N = basis_expansion_degree, basis_expansion=basis_expansion)
        values.append(current_val)

        print(f"Current number of timesteps: {num_timesteps}")
        print(f"Estimated value: {current_val}")
        print(f"Error: {abs(current_val - target_value)}\n")
        if abs(current_val - target_value) < tol:
            return num_timesteps, current_val, target_value, values
    
    print("Did not converge!")
    raise ValueError("LSM did not converge! Make sure M isn't to large (overfitting) and that time_step_range explores enough of the space.")

if __name__ == "__main__":
    # print(estimate_optimal_basis_degree(basis_expansions.laguerre_basis, range(1, 6)))
    # Note: Eventually, you reach a point where adding additional degrees to the basis expansion doesn't do anything. Choose the smallest order to avoid too much variance.

    # Estimate the optimal number of time steps to match finite difference solution
    time_step_range = range(10, 10000, 10)
    num_timesteps_optimal, final_option_val, target_value, values = estimate_optimal_time_steps(basis_expansions.polynomial_basis, 3, time_step_range, M = 10000) # polynomial basis 3 with 10k paths seems to converge; 20k seems to overfit; Laguerre just trends downward indefinitely
    # Generally, as the exercise becomes less discrete and more continuous, we start to match the finite difference solution.

    # Plot results
    plt.plot(time_step_range[:len(values)], values, label="LSM Solution")
    plt.axhline(y=target_value, color='black', linestyle='--', linewidth=2, label="Finite Difference Solution")
    plt.title("LSM Convergence over N")
    plt.xlabel("N")
    plt.ylabel("Option Value")
    plt.legend()
    plt.show()
