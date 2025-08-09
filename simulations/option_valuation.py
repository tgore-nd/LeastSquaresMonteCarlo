import numpy as np
import time
import itertools
from typing import Literal, Callable, Type
from math_apps import regression, basis_expansions
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from heston import generate_heston_paths, generate_heston_hull_white_paths


def get_exercise_values(S: np.ndarray, K: float, type: Literal["call", "put"]) -> tuple[np.ndarray, np.ndarray]:
    """Get the value from exercising the option at each time step."""
    # Define option value function
    if type.lower() == "call":
        def option_value(S: np.ndarray, K: float):
            return np.maximum(S - K, np.zeros_like(S), dtype=np.float64), S > K
    elif type.lower() == "put":
        def option_value(S: np.ndarray, K: float):
            return np.maximum(K - S, np.zeros_like(S), dtype=np.float64), K > S
    else:
        raise ValueError("Argument 'type' must be either 'call' or 'put'")
    
    # Get exercise value at each time step
    return option_value(S, K)


def estimate_cash_flow_matrix(S: np.ndarray, K: float, r: float | np.ndarray, v: np.ndarray | None, tau: float, type: Literal["call", "put"], basis_expansion: Callable, regressor_class: Callable, N: int, include_t0_column: bool = True) -> np.ndarray:
    """Return a matrix of estimated cashflows for each path."""
    dt = tau / (S.shape[1] - 3) # a parameter to manipulate the interest rate later; subtract 3 to adjust for unused columns
    # Get discount factors
    if isinstance(r, float):
        r_full = np.full_like(S, r, dtype=np.float64)
    elif isinstance(r, np.ndarray):
        r_full = r
    else:
        raise ValueError("r was not of type float or NumPy array")

    # Estimate continuation values
    continuation_values = np.zeros(S.shape)
    exercise_values, option_ITM = get_exercise_values(S, K, type)
    continuation_values[:, S.shape[1] - 1] = exercise_values[:, S.shape[1] - 1]

    def regression_step(t: int):
        regressor = regressor_class()

        # Only use paths where the option is ITM at t
        X = np.hstack([basis_expansion(arr[:, t][option_ITM[:, t]], N) for arr in (S, r, v) if isinstance(arr, np.ndarray)])

        if isinstance(r, np.ndarray) or isinstance(v, np.ndarray):
            interactions = np.array([a * b for a, b in itertools.combinations([arr[:, t][option_ITM[:, t]] for arr in (S, r, v) if isinstance(arr, np.ndarray)], 2)]).T
            X = np.hstack([X, interactions])

        y = exercise_values[:, t + 1][option_ITM[:, t]] * np.exp(-r_full[:, t][option_ITM[:, t]] * dt) # continuation value at t = exercise value at t + 1; t = 1 since we are computing the value one step in the future

        regressor.fit(X, y)
        continuation_values[:, t][option_ITM[:, t]] = regressor.predict(X)

    # Multithreading
    with ThreadPoolExecutor(cpu_count()) as executor:
        executor.map(regression_step, range(S.shape[1] - 2, 0, -1))

    continuation_values = np.maximum(continuation_values, np.zeros_like(continuation_values))

    # Stopping points: exercise >= continuation
    stopping_rules = ((exercise_values >= continuation_values) & ((exercise_values != 0) | (continuation_values != 0)))[:, 1:] # control for false positive when both matrices are zero

    # Modify for earliest stopping point
    for i, row in enumerate(stopping_rules):
        idx = np.where(row)[0]
        if idx.size > 0:
            stopping_rules[i, idx[0] + 1:] = False

    exercise_values[:, 1:][~stopping_rules] = 0. # zero out elements where we don't exercise
    
    # Discount cashflows back to t = 0
    discount_factors = np.hstack([np.ones((S.shape[0], 1)), np.exp(-np.cumsum(r_full * dt, axis=1))[:, :-1]]) # at t = 0, no discount
    exercise_values *= discount_factors

    if not include_t0_column: 
        return exercise_values[:, 1:]

    return exercise_values


def estimate_continuation_value(S: np.ndarray, K: float, r: float | np.ndarray, v: np.ndarray | None, tau: float, type: Literal["call", "put"], basis_expansion: Callable = basis_expansions.polynomial_basis, regressor_class: Type[regression.LinearRegressor] | Type[regression.DecisionTreeRegressor] | Type[regression.RandomForestRegressor] = regression.LinearRegressor, N: int = 2) -> float:
    """Estimate the continuation value of an option using least-squares Monte Carlo (LCM). Use an Nth-degree basis_expansion."""
    cash_flow_matrix = estimate_cash_flow_matrix(S, K, r, v, tau, type, basis_expansion, regressor_class, N, include_t0_column=False)

    return np.sum(np.mean(cash_flow_matrix, axis=0)) # average each path (already discounted), then add all averages


def paper_example() -> None:
    """Estimate the continuation value of a put with strike K = 1.10 and risk-free rate r = 6%
    
    Exactly replicates the value predicted by Longstaff, Schwartz (2001)."""
    K = 1.10
    r = 0.06
    tau = 1.0
    S = np.array([
        [1.00, 1.09, 1.08, 1.34],
        [1.00, 1.16, 1.26, 1.54],
        [1.00, 1.22, 1.07, 1.03],
        [1.00, 0.93, 0.97, 0.92],
        [1.00, 1.11, 1.56, 1.52],
        [1.00, 0.76, 0.77, 0.90],
        [1.00, 0.92, 0.84, 1.01],
        [1.00, 0.88, 1.22, 1.34]
    ]) # M > N
    
    print(estimate_continuation_value(S, K, r, None, tau, "put")) # this exactly equals the value in the paper for squared polynomial basis expansion
    # print(estimate_continuation_value(S, K, r, None, tau, "put", regressor_class=regression.DecisionTreeRegressor)) # the decision tree also converges nicely
    # print(estimate_continuation_value(S, K, r, None, tau, "put", regressor_class=regression.RandomForestRegressor))

if __name__ == "__main__":
    paper_example()
    # Parameters
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

    start = time.perf_counter()
    S, v, r = generate_heston_hull_white_paths(tau, kappa, theta, sigma, rho, v0, S0, r0, a, b, eta, 500, 20000)
    # S, v = generate_heston_paths(tau, kappa, theta, sigma, rho, v0, S0, 0.03, 40, 10000) # N = 40 seems to be good for basis degree = 5
    print(estimate_continuation_value(S, K, r, v, tau, "put", N = 5, basis_expansion=basis_expansions.polynomial_basis, regressor_class=regression.LinearRegressor))
    end = time.perf_counter()
    print(f"Elapsed time: {end - start} seconds")