import numpy as np
import time
from typing import Literal, Callable, Type
from math_apps import regression, basis_expansions
from heston import generate_heston_paths


def get_exercise_values(S: np.ndarray, K: float, type: Literal["call", "put"]) -> tuple[np.ndarray, np.ndarray]:
    """Get the value from exercising the option at each time step."""
    # Define option value function
    if type.lower() == "call":
        def option_value(S: np.ndarray, K: float):
            return np.maximum(S - K, np.zeros_like(S)), S > K
    elif type.lower() == "put":
        def option_value(S: np.ndarray, K: float):
            return np.maximum(K - S, np.zeros_like(S)), K > S
    else:
        raise ValueError("Argument 'type' must be either 'call' or 'put'")
    
    # Get exercise value at each time step
    return option_value(S, K)


def estimate_cash_flow_matrix(S: np.ndarray, K: float, r: float, type: Literal["call", "put"], basis_expansion: Callable, regressor_class: Callable, N: int, include_t0_column: bool = True) -> np.ndarray:
    """Return a matrix of estimated cashflows for each path."""
    # Estimate continuation values
    continuation_values = np.zeros(S.shape)
    exercise_values, option_ITM = get_exercise_values(S, K, type)
    continuation_values[:, S.shape[1] - 1] = exercise_values[:, S.shape[1] - 1]
    for t in range(S.shape[1] - 2, 0, -1):
        regressor = regressor_class()

        # only use paths where the option is ITM at t
        X = basis_expansion(S[:, t][option_ITM[:, t]], N) # stock price at t

        y = exercise_values[:, t + 1][option_ITM[:, t]] * np.exp(-r) # continuation value at t = exercise value at t + 1; tau = 1 since we are computing the value one step in the future

        regressor.fit(X, y)
        continuation_values[:, t][option_ITM[:, t]] = regressor.predict(X)


    continuation_values = np.maximum(continuation_values, np.zeros_like(continuation_values))
    
    # Stopping points: exercise > continuation
    stopping_rules = ((exercise_values >= continuation_values) & ((exercise_values != 0) | (continuation_values != 0)))[:, 1:] # control for false positive when both matrices are zero

    # Modify for earliest stopping point
    for i, row in enumerate(stopping_rules):
        idx = np.where(row)[0]
        if idx.size > 0:
            stopping_rules[i, idx[0] + 1:] = False

    exercise_values[:, 1:][~stopping_rules] = 0 # zero out elements where we don't exercise
    
    # Discount cashflows back to t = 0
    for t in range(exercise_values.shape[1]):
        exercise_values[:, t] *= np.exp(-r*t)
    
    if not include_t0_column: return exercise_values[:, 1:]

    return exercise_values


def estimate_continuation_value(S: np.ndarray, K: float, r: float, type: Literal["call", "put"], basis_expansion: Callable = basis_expansions.polynomial_basis, regressor_class: Type[regression.LinearRegressor] | Type[regression.DecisionTreeRegressor] | Type[regression.RandomForestRegressor] = regression.LinearRegressor, N: int = 2) -> float:
    """Estimate the continuation value of an option using least-squares Monte Carlo (LCM). Use an Nth-degree basis_expansion."""
    cash_flow_matrix = estimate_cash_flow_matrix(S, K, r, type, basis_expansion, regressor_class, N, include_t0_column=False)

    return np.sum(np.mean(cash_flow_matrix, axis=0)) # average each path (already discounted), then add all averages


def paper_example() -> None:
    """Estimate the continuation value of a put with strike K = 1.10 and risk-free rate r = 6%
    
    Exactly replicates the value predicted by Longstaff, Schwartz (2001)."""
    K = 1.10
    r = 0.06
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
    
    print(estimate_continuation_value(S, K, r, "put")) # this exactly equals the value in the paper for squared polynomial basis expansion
    print(estimate_continuation_value(S, K, r, "put", regressor_class=regression.DecisionTreeRegressor)) # the decision tree also converges nicely
    print(estimate_continuation_value(S, K, r, "put", regressor_class=regression.RandomForestRegressor))

if __name__ == "__main__":
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    r = 0.04
    S0 = 100.
    tau = 1.0
    K = 90.

    start = time.perf_counter()
    S = generate_heston_paths(tau, kappa, theta, sigma, rho, v0, S0, r, 1000, 1700)
    print(estimate_continuation_value(S, K, r, "call"))
    end = time.perf_counter()
    print(f"Elapsed time: {end - start} seconds")