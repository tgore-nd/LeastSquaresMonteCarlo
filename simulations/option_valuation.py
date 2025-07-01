import numpy as np
import polars as pl
from simulations import heston
from typing import Literal
from math_apps import regression

def get_exercise_values(S: np.array, K: float, r: float, type: Literal["call", "put"]) -> np.array:
    # Define option value function
    if type.lower() == "call":
        def option_value(S: np.array, K: float):
            return np.maximum(S - K, np.zeros(S.shape)) # add: risk-free rate discount!
    elif type.lower() == "put":
        def option_value(S: np.array, K: float):
            return np.maximum(K - S, np.zeros(S.shape))
    else:
        raise ValueError("Argument 'type' must be either 'call' or 'put'")
    
    # Get exercise value at each time step
    return option_value(S, K)


def estimate_continuation_values(S: np.array, K: float, r: float, type: Literal["call", "put"]) -> np.array:
    continuation_values = np.zeros(S.shape)
    exercise_values = get_exercise_values(S, K, r, type)
    continuation_values[:, S.shape[1] - 1] = exercise_values[:, S.shape[1] - 1]
    for t in range(1, S.shape[1] - 1):
        linear_regressor = regression.LinearRegressor()

        X = np.array([S[:, t]]).T # stock price at t
        y = exercise_values[:, t + 1] # continuation value at t = exercise value at t + 1

        linear_regressor.fit(X, y)
        continuation_values[:, t] = linear_regressor.predict(X)
    
    return np.maximum(continuation_values, np.zeros(continuation_values.shape))


def find_stopping_points(S: np.array):
    pass


if __name__ == "__main__":
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    r = 0.01
    S0 = 100.
    tau = 1.0

    K = 100

    # Get some sample values
    S = heston.generate_heston_paths(tau, kappa, theta, sigma, rho, v0, S0, r, 100, 10)[0]
    print(S.shape)

    print(pl.DataFrame(estimate_continuation_values(S, K, 0.08, "call")))