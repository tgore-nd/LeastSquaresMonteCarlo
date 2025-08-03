import numpy as np
from typing import Literal


def cn_max_payoff(option_type: Literal["call", "put"], S0: float, S_max: float, K: float, tau: float, r: float, sigma: float, N: int, M: int, early_exercise: bool = True) -> float:
    """
    option_type: call or put
    S_max: the largest price considered in the discretized S grid
    K: strike price
    tau: time to expiration (in years)
    r: risk-free rate
    sigma: annualized volatility of the underlying
    N: number of discrete steps to divide the time interval
    M: number of discrete steps to divide the stock price axis
    early_exercise: whether or not to include early exercise functionality
    """
    dt = tau / M
    S = np.linspace(0, S_max, N + 1)

    # Terminal payoff
    if option_type == "put":
        V = np.maximum(K - S, 0)
    elif option_type == "call":
        V = np.maximum(S - K, 0)
    else:
        raise ValueError('option_type must be either "call" or "put"')

    # Coefficients for tridiagonal matrix
    alpha = 0.25 * dt * (sigma**2 * (np.arange(N + 1)**2) - r * np.arange(N + 1))
    beta = -0.5 * dt * (sigma**2 * (np.arange(N + 1)**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (np.arange(N + 1)**2) + r * np.arange(N + 1))

    # Tridiagonal matrices A and B
    A = np.zeros((N - 1, N - 1))
    B = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        if i > 0:
            A[i, i - 1] = -alpha[i + 1]
            B[i, i - 1] = alpha[i + 1]
        A[i, i] = 1 - beta[i + 1]
        B[i, i] = 1 + beta[i + 1]
        if i < N-2:
            A[i, i + 1] = -gamma[i + 1]
            B[i, i + 1] = gamma[i + 1]

    # Backward time stepping
    for m in range(M):
        rhs = B @ V[1:-1]
        # Apply boundary conditions
        rhs[0] += alpha[1] * V[0]
        rhs[-1] += gamma[N - 1] * V[N]

        # Solve tridiagonal system
        V_inner = np.linalg.solve(A, rhs)

        if option_type == "put":
            # Max with payoff
            if early_exercise:
                V[1:-1] = np.maximum(V_inner, K - S[1:-1])
            else:
                V[1:-1] = V_inner

            # Boundary conditions
            V[0] = K * np.exp(-r * dt * (M - m))
            V[-1] = 0
        else:
            # Max with payoff
            if early_exercise:
                V[1:-1] = np.maximum(V_inner, S[1:-1] - K)
            else:
                V[1:-1] = V_inner

            # Boundary conditions
            V[0] = 0
            V[-1] = S_max - K * np.exp(-r * dt * (M - m))

    return np.interp(S0, S, V)


if __name__ == "__main__":
    S_max = 100
    K = 50
    S0 = 45
    tau = 1.0
    r = 0.05
    sigma = 0.25
    M = 500  # time steps
    N = 100  # price steps

    V = cn_max_payoff("put", S0, S_max, K, tau, r, sigma, N, M)
    print(V)