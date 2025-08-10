import numpy as np
import time
from typing import Literal
from scipy.interpolate import RegularGridInterpolator


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


def heston_hull_white_ADI(option_type: Literal["call", "put"], S0: float, v0: float, r0: float, K: float, S_max: float, v_max: float, r_max: float, N_S: int, N_v: int, N_r: int, M_time: int, tau: float, kappa: float, theta: float, sigma: float, rho: float, a: float, b: float, eta: float):
    """
    Parameters
    ----------
    option_type: call or put
    S0: the current actual price of the underlying
    r0: initial interest rate
    v0: initial variance
    K: strike price of the option
    S_max: maximum value of asset price grid S
    v_max: maximum value of volatility grid in Heston model
    r_max: maximum value of interest rate grid in Hull-White model
    N_S: number of grid steps in the asset price direction
    N_v: number of grid steps in the volatility direction
    N_r: number of grid steps in the interest rate direction
    M_time: number of time steps used in time discretization
    tau: time until expiration (in years)
    kappa: speed of mean reversion process in Heston model
    theta: long term mean of variance process
    sigma: volatility of volatility in Heston model
    rho: correlation between asset price and variance processes
    a: speed of mean reversion of the short rate in Hull-White model
    b: long-term mean level of the short rate
    eta: volatility of the short rate
    r_discount: discount rate or function used for present value calculations
    """
    if option_type == "call":
        payoff = lambda s: np.maximum(s - K, 0)
    elif option_type == "put":
        payoff = lambda s: np.maximum(K - s, 0)
    else:
        raise ValueError('option_type must be either "call" or "put"')
    
    dS, dv, dr, dt = S_max / N_S, v_max / N_v, r_max / N_r, tau / M_time

    # Initialize discrete axes
    S_grid = np.linspace(0, S_max, N_S + 1)
    v_grid = np.linspace(0, v_max, N_v + 1)
    r_grid = np.linspace(0, r_max, N_r + 1)

    # Terminal payoff at t=T
    V = np.zeros((N_S + 1, N_v+1, N_r + 1))
    for i, S in enumerate(S_grid):
        for j in range(N_v + 1):
            for k in range(N_r + 1):
                V[i, j, k] = payoff(S)

    # Thomas solver
    def thomas_solver(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
        """Solves tridiagonal system a x_{i-1} + b x_i + c x_{i+1} = d via the Thomas algorithm."""
        n = len(d)
        cp = np.empty(n)
        dp = np.empty(n)
        x = np.empty(n)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i]*cp[i-1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i-1]) / denom
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i]*x[i + 1]
        return x
    
    # ADI Step
    def step_direction(V: np.ndarray, dir: Literal["S", "v", "r"]):
        # Build tridiagonal coefficients along given dir for each fixed other indices
        # Preallocate new V
        V_new = np.zeros_like(V)
        if dir == 'S':
            for j in range(N_v + 1):
                for k in range(N_r + 1):
                    # Extract slice along S
                    f_arr = V[:, j, k]
                    # Build tri-diagonals
                    a_arr = np.zeros(N_S + 1)
                    b_arr = np.zeros(N_S + 1)
                    c_arr = np.zeros(N_S + 1)
                    d_arr = np.zeros(N_S + 1)
                    for i in range(1, N_S):
                        S = S_grid[i]
                        v = v_grid[j]
                        sigma2 = v
                        A = 0.5 * dt * (sigma2 * S * S / (dS ** 2) - r0 * S / dS)
                        B = 1 + dt * (sigma2 * S * S / (dS ** 2) + r0)
                        C = 0.5 * dt * (sigma2 * S * S / (dS ** 2) + r0 * S / dS)
                        a_arr[i], b_arr[i], c_arr[i] = -A, B, -C
                        # explicit contributions moved RHS
                        d_arr[i] = f_arr[i]
                    # boundary
                    b_arr[0] = 1
                    d_arr[0] = payoff(S_grid[0])
                    b_arr[-1] = 1
                    d_arr[-1] = 0
                    # solve tridiagonal
                    V_new[:, j, k] = thomas_solver(a_arr, b_arr, c_arr, d_arr)
        elif dir == 'v':
            for i in range(N_S + 1):
                for k in range(N_r + 1):
                    f_arr = V[i, :, k]
                    a_arr = np.zeros(N_v + 1)
                    b_arr = np.zeros(N_v + 1)
                    c_arr = np.zeros(N_v + 1)
                    d_arr = np.zeros(N_v + 1)
                    for j in range(1, N_v):
                        v = v_grid[j]
                        A = 0.5 * dt * (sigma ** 2 * v / (dv ** 2) - kappa * (theta - v) / dv)
                        B = 1 + dt * (sigma ** 2 * v / (dv ** 2) + kappa)
                        C = 0.5*dt*(sigma ** 2 * v / (dv ** 2) + kappa * (theta - v) / dv)
                        a_arr[j], b_arr[j], c_arr[j] = -A, B, -C
                        d_arr[j] = f_arr[j]
                    b_arr[0] = 1
                    d_arr[0] = f_arr[0]
                    b_arr[-1] = 1
                    d_arr[-1] = f_arr[-1]
                    V_new[i, :, k] = thomas_solver(a_arr, b_arr, c_arr, d_arr)
        elif dir == 'r':
            for i in range(N_S + 1):
                for j in range(N_v + 1):
                    f_arr = V[i, j, :]
                    a_arr = np.zeros(N_r + 1)
                    b_arr = np.zeros(N_r + 1)
                    c_arr = np.zeros(N_r + 1)
                    d_arr = np.zeros(N_r + 1)
                    for k in range(1, N_r):
                        r = r_grid[k]
                        A = 0.5 * dt * (eta ** 2 / (dr ** 2) - a * (b - r) / dr)
                        B = 1 + dt * (eta ** 2 / (dr ** 2) + a)
                        C = 0.5 * dt * (eta ** 2 / (dr ** 2) + a * (b - r) / dr)
                        a_arr[k], b_arr[k], c_arr[k] = -A, B, -C
                        d_arr[k] = f_arr[k]
                    b_arr[0] = 1
                    d_arr[0] = f_arr[0]
                    b_arr[-1] = 1
                    d_arr[-1] = f_arr[-1]
                    V_new[i, j, :] = thomas_solver(a_arr, b_arr, c_arr, d_arr)
        return V_new
    
    # time stepping backwards
    for m in range(M_time):
        # 1: implicit in S, explicit others
        V = step_direction(V, dir='S')
        # 2: implicit in v, explicit others
        V = step_direction(V, dir='v')
        # 3: implicit in r, explicit others
        V = step_direction(V, dir='r')
        # Early exercise
        V = np.maximum(V, payoff(S_grid)[:, None, None])
    
    interp = RegularGridInterpolator(
        (S_grid, v_grid, r_grid), 
        V, bounds_error=False
    )

    value_at_initial = interp([[S0, v0, r0]])[0]

    return value_at_initial


if __name__ == "__main__":
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

    start = time.perf_counter()
    # V = cn_max_payoff("put", S0, S_max, K, tau, 0.03, sigma, N_S, M_time)
    V = heston_hull_white_ADI("put", S0, v0, r0, K, S_max, v_max, r_max, N_S, N_v, N_r, M_time, tau, kappa, theta, sigma, rho, a, b, eta)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start} seconds")
    print(f"Option value: {V}")