import numpy as np
from multiprocessing import Pool, cpu_count


def simulate_batch_heston(Z: np.ndarray, S0: float, v0: float, r: float, dt: float, kappa: float, sigma: float, theta: float, N: int):
    num_paths_in_chunk = Z.shape[1]
    S = np.full(shape=(N + 1, num_paths_in_chunk), fill_value=S0)
    v = np.full(shape=(N + 1, num_paths_in_chunk), fill_value=v0)
    for i in range(1, N + 1):
        S[i, :] = S[i - 1] * np.exp((r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
        v[i, :] = np.maximum(v[i - 1] + kappa*(theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1], 0)

    return S.T, v.T


def simulate_batch_heston_hull_white(S0: float, v0: float, r0: float, kappa: float, sigma: float, theta: float, rho: float, a: float, b: float, eta: float, tau: float, N: int, n_paths: int):
    dt = tau / (N - 3)

    S_paths = np.zeros((n_paths, N + 1))
    v_paths = np.zeros((n_paths, N + 1))
    r_paths = np.zeros((n_paths, N + 1))

    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    r_paths[:, 0] = r0

    for t in range(1, N + 1):
        dW1 = np.random.randn(n_paths) * np.sqrt(dt)
        dW2 = np.random.randn(n_paths) * np.sqrt(dt)
        dW3 = np.random.randn(n_paths) * np.sqrt(dt)

        dW2_corr = rho * dW1 + np.sqrt(1 - rho**2) * dW2

        v_prev = v_paths[:, t - 1]
        v_sqrt = np.sqrt(np.maximum(v_prev, 1e-8))
        v_next = v_prev + kappa * (theta - v_prev) * dt + sigma * v_sqrt * dW2_corr
        v_next = np.maximum(v_next, 0.0)

        r_prev = r_paths[:, t - 1]
        r_next = r_prev + a * (b - r_prev) * dt + eta * dW3

        S_prev = S_paths[:, t - 1]
        S_next = S_prev * np.exp((r_prev - 0.5 * v_prev) * dt + np.sqrt(v_prev) * dW1)

        v_paths[:, t] = v_next
        r_paths[:, t] = r_next
        S_paths[:, t] = S_next

    return S_paths, v_paths, r_paths


def generate_heston_paths(tau: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, S0: float, r: float, N: int, M: int, num_parallel_procs: int = cpu_count()) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sample paths using Bayesian-estimated parameters.
    
    Parameters
    ---------
    tau : float
        time of simulation (in years)
    kappa : float
        rate of mean reversion in variance process
    theta : float
        long-term mean of variance process
    sigma : float
        vol of vol / volatility of variance process
    rho : float
        correlation between asset returns and variance
    S0 : float
        initial parameter for asset price
    v0 : float
        initial parameter for variance
    r : float
        interest rate
    N : int
        number of time steps
    M : int
        number of asset paths
    
    Returns
    -------
    asset_price_path : np.ndarray
    variance_path : np.ndarray
    """
    # initialise other parameters
    dt = tau/N
    mu = np.array([0, 0])
    cov = np.array([[1, rho], # covariance matrix for correlated Brownian motions
                    [rho, 1]])


    # Sample correlated Brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N, M))
    chunks = np.array_split(np.arange(M), num_parallel_procs)
    
    with Pool(num_parallel_procs) as pool:
        results = pool.starmap(simulate_batch_heston, [(Z[:, idxs, :], S0, v0, r, dt, kappa, sigma, theta, N) for idx, idxs in enumerate(chunks)])
        pool.close()
        pool.join()
    
    S_all = np.vstack([res[0] for res in results])
    v_all = np.vstack([res[1] for res in results])

    return S_all, v_all # take the transpose so this works more nicely with the regression seen in Longstaff-Schwartz


def generate_heston_hull_white_paths(tau: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, S0: float, r0: float, a: float, b: float, eta: float, N: int, M: int, n_procs=cpu_count()):
    # Split work
    paths_per_proc = [M // n_procs] * n_procs
    for i in range(M % n_procs):
        paths_per_proc[i] += 1

    with Pool(n_procs) as pool:
        results = pool.starmap(simulate_batch_heston_hull_white, [(S0, v0, r0, kappa, sigma, theta, rho, a, b, eta, tau, N, n) for n in paths_per_proc])
        pool.close()
        pool.join()

    S_all = np.vstack([res[0] for res in results])
    v_all = np.vstack([res[1] for res in results])
    r_all = np.vstack([res[2] for res in results])

    return S_all, v_all, r_all


if __name__ == "__main__":
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    r = 0.01
    S0 = 100.
    tau = 1.0
    a = 0.3
    b = 0.05
    eta = 0.02

    # Test getting price matrix
    S = generate_heston_hull_white_paths(tau, kappa, theta, sigma, rho, v0, S0, r, a, b, eta, N=390, M=1000)[0]
    print(S)