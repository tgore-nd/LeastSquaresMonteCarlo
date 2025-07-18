import numpy as np
from multiprocessing import Pool


def simulate_chunk(args: tuple[np.ndarray, float, float, float, float, float, float, float, int]):
    Z, S0, v0, r, dt, kappa, sigma, theta, N = args # for some reason, multiprocessing.Pool can't handle more than 3-4 arguments, so we unpack them as a tuple
    num_paths_in_chunk = Z.shape[1]
    S = np.full(shape=(N + 1, num_paths_in_chunk), fill_value=S0)
    v = np.full(shape=(N + 1, num_paths_in_chunk), fill_value=v0)
    for i in range(1, N + 1):
        S[i, :] = S[i - 1] * np.exp((r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
        v[i, :] = np.maximum(v[i - 1] + kappa*(theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1], 0)
     
    return S


def generate_heston_paths(tau: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, S0: float, r: float, N, M, num_parallel_procs: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
     - tau    : time of simulation in years
     - kappa  : rate of mean reversion in variance process
     - theta  : long-term mean of variance process
     - sigma  : vol of vol / volatility of variance process
     - rho    : correlation between asset returns and variance
     - S0, v0 : initial parameters for asset and variance
     - r      : interest rate
     - N      : number of time steps
     - M      : number of asset paths
    
    Outputs:
    - asset prices over time
    - variance over time
    """
    # initialise other parameters
    dt = tau/N
    mu = np.array([0, 0])
    cov = np.array([[1, rho], # covariance matrix for correlated Brownian motions
                    [rho, 1]])


    # Sample correlated Brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N, M))
    chunks = np.array_split(np.arange(M), num_parallel_procs)
    args = [(Z[:, idxs, :], S0, v0, r, dt, kappa, sigma, theta, N) for idx, idxs in enumerate(chunks)]
    
    with Pool(num_parallel_procs) as pool:
        results = pool.map(simulate_chunk, args)
    
    return np.concatenate(results, axis=0).T # take the transpose so this works more nicely with the regression seen in Longstaff-Schwartz

if __name__ == "__main__":
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    r = 0.01
    S0 = 100.
    tau = 1.0
    # Test getting price matrix
    print(generate_heston_paths(tau, kappa, theta, sigma, rho, v0, S0, r, 390, 10000))