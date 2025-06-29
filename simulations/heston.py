import numpy as np

def generate_heston_paths(tau, kappa, theta, sigma, rho, v0, S0, r, N, M):
    """
    Inputs:
     - tau   : time of simulation
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - rho   : correlation between asset returns and variance
     - S0, v0: initial parameters for asset and variance
     - r     : interest rate
     - N     : number of time steps
     - M     : number of asset paths
    
    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # initialise other parameters
    dt = tau/N
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])
    
    # Instantiate arrays
    S = np.full(shape=(N + 1, M), fill_value=S0)
    v = np.full(shape=(N + 1, M), fill_value=v0)

    # Sample correlated Brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N, M))
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp((r - 0.5 * v[i - 1])*dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
        v[i] = np.maximum(v[i - 1] + kappa*(theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1], 0)
    
    return S, v