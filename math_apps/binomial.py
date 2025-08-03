import numpy as np
from typing import Literal


def binomial_value(option_type: Literal["call", "put"], S0: float, K: float, r: float, tau: float, iv: float, n: int) -> np.ndarray:
    """
    option_type: call or put
    S0: current price of the underlying
    K: strike price
    r: interest rate (decimal between 0 and 1)
    tau: time until expiration (in years)
    iv: implied volatility (decimal between 0 and 1)
    n: number of timesteps
    """
    def get_stock_prices(S0, n, u, d):
        stock_prices = np.zeros((n + 1, n + 1))
        stock_prices[0, 0] = S0
        for i in range(1, n + 1):
            for j in range(i + 1):
                stock_prices[j, i] = S0 * (u**j) * (d**(i - j))
        return stock_prices

    dt = tau / n
    u = np.exp(iv * np.sqrt(dt)) # up factor
    d = 1/u # down factor
    p = (np.exp(r * dt) - d)/(u - d) # risk-neutral probability
    
    stock_prices = get_stock_prices(S0, n, u, d)
    option_values = np.zeros((n + 1, n + 1))

    # Exercise value function
    if option_type == "call":
        payoff = lambda s: np.maximum(s - K, 0)
    elif option_type == "put":
        payoff = lambda s: np.maximum(K - s, 0)
    else:
        raise ValueError('option_type must be either "call" or "put"')

    # Intrinsic values at expiration
    for j in range(n + 1): # higher j = more 'up' turns; higher (i - j), more 'down' turns
        option_values[j, n] = payoff(stock_prices[j, n])
    
    # Work backward through tree
    for i in range(n - 1, -1, -1): # total time steps
        for j in range(i + 1): # upward movements
            # Continuation value
            continuation_value = np.exp(-r * dt) * (p * option_values[j + 1, i + 1] + (1 - p) * option_values[j, i + 1]) # if there isn't an increase in the j index, we have a 'down' turn
            
            # Exercise value
            exercise_value = payoff(stock_prices[j, i])
            
            # Option value is the greater of the continuation and exercise values
            option_values[j, i] = np.maximum(continuation_value, exercise_value)        
    
    return option_values


if __name__ == "__main__":
    S0 = 45
    K = 50
    tau = 1.0
    r = 0.05
    sigma = 0.25
    n = 500 # time steps

    print(binomial_value("put", S0, K, r, tau, sigma, n)[0, 0])