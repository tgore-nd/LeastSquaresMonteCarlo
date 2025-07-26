import numpy as np
from typing import Literal

# Legendre, Hermite, Spline


def polynomial_basis(X: np.ndarray, N: int, i: int = 0) -> np.ndarray:
    """Modify the ith column of the input array for an Nth-degree basis expansion."""
    return np.concatenate([X] + [X[:, i] ** n for n in range(2, N + 1)], axis = 1)


def laguerre_basis(X: np.ndarray, N: int, i: int = 0) -> np.ndarray:
    L = np.zeros((X.shape[0], N))
    L[:, 0] = 1.0 # L0 = 1

    # Ln
    if N > 1:
        L[:, 1] = 1.0 - X[:, i]
        for n in range(1, N - 1):
            # Evaluate using recurrence relation
            L[:, n + 1] = ((2 * n + 1 - X[:, i]) * L[:, n] - n * L[:, n - 1]) / (n + 1)
    
    return np.concatenate([X, L], axis = 1)


def legendre_basis(X: np.ndarray, N: int, i: int = 0) -> np.ndarray:
    P = np.zeros((X.shape[0], N))
    P[:, 0] = 1.0 # P0 = 1

    # Pn
    if N > 1:
        P[:, 1] = X[:, i]
        for n in range(1, N - 1):
            # Evaluate using recurrence relation
            P[:, n + 1] = ((2 * n + 1) * X[:, i] * P[:, n] - n * P[:, n - 1]) / (n + 1)
    
    return np.concatenate([X, P], axis = 1)


def hermite_basis(X: np.ndarray, N: int, i: int = 0, type: Literal["probabilist", "physicist"] = "probabilist"):
    H = np.zeros((X.shape[0], N))
    H[:, 0] = 1.0

    if N > 1:
        if type == "probabilist":
            H[:, 1] = X[:, i]
            for n in range(1, N - 1):
                # Evaluate using recurrence relation
                H[:, n + 1] = X[:, i] * H[:, n] - n * H[:, n - 1]
        elif type == "physicist":
            H[:, 1] = 2 * X[:, i]
            for n in range(1, N - 1):
                H[:, n + 1] = 2 * X[:, i] * H[:, n] - 2 * n * H[:, n - 1]
        else:
            raise ValueError("Argument 'type' must be either 'probabilist' or 'physicist'")
    
    return np.concatenate([X, H], axis = 1)
