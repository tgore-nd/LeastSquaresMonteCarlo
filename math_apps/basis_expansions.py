import numpy as np
from typing import Literal


def polynomial_basis(X: np.ndarray, N: int) -> np.ndarray:
    """Take an Nth-degree polynomial basis expansion of vector X."""
    return np.column_stack([X, *[X ** n for n in range(2, N + 1)]])


def laguerre_basis(X: np.ndarray, N: int) -> np.ndarray:
    """Take an Nth-degree Laguerre basis expansion of vector X. The original feature is removed!"""
    L = np.zeros((X.shape[0], N))
    L[:, 0] = 1.0 # L0 = 1

    # Ln
    if N > 1:
        L[:, 1] = 1.0 - X
        for n in range(1, N - 1):
            # Evaluate using recurrence relation
            L[:, n + 1] = ((2 * n + 1 - X) * L[:, n] - n * L[:, n - 1]) / (n + 1)
    
    #return np.concatenate([X.T, L], axis = 0).T
    return np.column_stack([X, *[L[:, i] for i in range(L.shape[1])]])[:, 1:]


def legendre_basis(X: np.ndarray, N: int) -> np.ndarray:
    """Take an Nth-degree Legendre basis expansion of vector X. The original feature is removed!"""
    P = np.zeros((X.shape[0], N))
    P[:, 0] = 1.0 # P0 = 1

    # Pn
    if N > 1:
        P[:, 1] = X
        for n in range(1, N - 1):
            # Evaluate using recurrence relation
            P[:, n + 1] = ((2 * n + 1) * X * P[:, n] - n * P[:, n - 1]) / (n + 1)
    
    return np.column_stack([X, *[P[:, i] for i in range(P.shape[1])]])[:, 1:]


def hermite_basis(X: np.ndarray, N: int, type: Literal["probabilist", "physicist"] = "probabilist") -> np.ndarray:
    """Take an Nth-degree Hermite basis expansion of vector X. The original feature is removed!"""
    H = np.zeros((X.shape[0], N))
    H[:, 0] = 1.0

    if N > 1:
        if type == "probabilist":
            H[:, 1] = X
            for n in range(1, N - 1):
                # Evaluate using recurrence relation
                H[:, n + 1] = X * H[:, n] - n * H[:, n - 1]
        elif type == "physicist":
            H[:, 1] = 2 * X
            for n in range(1, N - 1):
                H[:, n + 1] = 2 * X * H[:, n] - 2 * n * H[:, n - 1]
        else:
            raise ValueError("Argument 'type' must be either 'probabilist' or 'physicist'")
    
    return np.column_stack([X, *[H[:, i] for i in range(H.shape[1])]])[:, 1:]


if __name__ == "__main__":
    X = np.array([1.08, 1.07, 0.97, 0.77, 0.84])
    X_exp = hermite_basis(X, 2)
    print(X_exp)