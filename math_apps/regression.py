import numpy as np

class LinearRegressor:
    def __init__(self):
        self.parameters = None
    
    def fit(self, X: np.array, y: np.ndarray, use_cholesky: bool = True) -> None:
        """Fit the LinearRegressor to some data."""
        # Add intercept term: X = [1, x1, ...]
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        if use_cholesky:
            A = X.T @ X # LHS of normal equation (times beta)
            b = X.T @ y # RHS of normal equation

            # Compute cholesky factorization: A = L @ L.T
            L = np.linalg.cholesky(A)

            # Solve Lz = b
            z = np.linalg.solve(L, b)

            self.parameters = np.linalg.solve(L.T, z)
            return

        # Compute parameters via normal equation
        self.parameters = np.linalg.inv(X.T @ X) @ X.T @ y
        return
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Given a feature matrix X, predict the value of the response vector y."""
        X = np.hstack(
            [np.ones((X.shape[0], 1)), X]
        )
        return X @ self.parameters