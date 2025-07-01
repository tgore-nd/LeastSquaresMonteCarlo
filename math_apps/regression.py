import numpy as np

class LinearRegressor:
    def __init__(self):
        self.parameters = None
    
    def fit(self, X: np.array, y: np.array) -> None:
        """Fit the LinearRegressor to some data."""
        # Add intercept term: X = [1, x1, ...]

        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        # Compute parameters via normal equation
        self.parameters = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X: np.array) -> np.array:
        """Given a feature matrix X, predict the value of the response vector y."""
        X = np.hstack(
            [np.ones((X.shape[0], 1)), X]
        )
        return X @ self.parameters