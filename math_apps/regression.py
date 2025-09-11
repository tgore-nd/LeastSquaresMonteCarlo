import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Literal


class LinearRegressor:
    def __init__(self, l: float = 0.0):
        self.parameters = None
        self.l = l
    
    def fit(self, X: np.ndarray, y: np.ndarray, use_cholesky: bool = True) -> None:
        """Fit the LinearRegressor to some data."""
        # Add intercept term: X = [1, x1, ...]
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        n_features = X.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0 # don't regularize intercept
        xtx = X.T @ X + I * self.l

        if self._is_singular(xtx): # check if singular
            self.parameters = np.linalg.pinv(xtx) @ X.T @ y
            return

        if use_cholesky:
            A = X.T @ X + I * self.l # LHS of normal equation (times beta)
            b = X.T @ y # RHS of normal equation

            # Compute Cholesky factorization: A = L @ L.T
            L = np.linalg.cholesky(A)

            # Solve Lz = b
            z = np.linalg.solve(L, b)

            self.parameters = np.linalg.solve(L.T, z)
            return

        # Compute parameters via normal equation
        self.parameters = np.linalg.inv(xtx) @ X.T @ y
        return
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Given a feature matrix X, predict the value of the response vector y."""
        X = np.hstack(
            [np.ones((X.shape[0], 1)), X]
        )
        return X @ self.parameters
    
    @staticmethod
    def _is_singular(A: np.ndarray, tol=1e-12) -> bool:
        A = np.asarray(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Must be a square matrix")
        # Determinant
        if abs(np.linalg.det(A)) < tol:
            return True
        # Rank
        if np.linalg.matrix_rank(A, tol=tol) < A.shape[0]:
            return True
        # Condition number
        if np.linalg.cond(A) > 1/np.finfo(A.dtype).eps:
            return True
        return False
    

class DecisionTreeRegressor:
    """A CART decision tree regressor implemented using only NumPy."""
    def __init__(self, max_depth: int = 4, min_samples_split: int = 5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        n, d = X.shape
        node = {}

        # Leaf value = mean target
        node['value'] = y.mean()

        # Stopping criteria
        if depth >= self.max_depth or n < self.min_samples_split:
            # Force the recursion to stop
            return node

        best_err = np.inf
        best_feat = None
        best_th = None

        y0 = y.astype(float)
        y0_sq = y0 ** 2 # needed for variance

        for feat in range(d):
            # Select sorted feature and response
            xs = X[:, feat]
            order = np.argsort(xs)
            xs_sorted = xs[order]
            y_sorted = y0[order]
            y_sq_sorted = y0_sq[order]

            # Candidate splits: midpoints between unique values
            diffs = xs_sorted[1:] != xs_sorted[:-1] # marks whether neighboring values are different
            if not np.any(diffs):
                continue
            thresholds = (xs_sorted[:-1] + xs_sorted[1:]) / 2 # candidate split thresholds
            idxs = np.nonzero(diffs)[0] + 1 # gives the indices where changes happen, so a split is possible

            # Cumulative sums
            cum_y = np.cumsum(y_sorted)
            cum_y_sq = np.cumsum(y_sq_sorted)

            # Consider each split
            left_count = idxs
            right_count = n - left_count

            # All samples that would go left/right if split here
            left_sum = cum_y[idxs - 1]
            right_sum = cum_y[-1] - left_sum

            left_sq = cum_y_sq[idxs - 1]
            right_sq = cum_y_sq[-1] - left_sq

            # Compute MSE for left + right partitions
            mse_left = left_sq / left_count - (left_sum / left_count) ** 2
            mse_right = right_sq / right_count - (right_sum / right_count) ** 2
            weighted_err = mse_left * left_count + mse_right * right_count

            # Pick best threshold
            loc = np.nanargmin(weighted_err)
            if weighted_err[loc] < best_err:
                best_err = weighted_err[loc]
                best_feat = feat
                best_th = thresholds[loc]

        if best_err == np.inf or best_feat is None or best_th is None:
            return node # no valid split

        node['feature'] = int(best_feat)
        node['threshold'] = float(best_th)

        # Partition samples
        mask = X[:, best_feat] <= best_th
        node['left'] = self._build_tree(X[mask], y[mask], depth + 1)
        node['right'] = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X: np.ndarray):
        out = np.empty(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            out[i] = self._predict_one(self.tree, x)
        return out

    def _predict_one(self, node, x):
        if 'feature' not in node:
            return node['value']
        fea = node['feature']
        th = node['threshold']
        return self._predict_one(node['left' if x[fea] <= th else 'right'], x)


class RandomForestRegressor:
    def __init__(self, n_estimators: int = 10, max_depth: int = 5, max_features: Literal["sqrt", "log2"] | int | None = 'sqrt', min_samples_split: int = 2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = cpu_count()
        self.trees = []
        self.features_idx = []

    def _get_max_features(self, n_features: int):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        elif self.max_features is None:
            return n_features
        else:
            raise ValueError("Invalid max_features")

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        max_feats = self._get_max_features(n_features)

        # Pre-seed each task with a random state to make results deterministic
        seeds = np.random.randint(0, int(1e9), size=self.n_estimators)

        with Pool(processes=self.n_jobs) as pool:
            results = pool.map(
                partial(
                    self._train_single_tree,
                    X, y,
                    self.max_depth,
                    self.min_samples_split,
                    max_feats,
                ),
                seeds
            )

        self.trees, self.features_idx = zip(*results)

    def predict(self, X: np.ndarray):
        preds = np.zeros((X.shape[0], len(self.trees)))
        # Iterate through trees
        for i, (tree, features) in enumerate(zip(self.trees, self.features_idx)):
            preds[:, i] = tree.predict(X[:, features])
        return np.mean(preds, axis=1) # average to produce prediction
    
    @staticmethod
    def _train_single_tree(X: np.ndarray, y: np.ndarray, max_depth: int, min_samples_split: int, max_feats, random_state=None):
        np.random.seed(random_state)
        n_samples, n_features = X.shape

        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True) # randomly choose subset
        X_sample = X[indices]
        y_sample = y[indices]

        # Feature bagging
        features = np.random.choice(n_features, max_feats, replace=False)
        X_sample_sub = X_sample[:, features]

        # Train tree
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
        tree.fit(X_sample_sub, y_sample)
        return tree, features
