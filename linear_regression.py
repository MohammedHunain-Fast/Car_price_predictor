import numpy as np


class LinearRegression:
    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = None
        self._beta = None

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.hstack([np.ones((n, 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        X_b = self._add_bias(X)
        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        try:
            self._beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self._beta = np.linalg.pinv(XtX) @ Xty
        self.intercept_ = self._beta[0]
        self.coefficients_ = self._beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._beta is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.array(X, dtype=float)
        X_b = self._add_bias(X)
        return X_b @ self._beta

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, p = 200, 5
    X_test = rng.standard_normal((n, p))
    true_beta = rng.standard_normal(p + 1)
    y_test = np.hstack([np.ones((n, 1)), X_test]) @ true_beta + rng.standard_normal(n) * 0.1
    model = LinearRegression()
    model.fit(X_test, y_test)
    print(f"Custom OLS sanity check")
    print(f"  True b0={true_beta[0]:.4f}  Estimated intercept={model.intercept_:.4f}")
    print(f"  R2 on training data = {model.score(X_test, y_test):.6f}")
