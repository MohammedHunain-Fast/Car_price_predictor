import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

from linear_regression import LinearRegression
from evaluation import evaluate, kfold_cv


class Scaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


def train_and_evaluate(X, y, feature_names, test_size=0.20, cv_folds=5, random_state=42, save_path="model_weights.npz"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"\n[split] Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

    scaler = Scaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("\n[fit]   Fitting Custom OLS Linear Regression ...")
    ols = LinearRegression()
    ols.fit(X_train_sc, y_train)
    y_pred_ols = ols.predict(X_test_sc)
    ols_metrics = evaluate(y_test, y_pred_ols, len(feature_names), "Custom OLS Linear Regression")

    print("\n[fit]   Fitting Ridge Regression (sklearn baseline) ...")
    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(X_train_sc, y_train)
    y_pred_ridge = ridge.predict(X_test_sc)
    ridge_metrics = evaluate(y_test, y_pred_ridge, len(feature_names), "Ridge Regression (sklearn, a=1)")

    print("\n[fit]   Fitting Lasso Regression (sklearn baseline) ...")
    lasso = Lasso(alpha=0.001, max_iter=10_000, random_state=random_state)
    lasso.fit(X_train_sc, y_train)
    y_pred_lasso = lasso.predict(X_test_sc)
    lasso_metrics = evaluate(y_test, y_pred_lasso, len(feature_names), "Lasso Regression (sklearn, a=0.001)")

    print(f"\n[cv]    Running {cv_folds}-Fold Cross-Validation on Custom OLS ...")
    X_all_sc = scaler.fit_transform(X)
    cv_results = kfold_cv(X_all_sc, y, k=cv_folds)

    print("\n[feat]  Top-10 Features by |OLS Coefficient|:")
    coef_abs = np.abs(ols.coefficients_)
    top10_idx = np.argsort(coef_abs)[::-1][:10]
    for rank, i in enumerate(top10_idx, 1):
        print(f"  {rank:>2}. {feature_names[i]:<45s}  coef={ols.coefficients_[i]:+.4f}")

    np.savez(save_path,
             intercept=np.array([ols.intercept_]),
             coefficients=ols.coefficients_,
             feature_names=np.array(feature_names),
             scaler_mean=scaler.mean_,
             scaler_std=scaler.std_)
    print(f"\n[save]  Model weights saved to '{save_path}'.")

    return {
        "ols": ols, "ridge": ridge, "lasso": lasso,
        "scaler": scaler, "feature_names": feature_names,
        "X_test": X_test, "y_test": y_test,
        "X_test_sc": X_test_sc, "y_pred_ols": y_pred_ols,
        "ols_metrics": ols_metrics, "ridge_metrics": ridge_metrics,
        "lasso_metrics": lasso_metrics, "cv_results": cv_results,
    }


def predict_price(input_dict: dict, weights_path: str = "model_weights.npz") -> float:
    data = np.load(weights_path, allow_pickle=True)
    intercept = float(data["intercept"][0])
    coefficients = data["coefficients"]
    feature_names = list(data["feature_names"])
    scaler_mean = data["scaler_mean"]
    scaler_std = data["scaler_std"]

    x = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        if name in input_dict:
            x[i] = float(input_dict[name])

    x_sc = (x - scaler_mean) / scaler_std
    log_price = intercept + coefficients @ x_sc
    return float(max(np.expm1(log_price), 0.0))
