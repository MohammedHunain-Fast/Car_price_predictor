import numpy as np
from linear_regression import LinearRegression


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n - n_features - 1 <= 0:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray, n_features: int, label: str = "Model") -> dict:
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    metrics = {
        "MAE":      mae(y_true, y_pred),
        "RMSE":     rmse(y_true, y_pred),
        "R2":       r2_score(y_true, y_pred),
        "Adj_R2":   adjusted_r2(y_true, y_pred, n_features),
        "MAPE (%)": mape(y_true, y_pred),
    }
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  MAE        : ${metrics['MAE']:>12,.2f}")
    print(f"  RMSE       : ${metrics['RMSE']:>12,.2f}")
    print(f"  R2         :   {metrics['R2']:>10.4f}")
    print(f"  Adj-R2     :   {metrics['Adj_R2']:>10.4f}")
    print(f"  MAPE       :   {metrics['MAPE (%)']:>10.2f} %")
    print(f"{'='*55}")
    return metrics


def kfold_cv(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    r2_scores, rmse_scores = [], []
    for fold_idx, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[i] for i in range(k) if i != fold_idx])
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        y_pred_log = model.predict(X_val)
        y_val_orig = np.expm1(y_val)
        y_pred_orig = np.clip(np.expm1(y_pred_log), 0, None)
        r2_scores.append(r2_score(y_val_orig, y_pred_orig))
        rmse_scores.append(rmse(y_val_orig, y_pred_orig))
        print(f"  Fold {fold_idx+1}/{k}  R2={r2_scores[-1]:.4f}  RMSE=${rmse_scores[-1]:,.0f}")
    result = {
        "R2_mean":   float(np.mean(r2_scores)),
        "R2_std":    float(np.std(r2_scores)),
        "RMSE_mean": float(np.mean(rmse_scores)),
        "RMSE_std":  float(np.std(rmse_scores)),
    }
    print(f"\n  Cross-Validation Summary (k={k})")
    print(f"    R2   : {result['R2_mean']:.4f}  +/- {result['R2_std']:.4f}")
    print(f"    RMSE : ${result['RMSE_mean']:,.0f}  +/- ${result['RMSE_std']:,.0f}")
    return result
