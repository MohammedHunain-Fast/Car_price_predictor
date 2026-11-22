import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#0f3460",
    "axes.labelcolor":   "#e0e0e0",
    "axes.titlecolor":   "#e94560",
    "xtick.color":       "#e0e0e0",
    "ytick.color":       "#e0e0e0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#0f3460",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        120,
})

ACCENT = "#e94560"
TEAL   = "#00b4d8"
GOLD   = "#ffd60a"
PLOTS_DIR = "plots"


def _save(fig, name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [plot] Saved -> {path}")
    plt.close(fig)


def plot_msrp_distribution(df_raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MSRP Distribution", fontsize=16, fontweight="bold", color="#e94560")
    msrp = df_raw["MSRP"].dropna()
    msrp = msrp[msrp > 0]
    axes[0].hist(msrp, bins=60, color=ACCENT, edgecolor="#000", alpha=0.85)
    axes[0].set_title("Raw MSRP")
    axes[0].set_xlabel("MSRP (USD)")
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[1].hist(np.log1p(msrp), bins=60, color=TEAL, edgecolor="#000", alpha=0.85)
    axes[1].set_title("log(MSRP + 1)")
    axes[1].set_xlabel("log(MSRP + 1)")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    _save(fig, "1_msrp_distribution.png")


def plot_correlation_heatmap(df_raw: pd.DataFrame):
    num_cols = ["Engine HP", "Engine Cylinders", "Number of Doors",
                "highway MPG", "city mpg", "Popularity", "MSRP"]
    present = [c for c in num_cols if c in df_raw.columns]
    corr = df_raw[present].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Numeric Feature Correlation Matrix", fontsize=15, fontweight="bold", color="#e94560")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, linecolor="#0f3460", vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 9})
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save(fig, "2_correlation_heatmap.png")


def plot_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray, label: str = "OLS"):
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"{label} - Actual vs Predicted & Residuals", fontsize=14, fontweight="bold", color="#e94560")
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color=TEAL, edgecolors="none")
    lim_min = min(y_true.min(), y_pred.min()) * 0.9
    lim_max = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=ACCENT, lw=1.5, ls="--", label="Perfect fit")
    ax.set_xlabel("Actual MSRP (USD)")
    ax.set_ylabel("Predicted MSRP (USD)")
    ax.set_title("Actual vs Predicted")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.legend()
    ax.grid(True)
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.3, s=12, color=GOLD, edgecolors="none")
    ax2.axhline(0, color=ACCENT, lw=1.5, ls="--")
    ax2.set_xlabel("Predicted MSRP (USD)")
    ax2.set_ylabel("Residual (USD)")
    ax2.set_title("Residual Plot")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax2.grid(True)
    plt.tight_layout()
    _save(fig, f"3_actual_vs_predicted_{label.replace(' ', '_')}.png")


def plot_residual_histogram(y_true_log: np.ndarray, y_pred_log: np.ndarray, label: str = "OLS"):
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=60, color=ACCENT, edgecolor="#000", alpha=0.85)
    ax.axvline(0, color=GOLD, lw=2, ls="--", label="Zero residual")
    ax.set_title(f"{label} - Residual Histogram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Residual (USD)")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    _save(fig, f"4_residual_histogram_{label.replace(' ', '_')}.png")


def plot_model_comparison(metrics: dict):
    model_names = list(metrics.keys())
    mae_vals  = [metrics[m]["MAE"]  for m in model_names]
    rmse_vals = [metrics[m]["RMSE"] for m in model_names]
    r2_vals   = [metrics[m]["R2"]   for m in model_names]
    colors = [ACCENT, TEAL, GOLD]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Model Comparison", fontsize=15, fontweight="bold", color="#e94560")
    for ax, vals, title, fmt in zip(
        axes, [mae_vals, rmse_vals, r2_vals],
        ["MAE (USD)", "RMSE (USD)", "R2"],
        ["${:,.0f}", "${:,.0f}", "{:.4f}"]
    ):
        bars = ax.bar(model_names, vals, color=colors, edgecolor="#000", alpha=0.9)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(title)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    fmt.format(val), ha="center", va="bottom", fontsize=9, color="#e0e0e0")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y")
    plt.tight_layout()
    _save(fig, "5_model_comparison.png")


def plot_feature_importances(feature_names, coefficients, top_n=15):
    abs_coef = np.abs(coefficients)
    idx = np.argsort(abs_coef)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    coefs = coefficients[idx]
    colors = [ACCENT if c > 0 else TEAL for c in coefs]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(names)), coefs[::-1], color=colors[::-1], edgecolor="#000", alpha=0.9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.axvline(0, color="#e0e0e0", lw=1)
    ax.set_title(f"Top-{top_n} Feature Coefficients (Custom OLS)", fontsize=13, fontweight="bold", color="#e94560")
    ax.set_xlabel("Coefficient Value (log-price scale)")
    ax.grid(True, axis="x")
    plt.tight_layout()
    _save(fig, "6_feature_importances.png")


def plot_price_vs_features(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df = df[(df["MSRP"] > 500) & (df["MSRP"] < df["MSRP"].quantile(0.99))]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("MSRP vs Key Features (EDA)", fontsize=15, fontweight="bold", color="#e94560")
    ax = axes[0, 0]
    ax.scatter(df["Engine HP"], df["MSRP"], alpha=0.2, s=10, color=TEAL)
    ax.set_xlabel("Engine HP")
    ax.set_ylabel("MSRP (USD)")
    ax.set_title("Engine HP vs MSRP")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.grid(True)
    ax = axes[0, 1]
    yr_grp = df.groupby("Year")["MSRP"].median().reset_index()
    ax.plot(yr_grp["Year"], yr_grp["MSRP"], color=GOLD, lw=2, marker="o", markersize=4)
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Median MSRP (USD)")
    ax.set_title("Model Year vs Median MSRP")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.grid(True)
    ax = axes[1, 0]
    trans_grp = df.groupby("Transmission Type")["MSRP"].median().sort_values(ascending=False)
    ax.bar(trans_grp.index, trans_grp.values, color=ACCENT, edgecolor="#000", alpha=0.9)
    ax.set_xlabel("Transmission Type")
    ax.set_ylabel("Median MSRP (USD)")
    ax.set_title("Transmission Type vs Median MSRP")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y")
    ax = axes[1, 1]
    top_makes = df["Make"].value_counts().head(8).index
    df_top = df[df["Make"].isin(top_makes)]
    make_grp = df_top.groupby("Make")["MSRP"].median().sort_values(ascending=False)
    bar_colors = [TEAL, ACCENT, GOLD, TEAL, ACCENT, GOLD, TEAL, ACCENT]
    ax.bar(make_grp.index, make_grp.values, color=bar_colors[:len(make_grp)], edgecolor="#000", alpha=0.9)
    ax.set_xlabel("Make")
    ax.set_ylabel("Median MSRP (USD)")
    ax.set_title("Top Makes vs Median MSRP")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y")
    plt.tight_layout()
    _save(fig, "7_eda_price_vs_features.png")
