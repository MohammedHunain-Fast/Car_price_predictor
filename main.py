import numpy as np
import pandas as pd

from data_preprocessing import load_data, clean_data, engineer_features
from model import train_and_evaluate, predict_price
from visualizations import (
    plot_msrp_distribution,
    plot_correlation_heatmap,
    plot_predictions,
    plot_residual_histogram,
    plot_model_comparison,
    plot_feature_importances,
    plot_price_vs_features,
)

DATASET_PATH = "data.csv"
MODEL_PATH   = "model_weights.npz"


def main():
    print("+--------------------------------------------------------------+")
    print("|           CAR PRICE PREDICTOR - FAST NUCES Karachi           |")
    print("|      Programming For AI  |  Section BAI-3B                   |")
    print("|  Hunain 25k-0068 | Hassan 25k-0061 | Vinod 25k-0114          |")
    print("+--------------------------------------------------------------+")

    print("\n" + "=" * 60)
    print("  STEP 1 - DATA LOADING")
    print("=" * 60)
    df_raw   = load_data(DATASET_PATH)
    df_clean = clean_data(df_raw.copy())

    print("\n" + "=" * 60)
    print("  STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print("[eda]   Generating EDA plots ...")
    plot_msrp_distribution(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_price_vs_features(df_clean)

    print("\n" + "=" * 60)
    print("  STEP 3 - FEATURE ENGINEERING")
    print("=" * 60)
    X, y, feature_names = engineer_features(df_clean)

    print("\n" + "=" * 60)
    print("  STEP 4 - MODEL TRAINING & EVALUATION")
    print("=" * 60)
    results = train_and_evaluate(X, y, feature_names, save_path=MODEL_PATH)

    print("\n" + "=" * 60)
    print("  STEP 5 - PERFORMANCE VISUALIZATIONS")
    print("=" * 60)
    print("[plot]  Generating model performance plots ...")
    plot_predictions(results["y_test"], results["y_pred_ols"], label="OLS")
    plot_residual_histogram(results["y_test"], results["y_pred_ols"], label="OLS")
    plot_model_comparison({
        "OLS":   results["ols_metrics"],
        "Ridge": results["ridge_metrics"],
        "Lasso": results["lasso_metrics"],
    })
    plot_feature_importances(results["feature_names"], results["ols"].coefficients_, top_n=15)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY REPORT")
    print("=" * 60)
    cv  = results["cv_results"]
    ols_m = results["ols_metrics"]
    print(f"""
  Dataset       : {DATASET_PATH}
  Samples used  : {X.shape[0]:,}
  Features      : {X.shape[1]}

  +--------------------------------------------------+
  |   Custom OLS Linear Regression (Test Set)        |
  +--------------------------------------------------+
  |  MAE        : ${ols_m['MAE']:>12,.2f}                 |
  |  RMSE       : ${ols_m['RMSE']:>12,.2f}                 |
  |  R2         :   {ols_m['R2']:>10.4f}                 |
  |  Adj-R2     :   {ols_m['Adj_R2']:>10.4f}                 |
  |  MAPE       :   {ols_m['MAPE (%)']:>10.2f} %              |
  +--------------------------------------------------+
  |   5-Fold Cross-Validation                        |
  +--------------------------------------------------+
  |  R2   mean  :   {cv['R2_mean']:>10.4f} +/- {cv['R2_std']:.4f}      |
  |  RMSE mean  : ${cv['RMSE_mean']:>12,.0f} +/- ${cv['RMSE_std']:,.0f}    |
  +--------------------------------------------------+

  Plots saved to: ./plots/
  Model saved to: {MODEL_PATH}
""")

    print("=" * 60)
    print("  DEMO PREDICTION")
    print("=" * 60)
    sample = {
        "Engine HP": 300.0, "Engine Cylinders": 6.0, "Number of Doors": 2.0,
        "highway MPG": 26, "city mpg": 19, "Popularity": 3916, "Car Age": 2,
        "Is Luxury": 1, "Is Performance": 1, "Is Exotic": 0, "Is Diesel": 0, "Is FlexFuel": 0,
        "Engine Fuel Type_premium unleaded (required)": 1,
        "Transmission Type_MANUAL": 1,
        "Driven_Wheels_rear wheel drive": 1,
        "Vehicle Size_Compact": 1,
        "Vehicle Style_Coupe": 1,
    }
    pred = predict_price(sample, MODEL_PATH)
    print(f"\n  Sample input: 2015 BMW Coupe, 300HP, 6-cyl, Manual, Rear-Wheel, Luxury")
    print(f"  Predicted MSRP : ${pred:,.2f}")
    print("\n  Pipeline complete. All plots are in the './plots/' folder.")


if __name__ == "__main__":
    main()
