import pandas as pd
import numpy as np


def load_data(filepath: str = "data.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    print(f"[load]  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[clean] Dropped {before - len(df):,} duplicate rows.")

    df.dropna(subset=["MSRP"], inplace=True)
    df = df[df["MSRP"] > 0]
    df.dropna(subset=["Engine HP", "Engine Cylinders"], inplace=True)

    cap = df["MSRP"].quantile(0.99)
    n_capped = (df["MSRP"] > cap).sum()
    df = df[df["MSRP"] <= cap]
    print(f"[clean] Removed {n_capped:,} MSRP outlier rows (above 99th pct = ${cap:,.0f}).")

    door_mode = df["Number of Doors"].mode()[0]
    df["Number of Doors"] = df["Number of Doors"].fillna(door_mode)
    df["Market Category"] = df["Market Category"].fillna("Unknown")

    print(f"[clean] Clean dataset: {df.shape[0]:,} rows remaining.")
    return df.reset_index(drop=True)


REFERENCE_YEAR = 2017


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Car Age"] = REFERENCE_YEAR - df["Year"]
    df["Is Luxury"]      = df["Market Category"].str.contains("Luxury",      na=False).astype(int)
    df["Is Performance"] = df["Market Category"].str.contains("Performance",  na=False).astype(int)
    df["Is Exotic"]      = df["Market Category"].str.contains("Exotic",       na=False).astype(int)
    df["Is Diesel"]      = df["Market Category"].str.contains("Diesel",       na=False).astype(int)
    df["Is FlexFuel"]    = df["Market Category"].str.contains("Flex Fuel",    na=False).astype(int)

    df["log_MSRP"] = np.log1p(df["MSRP"])

    drop_cols = ["Make", "Model", "Year", "Market Category", "MSRP"]
    df.drop(columns=drop_cols, inplace=True)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    y = df["log_MSRP"].values.astype(float)
    X = df.drop(columns=["log_MSRP"]).values.astype(float)
    feature_names = df.drop(columns=["log_MSRP"]).columns.tolist()

    print(f"[feat]  Feature matrix: {X.shape}  |  Target: {y.shape}")
    return X, y, feature_names


def preprocess(filepath: str = "data.csv"):
    df = load_data(filepath)
    df = clean_data(df)
    return engineer_features(df)


if __name__ == "__main__":
    X, y, names = preprocess()
    print(f"\nFeature names ({len(names)}):")
    for n in names:
        print(f"  {n}")
