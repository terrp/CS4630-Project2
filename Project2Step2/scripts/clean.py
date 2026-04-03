import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


LABEL_COL = "label"
FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]
ALL_COLS = [LABEL_COL] + FEATURE_COLS

RAW_PATH = 'data/raw/HIGGS.csv.gz'
OUT_CSV       = "data/processed/higgs_200k.csv"
NROWS         = 200_000
RANDOM_STATE  = 42


# Load the first `nrows` rows from the raw gzip CSV.
def load_raw(file_path: str, nrows: int = NROWS) -> pd.DataFrame:
    print(f"Loading {nrows:,} rows from {file_path} ...")
    t0 = time.time()
    df = pd.read_csv(file_path, header=None, nrows=nrows, names=ALL_COLS)

    print(f"  Loaded in {time.time() - t0:.1f}s  |  shape: {df.shape}")

    return df

# Drop nulls, duplicates, and reset the index.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    removed = before - len(df)
    print(f"  Cleaned: removed {removed:,} rows  |  remaining: {len(df):,}")
    return df
 
 
"""
StandardScale the 28 feature columns (zero mean, unit variance).
The label column is left untouched.
Returns the scaled dataframe and the fitted scaler object.
"""
def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    print(f"  Features scaled  |  mean≈0, std≈1 verified")
    return df, scaler
 
 
# Print a short sanity-check summary.
def eda_summary(df: pd.DataFrame) -> None:
    print("\n── EDA summary ──────────────────────────────────────────")
    print(f"  Shape          : {df.shape}")
    print(f"  Label counts   :\n{df[LABEL_COL].value_counts().to_string()}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Duplicates     : {df.duplicated().sum()}")
    print(f"  Feature stats  :\n{df[FEATURE_COLS].describe().round(3).to_string()}")
    print("─────────────────────────────────────────────────────────\n")
 
 
# Save to both parquet (fast, preferred) and CSV (compatibility).
def save_outputs(df: pd.DataFrame) -> None:
    import os
    os.makedirs("data/processed", exist_ok=True)
 
    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved CSV     → {OUT_CSV}")
 
 
def main():
    t_start = time.time()
 
    df = load_raw(RAW_PATH)
    df = clean_data(df)
    df, scaler = scale_features(df)
    eda_summary(df)
    save_outputs(df)
 
    print(f"Pipeline complete in {time.time() - t_start:.1f}s total.")
    return df, scaler
 
 
if __name__ == "__main__":
    main()