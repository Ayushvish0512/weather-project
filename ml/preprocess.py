"""
ml/preprocess.py
Loads and prepares data from CSV files (not DB).
"""
import glob
import pandas as pd

FEATURE_COLS = ["hour", "day_of_week", "month", "humidity", "dew_point",
                "pressure", "cloudcover", "wind_speed", "wind_direction", "wind_gusts"]


def load_raw_data() -> pd.DataFrame:
    files = sorted(glob.glob("data/weather_*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files in data/. Run data/bootstrap.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").drop_duplicates("recorded_at").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]        = df["recorded_at"].dt.hour
    df["day_of_week"] = df["recorded_at"].dt.dayofweek
    df["month"]       = df["recorded_at"].dt.month
    return df


def get_features_and_target(df: pd.DataFrame):
    df = engineer_features(df)
    return df[FEATURE_COLS].values, df["temperature"].values
