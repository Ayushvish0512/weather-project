"""
ml/preprocess.py
Loads CSVs and engineers all features including lag, rolling, and derived columns.
"""
import glob
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    # Time — cyclical encoding so model understands 23→0 wraps
    "hour", "hour_sin", "hour_cos", "day_of_week", "month", "season", "is_daytime",
    # Raw weather (feels_like removed — it's derived from temperature, causes leakage)
    "humidity", "dew_point", "pressure", "cloudcover",
    "wind_speed", "wind_direction", "wind_gusts",
    "precipitation", "rain", "weather_main",
    # Derived
    "humidity_pressure_ratio",
    "daily_temp_max", "daily_temp_min",
    # Lag features — extended for better trend signal
    "temp_lag_1h", "temp_lag_2h", "temp_lag_3h", "temp_lag_6h", "temp_lag_24h",
    # Rolling stats
    "temp_rolling_mean_6h", "temp_rolling_std_6h",
]


def load_raw_data() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "weather_*.csv")))
    if not files:
        raise FileNotFoundError("No CSV files in data/. Run data/bootstrap.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").drop_duplicates("recorded_at").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["recorded_at"]

    # Time features
    df["hour"]       = dt.dt.hour
    df["hour_sin"]   = np.sin(2 * np.pi * dt.dt.hour / 24)   # cyclical — 23→0 wraps smoothly
    df["hour_cos"]   = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["day_of_week"]= dt.dt.dayofweek
    df["month"]      = dt.dt.month
    df["season"]     = (dt.dt.month % 12 // 3)
    df["is_daytime"] = dt.dt.hour.between(6, 18).astype(int)

    # Derived
    df["humidity_pressure_ratio"] = df["humidity"] / df["pressure"].replace(0, np.nan)

    # Daily min/max up to current hour (expanding within each day)
    date_col = dt.dt.date
    df["daily_temp_max"] = df.groupby(date_col)["temperature"].cummax()
    df["daily_temp_min"] = df.groupby(date_col)["temperature"].cummin()

    # Lag features — extended set for better trend signal
    df["temp_lag_1h"]  = df["temperature"].shift(1)
    df["temp_lag_2h"]  = df["temperature"].shift(2)
    df["temp_lag_3h"]  = df["temperature"].shift(3)
    df["temp_lag_6h"]  = df["temperature"].shift(6)
    df["temp_lag_24h"] = df["temperature"].shift(24)

    # Rolling stats (min_periods so early rows aren't all NaN)
    df["temp_rolling_mean_6h"] = df["temperature"].rolling(6, min_periods=1).mean()
    df["temp_rolling_std_6h"]  = df["temperature"].rolling(6, min_periods=1).std().fillna(0)

    return df


def get_features_and_target(df: pd.DataFrame):
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS].values.astype("float32")
    y = df["temperature"].values.astype("float32")
    return X, y
