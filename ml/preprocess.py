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
    # Time — fully cyclical so model understands 23→0 and Dec→Jan wrap smoothly
    # day_of_week removed: correlation=0.003, zero feature importance — pure noise
    "hour", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",   # month now cyclical (was raw int)
    "season", "is_daytime",
    # Raw weather
    # feels_like removed — derived from temperature, causes target leakage
    # daily_temp_max / daily_temp_min removed — cummax/cummin within the day
    #   includes the current hour's temperature (the target), causing leakage
    "humidity", "dew_point", "pressure", "cloudcover",
    "wind_speed", "wind_direction", "wind_gusts",
    "precipitation", "rain", "weather_main",
    # Derived
    "humidity_pressure_ratio",
    # Lag features — all shift by ≥1 so no current-hour temperature leaks in
    "temp_lag_1h", "temp_lag_2h", "temp_lag_3h", "temp_lag_6h", "temp_lag_24h",
    # Rolling stats — computed on shift(1) series so current hour is excluded
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

    # Time features — hour and month both cyclically encoded so the model
    # understands that 23:00→00:00 and December→January are continuous, not jumps
    df["hour"]       = dt.dt.hour
    df["hour_sin"]   = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["month"]      = dt.dt.month
    df["month_sin"]  = np.sin(2 * np.pi * dt.dt.month / 12)  # Dec→Jan wraps smoothly
    df["month_cos"]  = np.cos(2 * np.pi * dt.dt.month / 12)
    df["season"]     = (dt.dt.month % 12 // 3)
    df["is_daytime"] = dt.dt.hour.between(6, 18).astype(int)
    # day_of_week intentionally removed — correlation=0.003, zero feature importance

    # Derived
    df["humidity_pressure_ratio"] = df["humidity"] / df["pressure"].replace(0, np.nan)

    # daily_temp_max / daily_temp_min intentionally removed:
    # cummax/cummin within the day includes the current hour's temperature,
    # which is the prediction target — this is direct target leakage.

    # Lag features — shift(≥1) ensures no current-hour temperature leaks in
    df["temp_lag_1h"]  = df["temperature"].shift(1)
    df["temp_lag_2h"]  = df["temperature"].shift(2)
    df["temp_lag_3h"]  = df["temperature"].shift(3)
    df["temp_lag_6h"]  = df["temperature"].shift(6)
    df["temp_lag_24h"] = df["temperature"].shift(24)

    # Rolling stats — computed on shift(1) series so the current hour's temperature
    # is excluded from the window. Without shift(1), rolling(6) includes the target
    # row itself, which is leakage.
    shifted = df["temperature"].shift(1)
    df["temp_rolling_mean_6h"] = shifted.rolling(6, min_periods=1).mean()
    df["temp_rolling_std_6h"]  = shifted.rolling(6, min_periods=1).std().fillna(0)

    return df


def get_features_and_target(df: pd.DataFrame):
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS].values.astype("float32")
    y = df["temperature"].values.astype("float32")
    return X, y
