import pandas as pd
from db.postgres import get_connection


def load_raw_data() -> pd.DataFrame:
    """Load weather_raw table into a DataFrame."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM weather_raw ORDER BY recorded_at", conn)
    conn.close()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from recorded_at."""
    df = df.copy()
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df["hour"] = df["recorded_at"].dt.hour
    df["day_of_week"] = df["recorded_at"].dt.dayofweek
    df["month"] = df["recorded_at"].dt.month
    return df


def get_features_and_target(df: pd.DataFrame):
    """Return X (features) and y (target: temperature)."""
    feature_cols = ["hour", "humidity", "pressure", "wind_speed"]
    df = engineer_features(df)
    X = df[feature_cols].values
    y = df["temperature"].values
    return X, y
