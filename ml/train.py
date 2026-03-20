"""
ml/train.py
Merges all CSVs from data/ folder, trains RandomForest, saves model.pkl
"""
import os
import glob
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

MODELS_DIR = "ml"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
FEATURE_COLS = ["hour", "day_of_week", "month", "humidity", "dew_point",
                "pressure", "cloudcover", "wind_speed", "wind_direction", "wind_gusts"]


def load_all_csvs() -> pd.DataFrame:
    files = sorted(glob.glob("data/weather_*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in data/. Run data/bootstrap.py first.")
    print(f"Found {len(files)} CSV file(s): {[os.path.basename(f) for f in files]}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").drop_duplicates("recorded_at").reset_index(drop=True)
    print(f"Total rows after merge: {len(df)}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]        = df["recorded_at"].dt.hour
    df["day_of_week"] = df["recorded_at"].dt.dayofweek
    df["month"]       = df["recorded_at"].dt.month
    return df


def train():
    df = load_all_csvs()
    df = engineer_features(df)
    df.dropna(subset=FEATURE_COLS + ["temperature"], inplace=True)

    X = df[FEATURE_COLS].values
    y = df["temperature"].values

    # Chronological split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Training RandomForest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Test MAE: {mae:.4f}°C")

    joblib.dump(model, os.path.join(MODELS_DIR, f"model_{MODEL_VERSION}.pkl"))
    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
    print(f"Saved: ml/model.pkl  ml/model_{MODEL_VERSION}.pkl")
    return model, MODEL_VERSION


if __name__ == "__main__":
    train()
