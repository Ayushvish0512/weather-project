"""
ml/predict.py
Loads model.pkl, predicts next-hour temperature, stores result in DB.
Uses full feature set including lag + rolling features from latest CSV rows.
"""
import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.postgres import insert_prediction
from ml.preprocess import engineer_features, FEATURE_COLS

MODEL_PATH    = str(ROOT / "ml" / "model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DATA_GLOB     = str(ROOT / "data" / "weather_*.csv")


def get_latest_features() -> np.ndarray:
    """Load last 30 rows from latest CSV, engineer all features, return last row as array."""
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError("No CSV files found. Run data/bootstrap.py first.")

    # Need 30 rows to compute lag_24h + rolling_6h correctly
    df = pd.read_csv(files[-1]).tail(30).copy()
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").reset_index(drop=True)
    df = engineer_features(df)
    df.dropna(subset=FEATURE_COLS, inplace=True)

    if df.empty:
        raise RuntimeError("Not enough rows to compute lag features.")

    return df[FEATURE_COLS].iloc[-1].values.astype("float32")


def predict_next_hour() -> dict:
    model    = joblib.load(MODEL_PATH)
    features = get_latest_features().reshape(1, -1)

    predicted_temp = round(float(model.predict(features)[0]), 2)
    prediction_for = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    insert_prediction(prediction_for, predicted_temp, MODEL_VERSION)

    result = {
        "prediction_for": prediction_for.isoformat(),
        "predicted_temp": predicted_temp,
        "model_version":  MODEL_VERSION
    }
    print(f"Prediction stored: {result}")
    return result


if __name__ == "__main__":
    predict_next_hour()
