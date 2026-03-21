"""
ml/predict.py
Loads model.pkl, predicts next-hour temperature, stores result in DB.
Features come from the latest row in the merged CSV data.
"""
import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Ensure project root is on sys.path so `db` resolves anywhere
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.postgres import insert_prediction

MODEL_PATH    = str(ROOT / "ml" / "model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DATA_GLOB     = str(ROOT / "data" / "weather_*.csv")
FEATURE_COLS  = ["hour", "day_of_week", "month", "humidity", "dew_point",
                 "pressure", "cloudcover", "wind_speed", "wind_direction", "wind_gusts"]


def get_latest_features() -> dict:
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No CSV files found at {DATA_GLOB}. Run data/bootstrap.py first.")
    # Read only last file for speed, take last row
    df = pd.read_csv(files[-1])
    row = df.iloc[-1]
    return {"humidity": row["humidity"], "dew_point": row["dew_point"],
            "pressure": row["pressure"], "cloudcover": row["cloudcover"],
            "wind_speed": row["wind_speed"], "wind_direction": row["wind_direction"],
            "wind_gusts": row["wind_gusts"]}


def predict_next_hour() -> dict:
    model = joblib.load(MODEL_PATH)
    latest = get_latest_features()

    prediction_for = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    features = np.array([[
        prediction_for.hour,
        prediction_for.weekday(),
        prediction_for.month,
        latest["humidity"],
        latest["dew_point"],
        latest["pressure"],
        latest["cloudcover"],
        latest["wind_speed"],
        latest["wind_direction"],
        latest["wind_gusts"]
    ]])

    predicted_temp = round(float(model.predict(features)[0]), 2)
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
