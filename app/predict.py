import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.postgres import insert_prediction

router = APIRouter(tags=["predictions"])

MODEL_PATH    = str(ROOT / "ml" / "model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DATA_GLOB     = str(ROOT / "data" / "weather_*.csv")
FEATURE_COLS  = ["hour", "day_of_week", "month", "humidity", "dew_point",
                 "pressure", "cloudcover", "wind_speed", "wind_direction", "wind_gusts"]


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


def get_latest_features() -> dict:
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise HTTPException(status_code=503, detail="No CSV data. Run data/bootstrap.py first.")
    row = pd.read_csv(files[-1]).iloc[-1]
    return {
        "humidity":       float(row["humidity"]),
        "dew_point":      float(row["dew_point"]),
        "pressure":       float(row["pressure"]),
        "cloudcover":     float(row["cloudcover"]),
        "wind_speed":     float(row["wind_speed"]),
        "wind_direction": float(row["wind_direction"]),
        "wind_gusts":     float(row["wind_gusts"]),
    }


def build_features(dt: datetime, c: dict) -> np.ndarray:
    return np.array([[dt.hour, dt.weekday(), dt.month,
                      c["humidity"], c["dew_point"], c["pressure"],
                      c["cloudcover"], c["wind_speed"], c["wind_direction"], c["wind_gusts"]]])


def weather_summary(temp: float, cloudcover: float, wind_speed: float) -> str:
    label = "Very Hot" if temp >= 35 else "Hot" if temp >= 28 else "Warm" if temp >= 20 else "Cool" if temp >= 10 else "Cold"
    label += ", Overcast" if cloudcover > 75 else ", Partly Cloudy" if cloudcover > 40 else ", Clear"
    if wind_speed > 40:   label += ", Strong Winds"
    elif wind_speed > 20: label += ", Breezy"
    return label


# ── Endpoints ──────────────────────────────────────────────

@router.get("/predict/next-hour")
def predict_next_hour():
    """Predict temperature for the next hour."""
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run ml/train.py first.")

    c  = get_latest_features()
    dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    temp = round(float(model.predict(build_features(dt, c))[0]), 2)
    insert_prediction(dt, temp, MODEL_VERSION)

    return {
        "prediction_for":   dt.isoformat() + "Z",
        "predicted_temp_c": temp,
        "summary":          weather_summary(temp, c["cloudcover"], c["wind_speed"]),
        "conditions_used":  c,
        "model_version":    MODEL_VERSION,
    }


@router.get("/predict/hours")
def predict_multiple_hours(hours: int = 6):
    """Predict temperature for the next N hours (max 24)."""
    if not 1 <= hours <= 24:
        raise HTTPException(status_code=400, detail="hours must be 1–24")

    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run ml/train.py first.")

    c   = get_latest_features()
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    forecast = []
    for h in range(1, hours + 1):
        dt   = now + timedelta(hours=h)
        temp = round(float(model.predict(build_features(dt, c))[0]), 2)
        insert_prediction(dt, temp, MODEL_VERSION)
        forecast.append({
            "hour":             h,
            "prediction_for":   dt.isoformat() + "Z",
            "predicted_temp_c": temp,
            "summary":          weather_summary(temp, c["cloudcover"], c["wind_speed"]),
        })

    return {
        "location":      "Gurgaon, IN",
        "generated_at":  now.isoformat() + "Z",
        "model_version": MODEL_VERSION,
        "forecast":      forecast,
    }


@router.get("/predict/today")
def predict_today():
    """Predict all remaining hours of today (UTC)."""
    hours = 23 - datetime.utcnow().hour
    if hours < 1:
        return {"message": "No remaining hours today."}
    return predict_multiple_hours(hours=hours)


@router.post("/train")
def trigger_training():
    """Retrain model from all CSVs in data/."""
    import subprocess, sys
    r = subprocess.run([sys.executable, "ml/train.py"], capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(status_code=500, detail=r.stderr)
    return {"status": "Training complete", "output": r.stdout.strip()}


@router.post("/evaluate")
def trigger_evaluate():
    """Evaluate stored predictions against actuals."""
    import subprocess, sys
    r = subprocess.run([sys.executable, "ml/evaluate.py"], capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(status_code=500, detail=r.stderr)
    return {"status": "Evaluation complete", "output": r.stdout.strip()}
