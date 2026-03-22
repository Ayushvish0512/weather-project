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
from ml.preprocess import engineer_features, FEATURE_COLS

router = APIRouter(tags=["predictions"])

MODEL_PATH    = str(ROOT / "ml" / "model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DATA_GLOB     = str(ROOT / "data" / "weather_*.csv")


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


def _fetch_live_data() -> pd.DataFrame:
    """
    Fetch last 30 hours of weather data directly from Open-Meteo archive API.
    Used on Render where no CSV files are present.
    Returns a DataFrame with the same columns as the CSV files.
    """
    import requests
    end   = datetime.utcnow().date()
    start = end - timedelta(days=3)   # 3 days back — enough for lag_24h + rolling_6h

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=28.4595&longitude=77.0266"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,"
        "dew_point_2m,pressure_msl,cloudcover,visibility,windspeed_10m,"
        "winddirection_10m,windgusts_10m,precipitation,rain,weathercode"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly).rename(columns={
        "time":                  "recorded_at",
        "temperature_2m":        "temperature",
        "apparent_temperature":  "feels_like",
        "relative_humidity_2m":  "humidity",
        "dew_point_2m":          "dew_point",
        "pressure_msl":          "pressure",
        "windspeed_10m":         "wind_speed",
        "winddirection_10m":     "wind_direction",
        "windgusts_10m":         "wind_gusts",
        "weathercode":           "weather_main",
    })
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    return df.tail(30).reset_index(drop=True)


def get_latest_features() -> tuple[np.ndarray, dict]:
    """
    Load last 30 rows and engineer all 24 features.
    Source priority:
      1. Local CSV files (local dev / if CSVs committed)
      2. Live Open-Meteo API fetch (Render — no CSVs)
    """
    files = sorted(glob.glob(DATA_GLOB))

    if files:
        df = pd.read_csv(files[-1]).tail(30).copy()
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        df = df.sort_values("recorded_at").reset_index(drop=True)
    else:
        # No CSVs — fetch live from Open-Meteo (Render environment)
        try:
            df = _fetch_live_data()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"No CSV data and live fetch failed: {e}")

    df = engineer_features(df)
    df.dropna(subset=FEATURE_COLS, inplace=True)
    if df.empty:
        raise HTTPException(status_code=503, detail="Not enough rows to compute lag features.")

    last = df.iloc[-1]
    raw = {
        "cloudcover": float(last.get("cloudcover", 0)),
        "wind_speed":  float(last.get("wind_speed", 0)),
    }
    return last[FEATURE_COLS].values.astype("float32").reshape(1, -1), raw


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

    features, raw = get_latest_features()
    dt   = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    temp = round(float(model.predict(features)[0]), 2)
    insert_prediction(dt, temp, MODEL_VERSION)

    return {
        "prediction_for":   dt.isoformat() + "Z",
        "predicted_temp_c": temp,
        "summary":          weather_summary(temp, raw["cloudcover"], raw["wind_speed"]),
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

    features, raw = get_latest_features()
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    forecast = []
    for h in range(1, hours + 1):
        dt   = now + timedelta(hours=h)
        temp = round(float(model.predict(features)[0]), 2)
        insert_prediction(dt, temp, MODEL_VERSION)
        forecast.append({
            "hour":             h,
            "prediction_for":   dt.isoformat() + "Z",
            "predicted_temp_c": temp,
            "summary":          weather_summary(temp, raw["cloudcover"], raw["wind_speed"]),
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
