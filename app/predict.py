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
    import httpx
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
    resp = httpx.get(url, timeout=15)
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


def weather_summary(temp: float, cloudcover: float, wind_speed: float) -> str:
    label = "Very Hot" if temp >= 35 else "Hot" if temp >= 28 else "Warm" if temp >= 20 else "Cool" if temp >= 10 else "Cold"
    label += ", Overcast" if cloudcover > 75 else ", Partly Cloudy" if cloudcover > 40 else ", Clear"
    if wind_speed > 40:   label += ", Strong Winds"
    elif wind_speed > 20: label += ", Breezy"
    return label


def _get_base_df() -> tuple[pd.DataFrame, dict]:
    """
    Load last 30 rows as a DataFrame (with engineered features) so we can
    roll the lag/rolling columns forward for multi-step forecasting.
    Returns (df_with_features, raw_dict).
    """
    files = sorted(glob.glob(DATA_GLOB))

    if files:
        df = pd.read_csv(files[-1]).tail(30).copy()
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        df = df.sort_values("recorded_at").reset_index(drop=True)
    else:
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
    return df, raw


def _roll_features(df: pd.DataFrame, predicted_temp: float, next_dt: datetime) -> pd.DataFrame:
    """
    Append a synthetic row using the predicted temperature so that the next
    step's lag and rolling features reflect the forecast, not stale history.
    """
    last = df.iloc[-1].copy()
    last["recorded_at"]  = pd.Timestamp(next_dt)
    last["temperature"]  = predicted_temp

    # Append and recompute only the lag/rolling columns that change
    df = pd.concat([df, last.to_frame().T], ignore_index=True)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])

    df["temp_lag_1h"]  = df["temperature"].shift(1)
    df["temp_lag_2h"]  = df["temperature"].shift(2)
    df["temp_lag_3h"]  = df["temperature"].shift(3)
    df["temp_lag_6h"]  = df["temperature"].shift(6)
    df["temp_lag_24h"] = df["temperature"].shift(24)
    # shift(1) before rolling so the current row's temperature is excluded from
    # its own window — matches the fix applied in ml/preprocess.py
    shifted = df["temperature"].shift(1)
    df["temp_rolling_mean_6h"] = shifted.rolling(6, min_periods=1).mean()
    df["temp_rolling_std_6h"]  = shifted.rolling(6, min_periods=1).std().fillna(0)

    # Update time features for the new row
    idx = df.index[-1]
    dt  = df.at[idx, "recorded_at"]
    df.at[idx, "hour"]       = dt.hour
    df.at[idx, "hour_sin"]   = np.sin(2 * np.pi * dt.hour / 24)
    df.at[idx, "hour_cos"]   = np.cos(2 * np.pi * dt.hour / 24)
    df.at[idx, "month"]      = dt.month
    df.at[idx, "month_sin"]  = np.sin(2 * np.pi * dt.month / 12)
    df.at[idx, "month_cos"]  = np.cos(2 * np.pi * dt.month / 12)
    df.at[idx, "season"]     = dt.month % 12 // 3
    df.at[idx, "is_daytime"] = int(6 <= dt.hour <= 18)

    return df


# ── Endpoints ──────────────────────────────────────────────

@router.get("/predict/next-hour")
def predict_next_hour():
    """Predict temperature for the next hour."""
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run ml/train.py first.")

    df, raw = _get_base_df()
    features = df[FEATURE_COLS].iloc[-1].values.astype("float32").reshape(1, -1)
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

    df, raw = _get_base_df()
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    forecast = []
    for h in range(1, hours + 1):
        dt       = now + timedelta(hours=h)
        features = df[FEATURE_COLS].iloc[-1].values.astype("float32").reshape(1, -1)
        temp     = round(float(model.predict(features)[0]), 2)
        insert_prediction(dt, temp, MODEL_VERSION)
        forecast.append({
            "hour":             h,
            "prediction_for":   dt.isoformat() + "Z",
            "predicted_temp_c": temp,
            "summary":          weather_summary(temp, raw["cloudcover"], raw["wind_speed"]),
        })
        # Roll the dataframe forward so the next step uses this prediction as lag input
        df = _roll_features(df, temp, dt)

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
