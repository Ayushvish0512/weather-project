import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta

router = APIRouter()

MODEL_PATH = "ml/model.pkl"


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


@router.get("/predict")
def predict(hours_ahead: int = 1):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")

    future_time = datetime.utcnow() + timedelta(hours=hours_ahead)
    features = np.array([[
        future_time.hour,
        60.0,   # placeholder humidity
        1013.0, # placeholder pressure
        5.0     # placeholder wind_speed
    ]])

    prediction = model.predict(features)[0]
    return {
        "predicted_temperature": round(float(prediction), 2),
        "hours_ahead": hours_ahead,
        "predicted_at": future_time.isoformat()
    }


@router.post("/train")
def trigger_training():
    import subprocess
    result = subprocess.run(["python", "ml/train.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return {"status": "Training complete", "output": result.stdout}
