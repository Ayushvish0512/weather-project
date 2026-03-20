import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from db.postgres import insert_prediction
import os

router = APIRouter()

MODEL_PATH = "ml/model.pkl"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


@router.get("/predict")
def predict(hours_ahead: int = 1):
    """Predict temperature N hours ahead and persist to weather_predictions."""
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run ml/train.py first.")

    prediction_for = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_ahead)
    features = np.array([[
        prediction_for.hour,
        60.0,    # placeholder humidity
        1013.0,  # placeholder pressure
        5.0      # placeholder wind_speed
    ]])

    predicted_temp = round(float(model.predict(features)[0]), 2)
    insert_prediction(prediction_for, predicted_temp, MODEL_VERSION)

    return {
        "prediction_for": prediction_for.isoformat(),
        "predicted_temperature": predicted_temp,
        "model_version": MODEL_VERSION,
        "hours_ahead": hours_ahead
    }


@router.post("/train")
def trigger_training():
    """Trigger model retraining via ml/train.py."""
    import subprocess
    result = subprocess.run(["python", "ml/train.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return {"status": "Training complete", "output": result.stdout}


@router.post("/evaluate")
def trigger_evaluate():
    """Trigger evaluation of predictions vs actuals."""
    import subprocess
    result = subprocess.run(["python", "ml/evaluate.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return {"status": "Evaluation complete", "output": result.stdout}
