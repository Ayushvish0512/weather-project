"""
ml/predict.py
Loads the trained model, predicts next-hour temperature,
and stores the result in weather_predictions.
"""
import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from db.postgres import insert_prediction

MODEL_PATH = "ml/model.pkl"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


def predict_next_hour() -> dict:
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise RuntimeError("model.pkl not found. Run ml/train.py first.")

    prediction_for = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # Use placeholder features — replace with live data from weather_raw if available
    features = np.array([[
        prediction_for.hour,
        60.0,    # humidity placeholder
        1013.0,  # pressure placeholder
        5.0      # wind_speed placeholder
    ]])

    predicted_temp = round(float(model.predict(features)[0]), 2)

    insert_prediction(prediction_for, predicted_temp, MODEL_VERSION)

    result = {
        "prediction_for": prediction_for.isoformat(),
        "predicted_temp": predicted_temp,
        "model_version": MODEL_VERSION
    }
    print(f"Prediction stored: {result}")
    return result


if __name__ == "__main__":
    predict_next_hour()
