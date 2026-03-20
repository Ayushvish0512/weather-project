"""
ml/evaluate.py
Joins weather_predictions with weather_raw on prediction_for = recorded_at,
computes MAE & RMSE, and stores results in model_metrics.
"""
import os
import numpy as np
from db.postgres import fetch_prediction_vs_actual, insert_metrics

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


def evaluate():
    rows = fetch_prediction_vs_actual()

    if not rows:
        print("No matched predictions vs actuals yet. Collect more data.")
        return None

    predicted = np.array([r["predicted_temp"] for r in rows])
    actual = np.array([r["actual_temp"] for r in rows])

    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    print(f"Evaluated {len(rows)} predictions | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    insert_metrics(mae, rmse, MODEL_VERSION)

    return {"mae": mae, "rmse": rmse, "samples": len(rows)}


if __name__ == "__main__":
    evaluate()
