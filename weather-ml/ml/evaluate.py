import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ml.preprocess import load_raw_data, get_features_and_target
from db.postgres import get_connection

MODEL_PATH = "ml/model.pkl"
MODEL_VERSION = "v1"


def evaluate():
    model = joblib.load(MODEL_PATH)
    df = load_raw_data()
    X, y = get_features_and_target(df)

    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # Store metrics in DB
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO model_metrics (evaluated_at, mae, rmse, model_version)
            VALUES (%s, %s, %s, %s)
        """, (datetime.utcnow(), mae, rmse, MODEL_VERSION))
    conn.commit()
    conn.close()

    return {"mae": mae, "rmse": rmse}


if __name__ == "__main__":
    evaluate()
