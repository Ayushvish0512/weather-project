"""
ml/train.py — quick single-model train (RandomForest).
For full multi-model training run ml/train_all.py
"""
import os
import gc
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from ml.preprocess import load_raw_data, get_features_and_target

ROOT          = Path(__file__).resolve().parent.parent
MODELS_DIR    = ROOT / "ml"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


def train():
    print("Loading data...")
    df = load_raw_data()
    X, y = get_features_and_target(df)
    del df; gc.collect()

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training RandomForest on {len(X_train)} rows ({X_train.shape[1]} features)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Test MAE: {mae:.4f}°C")

    joblib.dump(model, MODELS_DIR / f"model_{MODEL_VERSION}.pkl")
    joblib.dump(model, MODELS_DIR / "model.pkl")
    print(f"Saved: ml/model.pkl  ml/model_{MODEL_VERSION}.pkl")
    return model, MODEL_VERSION


if __name__ == "__main__":
    train()
