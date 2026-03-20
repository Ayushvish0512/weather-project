import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ml.preprocess import load_raw_data, get_features_and_target

MODELS_DIR = "ml"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


def get_model_path(version: str) -> str:
    return os.path.join(MODELS_DIR, f"model_{version}.pkl")


def train():
    df = load_raw_data()
    if len(df) < 10:
        print("Not enough data to train. Need at least 10 records.")
        return None, None

    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    path = get_model_path(MODEL_VERSION)
    joblib.dump(model, path)

    # Also save as latest for easy loading
    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))

    print(f"Model trained and saved: {path} (version={MODEL_VERSION})")
    return model, MODEL_VERSION


if __name__ == "__main__":
    train()
