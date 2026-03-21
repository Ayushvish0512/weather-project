"""
ml/train_all.py
Trains multiple models, compares MAE, saves each as model_<name>.pkl
Best model is also saved as model.pkl
"""
import os
import glob
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

MODELS_DIR   = "ml"
FEATURE_COLS = ["hour", "day_of_week", "month", "humidity", "dew_point",
                "pressure", "cloudcover", "wind_speed", "wind_direction", "wind_gusts"]

MODELS = {
    "linear_regression":   LinearRegression(),
    "ridge":               Ridge(alpha=1.0),
    "decision_tree":       DecisionTreeRegressor(max_depth=10, random_state=42),
    "random_forest":       RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "gradient_boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42),
    "xgboost":             XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "knn":                 KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
}


def load_data():
    files = sorted(glob.glob("data/weather_*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files in data/. Run data/bootstrap.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").drop_duplicates("recorded_at").reset_index(drop=True)
    df["hour"]        = df["recorded_at"].dt.hour
    df["day_of_week"] = df["recorded_at"].dt.dayofweek
    df["month"]       = df["recorded_at"].dt.month
    df.dropna(subset=FEATURE_COLS + ["temperature"], inplace=True)
    return df


def train_all():
    df    = load_data()
    X     = df[FEATURE_COLS].values
    y     = df["temperature"].values
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows\n")
    print(f"{'Model':<25} {'MAE':>8}")
    print("-" * 35)

    results = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        results[name] = (model, mae)
        path = os.path.join(MODELS_DIR, f"model_{name}.pkl")
        joblib.dump(model, path)
        print(f"{name:<25} {mae:>8.4f}°C")

    # Save best as model.pkl
    best_name, (best_model, best_mae) = min(results.items(), key=lambda x: x[1][1])
    joblib.dump(best_model, os.path.join(MODELS_DIR, "model.pkl"))
    print(f"\nBest: {best_name} (MAE {best_mae:.4f}°C) → saved as model.pkl")


if __name__ == "__main__":
    train_all()
