"""
ml/train_all.py
Trains multiple models with epoch-style progress logging.
- RandomForest / GradientBoosting / DecisionTree: simulated epochs (incremental n_estimators)
- XGBoost: real per-round eval logging
- Linear / Ridge / KNN: single pass (no epochs concept)
Best model saved as model.pkl
"""
import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, callback

ROOT         = Path(__file__).resolve().parent.parent
MODELS_DIR   = ROOT / "ml"
DATA_GLOB    = str(ROOT / "data" / "weather_*.csv")
FEATURE_COLS = ["hour", "day_of_week", "month", "humidity", "dew_point",
                "pressure", "cloudcover", "wind_speed", "wind_direction", "wind_gusts"]

EPOCHS       = 100  # simulated epoch rounds for tree models
TREES_TOTAL  = 500  # total n_estimators split across epochs


def load_data():
    files = sorted(glob.glob(DATA_GLOB))
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


def train_with_epochs_rf(X_train, y_train, X_test, y_test) -> tuple:
    """RandomForest: warm_start lets us add trees per epoch."""
    trees_per_epoch = TREES_TOTAL // EPOCHS
    model = RandomForestRegressor(n_estimators=0, warm_start=True, random_state=42, n_jobs=-1)
    print(f"\n  [RandomForest] {EPOCHS} epochs × {trees_per_epoch} trees = {EPOCHS*trees_per_epoch} total")
    for epoch in range(1, EPOCHS + 1):
        model.n_estimators = epoch * trees_per_epoch
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"    Epoch {epoch}/{EPOCHS} — trees: {model.n_estimators:>4}  MAE: {mae:.4f}°C")
    return model, mae


def train_with_epochs_gb(X_train, y_train, X_test, y_test) -> tuple:
    """GradientBoosting: warm_start per epoch."""
    trees_per_epoch = TREES_TOTAL // EPOCHS
    model = GradientBoostingRegressor(n_estimators=0, warm_start=True, random_state=42)
    print(f"\n  [GradientBoosting] {EPOCHS} epochs × {trees_per_epoch} trees = {EPOCHS*trees_per_epoch} total")
    for epoch in range(1, EPOCHS + 1):
        model.n_estimators = epoch * trees_per_epoch
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"    Epoch {epoch}/{EPOCHS} — trees: {model.n_estimators:>4}  MAE: {mae:.4f}°C")
    return model, mae


def train_xgboost(X_train, y_train, X_test, y_test) -> tuple:
    """XGBoost: real per-round logging via verbose_eval."""
    print(f"\n  [XGBoost] {TREES_TOTAL} rounds (real boosting iterations)")
    class EpochPrinter(callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if (epoch + 1) % (TREES_TOTAL // EPOCHS) == 0 or epoch == 0:
                key = list(evals_log.keys())[0]
                mae = list(evals_log[key].values())[0][-1]
                print(f"    Round {epoch+1:>4}/{TREES_TOTAL}  MAE: {mae:.4f}°C")
            return False

    model = XGBRegressor(
        n_estimators=TREES_TOTAL,
        random_state=42,
        verbosity=0,
        eval_metric="mae",
        callbacks=[EpochPrinter()]
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    model.set_params(callbacks=None)  # remove callback before pickling
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mae


def train_simple(name, model, X_train, y_train, X_test, y_test) -> tuple:
    """Single-pass models — no epoch concept."""
    print(f"\n  [{name}] single pass (no epochs)")
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"    MAE: {mae:.4f}°C")
    return model, mae


def train_all():
    df    = load_data()
    X     = df[FEATURE_COLS].values
    y     = df["temperature"].values
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Dataset: {len(X_train)} train / {len(X_test)} test rows")
    print("=" * 50)

    results = {}

    m, mae = train_with_epochs_rf(X_train, y_train, X_test, y_test)
    results["random_forest"] = (m, mae)

    m, mae = train_with_epochs_gb(X_train, y_train, X_test, y_test)
    results["gradient_boosting"] = (m, mae)

    m, mae = train_xgboost(X_train, y_train, X_test, y_test)
    results["xgboost"] = (m, mae)

    for name, model in [
        ("decision_tree",     DecisionTreeRegressor(max_depth=10, random_state=42)),
        ("linear_regression", LinearRegression()),
        ("ridge",             Ridge(alpha=1.0)),
        ("knn",               KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
    ]:
        m, mae = train_simple(name, model, X_train, y_train, X_test, y_test)
        results[name] = (m, mae)

    # Save all models
    print("\n" + "=" * 50)
    print(f"{'Model':<25} {'MAE':>8}")
    print("-" * 35)
    for name, (model, mae) in sorted(results.items(), key=lambda x: x[1][1]):
        path = MODELS_DIR / f"model_{name}.pkl"
        joblib.dump(model, path)
        print(f"{name:<25} {mae:>8.4f}°C")

    best_name, (best_model, best_mae) = min(results.items(), key=lambda x: x[1][1])
    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    print(f"\nBest: {best_name} (MAE {best_mae:.4f}°C) → saved as model.pkl")


if __name__ == "__main__":
    train_all()
