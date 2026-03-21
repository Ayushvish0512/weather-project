"""
ml/train_all.py
100 epochs, RAM-safe:
- Models trained and saved one at a time (never held in memory together)
- gc.collect() + del between each model
- numpy arrays kept as float32 (half the RAM of float64)
- XGBoost uses hist tree method (low RAM)
- RF/GB use max_samples + max_features to limit per-tree memory
"""
import os
import gc
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

EPOCHS       = 300  # simulated epoch rounds for tree models
TREES_TOTAL  = 1500  # total n_estimators split across epochs


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


def free(model):
    """Delete model and force garbage collection to free RAM."""
    del model
    gc.collect()


def train_rf(X_train, y_train, X_test, y_test, name="random_forest"):
    trees_per_epoch = TREES_TOTAL // EPOCHS
    model = RandomForestRegressor(
        n_estimators=0, warm_start=True, random_state=42,
        n_jobs=1,           # 1 core = no parallel RAM spikes
        max_depth=12,
        max_samples=0.6,    # each tree sees 60% of rows
        max_features=0.7,
    )
    print(f"\n  [{name}] {EPOCHS} epochs × {trees_per_epoch} trees = {TREES_TOTAL} total")
    mae = None
    for epoch in range(1, EPOCHS + 1):
        model.n_estimators = epoch * trees_per_epoch
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"    Epoch {epoch:>3}/{EPOCHS} — trees: {model.n_estimators:>4}  MAE: {mae:.4f}°C")
        gc.collect()   # free intermediate memory each epoch
    return model, mae


def train_gb(X_train, y_train, X_test, y_test, name="gradient_boosting"):
    trees_per_epoch = TREES_TOTAL // EPOCHS
    model = GradientBoostingRegressor(
        n_estimators=0, warm_start=True, random_state=42,
        max_depth=4,
        subsample=0.6,
        max_features=0.7,
    )
    print(f"\n  [{name}] {EPOCHS} epochs × {trees_per_epoch} trees = {TREES_TOTAL} total")
    mae = None
    for epoch in range(1, EPOCHS + 1):
        model.n_estimators = epoch * trees_per_epoch
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"    Epoch {epoch:>3}/{EPOCHS} — trees: {model.n_estimators:>4}  MAE: {mae:.4f}°C")
        gc.collect()
    return model, mae


def train_xgb(X_train, y_train, X_test, y_test, name="xgboost"):
    print(f"\n  [{name}] {TREES_TOTAL} rounds — printing every {TREES_TOTAL // EPOCHS}")

    class EpochPrinter(callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            step = TREES_TOTAL // EPOCHS
            if (epoch + 1) % step == 0 or epoch == 0:
                key = list(evals_log.keys())[0]
                mae = list(evals_log[key].values())[0][-1]
                print(f"    Round {epoch+1:>4}/{TREES_TOTAL}  MAE: {mae:.4f}°C")
            return False

    model = XGBRegressor(
        n_estimators=TREES_TOTAL,
        random_state=42,
        verbosity=0,
        eval_metric="mae",
        tree_method="hist",    # low RAM histogram method
        max_depth=6,
        subsample=0.6,
        colsample_bytree=0.7,
        nthread=1,             # single thread = no RAM spikes
        callbacks=[EpochPrinter()]
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    model.set_params(callbacks=None)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    gc.collect()
    return model, mae


def train_simple(name, model, X_train, y_train, X_test, y_test):
    print(f"\n  [{name}] single pass (no epochs)")
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"    MAE: {mae:.4f}°C")
    gc.collect()
    return model, mae


def train_all():
    df    = load_data()
    # float32 = half the RAM of float64
    X     = df[FEATURE_COLS].values.astype(np.float32)
    y     = df["temperature"].values.astype(np.float32)
    del df; gc.collect()

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    del X, y; gc.collect()

    print(f"Dataset: {len(X_train)} train / {len(X_test)} test rows")
    print(f"Epochs: {EPOCHS}  |  Total trees: {TREES_TOTAL}")
    print("=" * 50)

    summary = {}

    # ── Train one model at a time, save immediately, then free RAM ──

    for train_fn, fname in [
        (lambda: train_rf(X_train, y_train, X_test, y_test),  "random_forest"),
        (lambda: train_gb(X_train, y_train, X_test, y_test),  "gradient_boosting"),
        (lambda: train_xgb(X_train, y_train, X_test, y_test), "xgboost"),
    ]:
        model, mae = train_fn()
        joblib.dump(model, MODELS_DIR / f"model_{fname}.pkl")
        summary[fname] = mae
        free(model)

    for fname, mdl in [
        ("decision_tree",     DecisionTreeRegressor(max_depth=12, random_state=42)),
        ("linear_regression", LinearRegression()),
        ("ridge",             Ridge(alpha=1.0)),
        ("knn",               KNeighborsRegressor(n_neighbors=5, n_jobs=1)),
    ]:
        model, mae = train_simple(fname, mdl, X_train, y_train, X_test, y_test)
        joblib.dump(model, MODELS_DIR / f"model_{fname}.pkl")
        summary[fname] = mae
        free(model)

    # ── Final summary ──
    print("\n" + "=" * 50)
    print(f"{'Model':<25} {'MAE':>8}")
    print("-" * 35)
    for name, mae in sorted(summary.items(), key=lambda x: x[1]):
        print(f"{name:<25} {mae:>8.4f}°C")

    best = min(summary, key=summary.get)
    best_model = joblib.load(MODELS_DIR / f"model_{best}.pkl")
    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    free(best_model)
    print(f"\nBest: {best} (MAE {summary[best]:.4f}°C) → saved as model.pkl")


if __name__ == "__main__":
    train_all()
