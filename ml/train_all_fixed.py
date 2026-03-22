"""
ml/train_all.py
100 epochs, RAM-safe:
- Models trained and saved one at a time (never held in memory together)
- gc.collect() + del between each model
- numpy arrays kept as float32 (half the RAM of float64)
- XGBoost uses hist tree method (low RAM)
- RF/GB use max_samples + max_features to limit per-tree memory
- GPU auto-detected: uses CUDA if available, falls back to CPU silently
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, callback
import xgboost
import time

# Ensure project root is on sys.path so ml.preprocess resolves
# regardless of working directory or how the script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.preprocess import load_raw_data, engineer_features, get_features_and_target, FEATURE_COLS

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "ml"

EPOCHS      = 20
TREES_TOTAL = 100
CPU_CORES   = os.cpu_count() or 4


def detect_gpu() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import json, warnings
        _X = np.ones((4, 2), dtype="float32")
        _y = np.ones(4, dtype="float32")
        print("XGBoost build config:")
        print(xgboost.__config__.show())
        print()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _m = XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
            _m.fit(_X, _y)
        cfg = json.loads(_m.get_booster().save_config())
        device = cfg.get("learner", {}).get("generic_param", {}).get("device", "cpu")
        print(f"XGBoost detected device from config: {device}")
        if device == "cuda":
            try:
                import torch
                print(f"Torch CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
            except ImportError:
                print("Torch not installed - install for better GPU checks")
        return device if device == "cuda" else "cpu"
    except Exception:
        return "cpu"


def load_data():
    # Auto-download data if CSVs are missing
    if not list((ROOT / "data").glob("weather_*.csv")):
        print("No CSV files found in data/. Running bootstrap to download data...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "data" / "bootstrap.py")],
            cwd=str(ROOT)
        )
        if result.returncode != 0:
            raise RuntimeError("Bootstrap failed. Check your internet connection and try running data/bootstrap.py manually.")
        print("Bootstrap complete. Continuing with training...\n")

    df = load_raw_data()
    X, y = get_features_and_target(df)
    return X, y


def free(model):
    """Delete model and force garbage collection to free RAM."""
    del model
    gc.collect()


def train_rf(X_train, y_train, X_test, y_test, name="random_forest"):
    trees_per_epoch = TREES_TOTAL // EPOCHS
    model = RandomForestRegressor(
        n_estimators=0, warm_start=True, random_state=42,
        n_jobs=-1,          # use all CPU cores
        max_depth=12,
        max_samples=0.8,    # each tree sees 80% of rows
        max_features=0.8,
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
        subsample=0.8,
        max_features=0.8,
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


def train_xgb(X_train, y_train, X_test, y_test, name="xgboost", device="cpu"):
    print(f"\n  [{name}] {TREES_TOTAL} rounds — printing every {TREES_TOTAL // EPOCHS}  [device: {device}]")

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
        tree_method="hist",
        device=device,
        gpu_id=0 if device == "cuda" else None,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=CPU_CORES,
        callbacks=[EpochPrinter()]
    )
    start_time = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = time.time() - start_time
    print(f"XGBoost training completed in {train_time:.2f}s on device {device}")
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


def train_torch_gpu(X_train, y_train, X_test, y_test, name="torch_gpu"):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        if torch.cuda.is_available():
            device = 'cuda'
            X_train_t = X_train_t.to(device)
            X_test_t = X_test_t.to(device)
            y_train_t = y_train_t.to(device)
            print(f"  [{name}] Torch GPU training on {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print(f"  [{name}] Torch CPU (CUDA not available)")
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(X_train_t.shape[1], 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )
            def forward(self, x):
                return self.layers(x)
        model = MLP().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"    Epoch {epoch} loss: {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t)
            mae = torch.mean(torch.abs(pred.cpu() - y_test_t.cpu())).item()
        print(f"    Test MAE: {mae:.4f}°C")
        return model.cpu(), mae
    except Exception as e:
        print(f"  [{name}] Failed: {e}")
        return None, float('inf')

def train_all():
    device = detect_gpu()
    print(f"XGBoost Device: {device.upper()}  |  CPU cores: {CPU_CORES}")

    X, y = load_data()   # already float32, features engineered
    gc.collect()

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
        (lambda: train_rf(X_train, y_train, X_test, y_test),                  "random_forest"),
        (lambda: train_gb(X_train, y_train, X_test, y_test),                  "gradient_boosting"),
        (lambda: train_xgb(X_train, y_train, X_test, y_test, device=device),  "xgboost"),
        (lambda: train_torch_gpu(X_train, y_train, X_test, y_test),           "torch_gpu"),
    ]:
        model, mae = train_fn()
        if model is not None:
            joblib.dump(model, MODELS_DIR / f"model_{fname}.pkl")
            summary[fname] = mae
            free(model)


    for fname, mdl in [
        ("decision_tree",     DecisionTreeRegressor(max_depth=12, random_state=42)),
        # Linear models and KNN need feature scaling — wrap in Pipeline to fix
        # the ill-conditioned matrix warning and improve accuracy
        ("linear_regression", Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])),
        ("ridge",             Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("knn",               Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5, n_jobs=-1))])),
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
