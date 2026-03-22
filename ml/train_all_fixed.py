"""
ml/train_all_fixed.py
RAM-safe multi-model training with GPU enforcement layer.

GPU behaviour:
- Imports gpu_setup.py for full environment validation at startup
- FORCE_GPU = False  → warns if GPU missing, continues on CPU
- FORCE_GPU = True   → hard-fails if GPU not available
- XGBoost uses CUDA when available; sklearn models always run on CPU
- Each model prints which device it is using before training

RAM safety:
- Models trained and saved one at a time (never held in memory together)
- gc.collect() + del between each model
- numpy arrays kept as float32 (half the RAM of float64)
- XGBoost uses hist tree method (low RAM)
- RF/GB use max_samples + max_features to limit per-tree memory
"""
import os
import gc
import sys
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, callback
import time

# ── PyTorch MLP — defined at module level so joblib/pickle can serialize it ──
# (classes defined inside functions cannot be pickled — Python limitation)
try:
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        """Simple feedforward network for tabular regression."""
        def __init__(self, n_features: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.layers(x)

    class TorchRegressorWrapper:
        """
        Wraps a trained MLP + its StandardScaler into a sklearn-compatible
        interface so it can be saved with joblib and used by ml/predict.py
        via model.predict(X).
        """
        def __init__(self, model: MLP, scaler: StandardScaler):
            self.model  = model
            self.scaler = scaler

        def predict(self, X: np.ndarray) -> np.ndarray:
            import torch
            self.model.eval()
            Xs = self.scaler.transform(X)
            t  = torch.tensor(Xs, dtype=torch.float32)
            with torch.no_grad():
                out = self.model(t).squeeze(1).numpy()
            return out

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False

# ── Project root on sys.path so imports work from any working directory ──────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.preprocess import load_raw_data, get_features_and_target, FEATURE_COLS
from ml.gpu_setup import validate_gpu, detect_gpu, device_label

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "ml"

# ── Config ───────────────────────────────────────────────────────────────────
EPOCHS      = 20
TREES_TOTAL = 100
CPU_CORES   = os.cpu_count() or 4

# Set True to hard-fail if GPU is not available instead of silently using CPU
FORCE_GPU   = False


def load_data():
    """Load CSVs, auto-bootstrap if missing."""
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
    print(f"\n  [{name}] {device_label(name, 'cpu')}  |  {EPOCHS} epochs × {trees_per_epoch} trees = {TREES_TOTAL} total")
    model = RandomForestRegressor(
        n_estimators=0, warm_start=True, random_state=42,
        n_jobs=-1,
        max_depth=12,
        max_samples=0.8,
        max_features=0.8,
    )
    mae = None
    for epoch in range(1, EPOCHS + 1):
        model.n_estimators = epoch * trees_per_epoch
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"    Epoch {epoch:>3}/{EPOCHS} — trees: {model.n_estimators:>4}  MAE: {mae:.4f}°C")
        gc.collect()
    return model, mae


def train_gb(X_train, y_train, X_test, y_test, name="gradient_boosting"):
    trees_per_epoch = TREES_TOTAL // EPOCHS
    print(f"\n  [{name}] {device_label(name, 'cpu')}  |  {EPOCHS} epochs × {trees_per_epoch} trees = {TREES_TOTAL} total")
    model = GradientBoostingRegressor(
        n_estimators=0, warm_start=True, random_state=42,
        max_depth=4,
        subsample=0.8,
        max_features=0.8,
    )
    mae = None
    for epoch in range(1, EPOCHS + 1):
        model.n_estimators = epoch * trees_per_epoch
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"    Epoch {epoch:>3}/{EPOCHS} — trees: {model.n_estimators:>4}  MAE: {mae:.4f}°C")
        gc.collect()
    return model, mae


def train_xgb(X_train, y_train, X_test, y_test, name="xgboost", device="cpu"):
    print(f"\n  [{name}] {device_label(name, device)}  |  {TREES_TOTAL} rounds — printing every {TREES_TOTAL // EPOCHS}")

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
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=CPU_CORES,
        callbacks=[EpochPrinter()]
    )
    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"    Completed in {time.time() - start:.2f}s")
    model.set_params(callbacks=None)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    gc.collect()
    return model, mae


def train_simple(name, model, X_train, y_train, X_test, y_test):
    print(f"\n  [{name}] {device_label(name, 'cpu')}  |  single pass (no epochs)")
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"    MAE: {mae:.4f}°C")
    gc.collect()
    return model, mae


def train_torch_gpu(X_train, y_train, X_test, y_test, name="torch_gpu"):
    if not _TORCH_AVAILABLE:
        print(f"\n  [{name}] skipped — PyTorch not installed")
        print("    Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return None, float("inf")

    try:
        import torch

        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n  [{name}] {device_label(name, dev)}")

        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(dev)
        X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32).to(dev)
        y_train_t = torch.tensor(y_train,   dtype=torch.float32).unsqueeze(1).to(dev)
        y_test_t  = torch.tensor(y_test,    dtype=torch.float32).unsqueeze(1)

        net       = MLP(n_features=X_train_t.shape[1]).to(dev)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        net.train()
        for epoch in range(100):
            optimizer.zero_grad()
            loss = criterion(net(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:>3}  loss: {loss.item():.4f}")

        net.eval()
        with torch.no_grad():
            mae = torch.mean(torch.abs(net(X_test_t).cpu() - y_test_t)).item()
        print(f"    MAE: {mae:.4f}°C")

        # Wrap with scaler so predict(X) works identically to sklearn models
        wrapper = TorchRegressorWrapper(model=net.cpu(), scaler=scaler)
        return wrapper, mae

    except Exception as e:
        print(f"\n  [{name}] failed: {e}")
        return None, float("inf")

def train_all():
    # ── GPU enforcement — runs checks, prints results, fails if FORCE_GPU=True ──
    validate_gpu(require_gpu=FORCE_GPU)

    device = detect_gpu()
    print(f"Device: {device.upper()}  |  CPU cores: {CPU_CORES}")

    X, y = load_data()
    gc.collect()

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    del X, y; gc.collect()

    print(f"Dataset: {len(X_train)} train / {len(X_test)} test rows")
    print(f"Epochs: {EPOCHS}  |  Total trees: {TREES_TOTAL}")
    print("=" * 50)

    summary = {}

    # GPU-capable models
    for train_fn, fname in [
        (lambda: train_rf(X_train, y_train, X_test, y_test),                 "random_forest"),
        (lambda: train_gb(X_train, y_train, X_test, y_test),                 "gradient_boosting"),
        (lambda: train_xgb(X_train, y_train, X_test, y_test, device=device), "xgboost"),
        (lambda: train_torch_gpu(X_train, y_train, X_test, y_test),          "torch_gpu"),
    ]:
        model, mae = train_fn()
        if model is not None:
            joblib.dump(model, MODELS_DIR / f"model_{fname}.pkl")
            summary[fname] = mae
            free(model)

    # CPU-only sklearn models
    for fname, mdl in [
        ("decision_tree",     DecisionTreeRegressor(max_depth=12, random_state=42)),
        ("linear_regression", Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])),
        ("ridge",             Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("knn",               Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5, n_jobs=-1))])),
    ]:
        model, mae = train_simple(fname, mdl, X_train, y_train, X_test, y_test)
        joblib.dump(model, MODELS_DIR / f"model_{fname}.pkl")
        summary[fname] = mae
        free(model)

    # ── Summary ──────────────────────────────────────────────────────────────
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
