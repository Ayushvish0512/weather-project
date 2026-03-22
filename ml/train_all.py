"""
ml/train_all.py
RAM-safe multi-model training with GPU enforcement layer.

GPU behaviour:
- Imports gpu_setup.py for full environment validation at startup
- FORCE_GPU = False  → warns if GPU missing, continues on CPU
- FORCE_GPU = True   → hard-fails if GPU not available
- XGBoost uses CUDA when available; sklearn models always run on CPU
- PyTorch MLP competes as a GPU model when torch is installed
- Each model prints which device it is using before training

RAM safety:
- Models trained and saved one at a time (never held in memory together)
- gc.collect() + del between each model
- numpy arrays kept as float32 (half the RAM of float64)
- XGBoost uses hist tree method (low RAM)
- RF/GB use max_samples + max_features to limit per-tree memory

Data freshness:
- Checks if CSVs contain data up to yesterday (D-1) before training
- Prompts to download missing data or proceed with what exists
"""
import os
import gc
import sys
import joblib
import numpy as np
from datetime import date, timedelta
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
        Wraps a trained MLP + scalers into a sklearn-compatible predict(X) interface.
        Handles both X feature scaling and y target un-scaling so output is always °C.
        """
        def __init__(self, model: MLP, scaler: StandardScaler,
                     y_mean: float = 0.0, y_std: float = 1.0):
            self.model  = model
            self.scaler = scaler
            self.y_mean = y_mean
            self.y_std  = y_std

        def predict(self, X: np.ndarray) -> np.ndarray:
            import torch
            self.model.eval()
            Xs = self.scaler.transform(X)
            t  = torch.tensor(Xs, dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = self.model(t).squeeze(1).numpy()
            return pred_scaled * self.y_std + self.y_mean  # un-scale to °C

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


def _latest_csv_date():
    """Return the latest recorded_at date found across all CSVs (fast — reads last row only)."""
    import pandas as pd
    csvs = sorted((ROOT / "data").glob("weather_*.csv"))
    if not csvs:
        return None
    df_tail = pd.read_csv(csvs[-1], usecols=["recorded_at"]).tail(1)
    if df_tail.empty:
        return None
    return pd.to_datetime(df_tail["recorded_at"].iloc[0]).date()


def _check_data_freshness():
    """
    Check if CSVs contain data up to yesterday (D-1).
    Prompts the user: Y = run bootstrap first, N = train on existing data.
    """
    yesterday = date.today() - timedelta(days=1)
    latest    = _latest_csv_date()

    if latest is None:
        print("No CSV files found in data/. Running bootstrap to download data...")
        _run_bootstrap()
        return

    gap_days = (yesterday - latest).days

    if gap_days <= 0:
        print(f"✅ Data is up to date — latest row: {latest}  (yesterday: {yesterday})")
        return

    print(f"\n⚠️  Data is {gap_days} day(s) behind.")
    print(f"   Latest row in CSVs : {latest}")
    print(f"   Yesterday (D-1)    : {yesterday}")
    print(f"   Missing            : {gap_days} day(s) of hourly data\n")

    answer = input("Download missing data before training? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        _run_bootstrap()
    else:
        print(f"Proceeding with existing data (up to {latest}). Training may miss recent patterns.\n")


def _run_bootstrap():
    import subprocess
    print("Running data/bootstrap.py ...\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "data" / "bootstrap.py")],
        cwd=str(ROOT)
    )
    if result.returncode != 0:
        raise RuntimeError("Bootstrap failed. Run data/bootstrap.py manually to debug.")
    print("Bootstrap complete. Continuing with training...\n")


def load_data():
    """Check data freshness, optionally sync, then load all CSVs."""
    _check_data_freshness()
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

        # Scale both features AND target — critical for MLP convergence
        # Without target scaling, MSELoss operates on raw °C values (~30²=900)
        # causing exploding gradients and slow convergence
        X_scaler = StandardScaler()
        y_mean   = float(y_train.mean())
        y_std    = float(y_train.std()) or 1.0

        X_train_s = X_scaler.fit_transform(X_train)
        X_test_s  = X_scaler.transform(X_test)
        y_train_s = (y_train - y_mean) / y_std   # normalise target to ~N(0,1)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n  [{name}] {device_label(name, dev)}")

        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(dev)
        X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32).to(dev)
        y_train_t = torch.tensor(y_train_s, dtype=torch.float32).unsqueeze(1).to(dev)
        y_test_t  = torch.tensor(y_test,    dtype=torch.float32).unsqueeze(1)  # raw °C for MAE

        net       = MLP(n_features=X_train_t.shape[1]).to(dev)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        # Reduce LR by 0.5 if loss doesn't improve for 20 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20
        )

        TORCH_EPOCHS = 300
        net.train()
        for epoch in range(TORCH_EPOCHS):
            optimizer.zero_grad()
            loss = criterion(net(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            if epoch % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"    Epoch {epoch:>3}/{TORCH_EPOCHS}  loss: {loss.item():.4f}  lr: {lr_now:.2e}")

        # Evaluate — convert scaled predictions back to °C
        net.eval()
        with torch.no_grad():
            pred_scaled = net(X_test_t).cpu()
            pred_celsius = pred_scaled * y_std + y_mean   # un-scale
            mae = torch.mean(torch.abs(pred_celsius - y_test_t)).item()
        print(f"    MAE: {mae:.4f}°C")

        # Wrapper stores X_scaler + y stats so predict() returns °C correctly
        wrapper = TorchRegressorWrapper(
            model=net.cpu(), scaler=X_scaler,
            y_mean=y_mean, y_std=y_std
        )
        return wrapper, mae

    except Exception as e:
        print(f"\n  [{name}] failed: {e}")
        return None, float("inf")

def train_all(only: list[str] | None = None):
    """
    Train models and save the best as model.pkl.

    Args:
        only: list of model names to train. None = train all.
              Valid names: random_forest, gradient_boosting, xgboost,
                           torch_gpu, decision_tree, linear_regression, ridge, knn
    """
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
    if only:
        print(f"Training only: {', '.join(only)}")
    print("=" * 50)

    summary = {}

    all_iterative = [
        (lambda: train_rf(X_train, y_train, X_test, y_test),                 "random_forest"),
        (lambda: train_gb(X_train, y_train, X_test, y_test),                 "gradient_boosting"),
        (lambda: train_xgb(X_train, y_train, X_test, y_test, device=device), "xgboost"),
        (lambda: train_torch_gpu(X_train, y_train, X_test, y_test),          "torch_gpu"),
    ]

    all_simple = [
        ("decision_tree",     DecisionTreeRegressor(max_depth=12, random_state=42)),
        ("linear_regression", Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])),
        ("ridge",             Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("knn",               Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5, n_jobs=-1))])),
    ]

    for train_fn, fname in all_iterative:
        if only and fname not in only:
            continue
        model, mae = train_fn()
        if model is not None:
            joblib.dump(model, MODELS_DIR / f"model_{fname}.pkl")
            summary[fname] = mae
            free(model)

    for fname, mdl in all_simple:
        if only and fname not in only:
            continue
        model, mae = train_simple(fname, mdl, X_train, y_train, X_test, y_test)
        joblib.dump(model, MODELS_DIR / f"model_{fname}.pkl")
        summary[fname] = mae
        free(model)

    if not summary:
        print("\nNo models trained. Check the model names passed via CLI.")
        return

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
    import argparse

    VALID = {"random_forest", "gradient_boosting", "xgboost", "torch_gpu",
             "decision_tree", "linear_regression", "ridge", "knn"}

    parser = argparse.ArgumentParser(
        description="Train weather ML models.",
        epilog=(
            "Examples:\n"
            "  python ml/train_all.py                          # train all\n"
            "  python ml/train_all.py torch_gpu                # one model\n"
            "  python ml/train_all.py random_forest xgboost    # multiple\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "models", nargs="*",
        help=f"Model(s) to train. Valid: {', '.join(sorted(VALID))}. Omit to train all."
    )
    args = parser.parse_args()

    if args.models:
        invalid = set(args.models) - VALID
        if invalid:
            print(f"Unknown model(s): {', '.join(invalid)}")
            print(f"Valid options: {', '.join(sorted(VALID))}")
            sys.exit(1)
        train_all(only=args.models)
    else:
        train_all()
