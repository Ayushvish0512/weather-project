"""
preflight.py — Hardware & Environment Preflight Check

Run this ONCE before:
  - Training any model (ml/train_all.py, ml/train.py)
  - Downloading data (data/bootstrap.py, data/collect.py)
  - Starting the API server (app/main.py)

Usage:
    python preflight.py

What it checks:
  1. Python version (3.9+)
  2. Required pip packages (all libs in requirements.txt)
  3. GPU hardware (nvidia-smi, PyTorch CUDA, XGBoost CUDA)
  4. .env file and required environment variables
  5. Database connectivity (PostgreSQL via DATABASE_URL)
  6. Data directory (CSV files present)
  7. Disk space (warn if < 1 GB free)
  8. RAM (warn if < 2 GB available)

Exit codes:
  0 — all checks passed (or only warnings)
  1 — one or more critical checks failed
"""

import sys
import os
import subprocess
import importlib
import shutil
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ── Formatting helpers ────────────────────────────────────────────────────────

PASS  = "  ✅"
FAIL  = "  ❌"
WARN  = "  ⚠️ "
INFO  = "  ℹ️ "

_failures  = []
_warnings  = []

def ok(msg):    print(f"{PASS} {msg}")
def fail(msg):  print(f"{FAIL} {msg}"); _failures.append(msg)
def warn(msg):  print(f"{WARN} {msg}"); _warnings.append(msg)
def info(msg):  print(f"{INFO} {msg}")
def section(title): print(f"\n── {title} {'─' * (50 - len(title))}")


# ── 1. Python version ─────────────────────────────────────────────────────────

def check_python():
    section("Python")
    v = sys.version_info
    if v >= (3, 9):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor}.{v.micro} — need 3.9+")


# ── 2. Required packages ──────────────────────────────────────────────────────

REQUIRED_PACKAGES = {
    # import_name       : pip_name
    "fastapi"           : "fastapi",
    "uvicorn"           : "uvicorn",
    "httpx"             : "httpx",
    "psycopg2"          : "psycopg2-binary",
    "requests"          : "requests",
    "pandas"            : "pandas",
    "sklearn"           : "scikit-learn",
    "joblib"            : "joblib",
    "numpy"             : "numpy",
    "xgboost"           : "xgboost",
    "statsmodels"       : "statsmodels",
    "dotenv"            : "python-dotenv",
    "dateutil"          : "python-dateutil",
}

OPTIONAL_PACKAGES = {
    "torch"             : "torch  (GPU neural net support — optional)",
}

def check_packages():
    section("Python Packages")
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pip_name} ({ver})")
        except ImportError:
            fail(f"{pip_name} — NOT installed")
            missing.append(pip_name)

    for import_name, label in OPTIONAL_PACKAGES.items():
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            ok(f"torch ({ver})  [optional — GPU neural nets]")
        except ImportError:
            warn(f"torch not installed — GPU neural net training will be skipped")
            info("Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    if missing:
        info(f"Fix: pip install {' '.join(missing)}")


# ── 3. GPU hardware ───────────────────────────────────────────────────────────

def check_gpu():
    section("GPU Hardware")

    # nvidia-smi
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
            shell=True, stderr=subprocess.DEVNULL
        ).decode().strip()
        ok(f"nvidia-smi — {out}")
    except Exception:
        warn("nvidia-smi not found — no NVIDIA GPU detected or driver not installed")
        info("If you have an NVIDIA GPU, install drivers from https://www.nvidia.com/drivers")
        return  # no point checking CUDA libs without a GPU

    # PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            ok(f"PyTorch CUDA — {name}  ({mem} MB VRAM)")
        else:
            warn("PyTorch installed but CUDA not available — CPU-only build")
            info("Fix: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        warn("PyTorch not installed — GPU neural net training unavailable")
        info("Fix: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    # XGBoost CUDA (actual fit test — ground truth)
    try:
        import json, warnings as _w
        import numpy as np
        from xgboost import XGBRegressor

        X = np.random.rand(10, 3).astype("float32")
        y = np.random.rand(10).astype("float32")

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            m = XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
            m.fit(X, y)

        cfg  = json.loads(m.get_booster().save_config())
        dev  = cfg.get("learner", {}).get("generic_param", {}).get("device", "cpu")

        if dev == "cuda":
            import xgboost
            ok(f"XGBoost CUDA — version {xgboost.__version__}")
        else:
            warn("XGBoost installed but running on CPU (not a CUDA build)")
            info("Fix: pip install xgboost==2.0.3")
    except Exception as e:
        warn(f"XGBoost GPU test failed: {e}")
        info("Fix: pip install xgboost==2.0.3")


# ── 4. Environment variables ──────────────────────────────────────────────────

REQUIRED_ENV = {
    "DATABASE_URL"        : "PostgreSQL connection string",
    "OPENWEATHER_API_KEY" : "OpenWeather API key (for data/collect.py)",
}

OPTIONAL_ENV = {
    "WEATHER_CITY"  : "Gurgaon",
    "MODEL_VERSION" : "v1",
    "WEBHOOK_URL"   : "(optional) n8n webhook URL",
}

def check_env():
    section(".env Variables")

    env_file = ROOT / ".env"
    if not env_file.exists():
        fail(".env file not found — copy .env.example to .env and fill in values")
        return

    ok(".env file found")

    for key, desc in REQUIRED_ENV.items():
        val = os.getenv(key, "")
        if val and val not in ("your_key_here", "postgresql://user:password@host/dbname"):
            ok(f"{key} — set")
        else:
            fail(f"{key} — missing or placeholder  ({desc})")

    for key, default in OPTIONAL_ENV.items():
        val = os.getenv(key, "")
        if val:
            ok(f"{key} = {val}")
        else:
            warn(f"{key} not set — will use default: {default}")


# ── 5. Database connectivity ──────────────────────────────────────────────────

def check_database():
    section("Database")

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        warn("DATABASE_URL not set — skipping DB check")
        return

    try:
        import psycopg2
        conn = psycopg2.connect(db_url, connect_timeout=5)
        cur  = conn.cursor()
        cur.execute("SELECT version();")
        ver  = cur.fetchone()[0].split(",")[0]
        conn.close()
        ok(f"PostgreSQL connected — {ver}")
    except Exception as e:
        fail(f"PostgreSQL connection failed: {e}")
        info("Check DATABASE_URL in .env and ensure the DB is running")


# ── 6. Data files ─────────────────────────────────────────────────────────────

def check_data():
    section("Training Data (CSV files)")

    data_dir = ROOT / "data"
    csvs = sorted(data_dir.glob("weather_*.csv"))

    if not csvs:
        warn("No weather CSV files found in data/")
        info("Run: python data/bootstrap.py")
        return

    total_rows = 0
    for f in csvs:
        try:
            import pandas as pd
            n = sum(1 for _ in open(f)) - 1  # fast line count
            total_rows += n
            ok(f"{f.name}  ({n:,} rows)")
        except Exception:
            warn(f"{f.name} — could not read")

    info(f"Total rows across all CSVs: {total_rows:,}")
    if total_rows < 10_000:
        warn("Less than 10,000 rows — model accuracy may be low. Run data/bootstrap.py to download more history.")


# ── 7. Disk space ─────────────────────────────────────────────────────────────

def check_disk():
    section("Disk Space")

    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / (1024 ** 3)

    if free_gb >= 1.0:
        ok(f"{free_gb:.1f} GB free")
    else:
        warn(f"Only {free_gb:.2f} GB free — training may fail if disk fills up")


# ── 8. RAM ────────────────────────────────────────────────────────────────────

def check_ram():
    section("RAM")

    try:
        import psutil
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024 ** 3)
        total_gb = mem.total    / (1024 ** 3)

        if avail_gb >= 2.0:
            ok(f"{avail_gb:.1f} GB available  /  {total_gb:.1f} GB total")
        else:
            warn(f"Only {avail_gb:.1f} GB available — training may be slow or crash")
            info("Close other applications before training, or reduce TREES_TOTAL in train_all.py")
    except ImportError:
        warn("psutil not installed — cannot check RAM")
        info("Install: pip install psutil  (optional but recommended)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Preflight Check — Weather ML Project")
    print("=" * 55)

    check_python()
    check_packages()
    check_gpu()
    check_env()
    check_database()
    check_data()
    check_disk()
    check_ram()

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)

    if _failures:
        print(f"❌  {len(_failures)} critical issue(s) found — fix before proceeding:\n")
        for f in _failures:
            print(f"    • {f}")
        if _warnings:
            print(f"\n⚠️   {len(_warnings)} warning(s) — non-blocking but worth reviewing.")
        print()
        sys.exit(1)
    elif _warnings:
        print(f"⚠️   All critical checks passed.  {len(_warnings)} warning(s) — see above.")
        print("\n  Safe to proceed with training and data download.\n")
        sys.exit(0)
    else:
        print("✅  All checks passed — environment is fully ready.\n")
        print("  Next steps:")
        print("    python data/bootstrap.py          # download historical data")
        print("    python ml/train_all.py            # train all models")
        print("    python ml/predict.py              # generate prediction")
        sys.exit(0)


if __name__ == "__main__":
    main()
