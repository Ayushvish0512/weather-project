# GPU Enforcement Layer — PRD

---

## Problem Statement

Code alone cannot force GPU usage. If CUDA-enabled library builds are not installed, the pipeline silently falls back to CPU with no warning. This means:

- `nvidia-smi` shows a healthy GPU
- Training still runs on CPU
- You never know it failed

This document defines the requirements for a GPU enforcement layer that validates the environment at startup, fails loudly when GPU is requested but unavailable, and tells you exactly what to install.

---

## Root Causes (Why GPU "Doesn't Work")

| # | Cause | Likelihood |
|---|---|---|
| 1 | CPU-only `torch` or `xgboost` installed via plain `pip install` | 90% of cases |
| 2 | XGBoost installed without CUDA build | Very common |
| 3 | CUDA version mismatch between driver and ML lib | Less likely if driver is recent |
| 4 | Code silently catching GPU errors and falling back to CPU | Always present without this layer |

---

## Hardware Context

- GPU: NVIDIA GTX 1650
- Driver: supports CUDA 13.1
- ML libs internally use CUDA 11.x / 12.x — this is fine as long as correct builds are installed

### What GTX 1650 helps with

| Model | GPU benefit |
|---|---|
| XGBoost | Yes — `device="cuda"` |
| PyTorch models (LSTM, etc.) | Yes |
| RandomForest / sklearn | No — CPU only, always |
| GradientBoosting (sklearn) | No — CPU only, always |

---

## Requirements

### R1 — GPU Validator Function

A `validate_gpu_or_exit(require_gpu=False)` function must exist in `ml/train_all.py` (or a shared utility).

It must check:

1. `nvidia-smi` is callable and exits 0
2. PyTorch is installed and `torch.cuda.is_available()` returns `True`
3. XGBoost can actually run a tiny fit on `device="cuda"` without error, and the saved config contains `"cuda"`

If any check fails:
- Print each failing check clearly
- Print the exact `pip install` commands needed to fix it
- If `require_gpu=True`, raise `RuntimeError` and stop execution
- If `require_gpu=False`, print warnings and continue on CPU

### R2 — Called at Pipeline Start

`validate_gpu_or_exit()` must be called at the top of `train_all()` before any model training begins.

### R3 — FORCE_GPU Flag

A module-level flag `FORCE_GPU = False` must exist. When set to `True`, it passes `require_gpu=True` to the validator, causing hard failure if GPU is not available.

### R4 — Visible Device Logging During Training

Every model training block must print which device it is using:

```
🚀 USING GPU  (xgboost)
🐌 USING CPU  (random_forest)
```

### R5 — Correct Install Commands in Error Output

When GPU is not configured, the error output must print:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install xgboost==2.0.3
```

---

## Implementation

### Step 1 — GPU Validator

Add to `ml/train_all.py` (or `ml/gpu_utils.py`):

```python
def validate_gpu_or_exit(require_gpu=False):
    issues = []

    # Check 1: nvidia-smi
    try:
        import subprocess
        subprocess.check_output("nvidia-smi", shell=True)
    except Exception:
        issues.append("NVIDIA GPU not detected (nvidia-smi failed)")

    # Check 2: PyTorch CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("PyTorch installed but CUDA not available")
    except ImportError:
        issues.append("PyTorch not installed")

    # Check 3: XGBoost GPU (actual fit test)
    try:
        from xgboost import XGBRegressor
        import numpy as np

        X = np.random.rand(10, 3).astype("float32")
        y = np.random.rand(10).astype("float32")

        model = XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
        model.fit(X, y)

        config = model.get_booster().save_config()
        if "cuda" not in config:
            issues.append("XGBoost installed but NOT GPU-enabled")
    except Exception:
        issues.append("XGBoost GPU not working")

    if issues:
        print("\n🚨 GPU NOT PROPERLY CONFIGURED:\n")
        for issue in issues:
            print(f"  - {issue}")
        print("\n✅ FIX BY INSTALLING:\n")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("  pip install xgboost==2.0.3")
        print()
        if require_gpu:
            raise RuntimeError("GPU required but not available. See install commands above.")
    else:
        print("✅ GPU fully functional!\n")
```

### Step 2 — Call at Pipeline Start

```python
FORCE_GPU = False  # set True to hard-fail if GPU unavailable

def train_all():
    validate_gpu_or_exit(require_gpu=FORCE_GPU)
    # ... rest of training
```

### Step 3 — Device Detection Helper

```python
def detect_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
```

### Step 4 — Per-Model Device Logging

```python
device = detect_gpu()

# XGBoost (GPU-capable)
xgb_params = {"device": device, "tree_method": "hist"}
print(f"  🚀 USING GPU  (xgboost)" if device == "cuda" else "  🐌 USING CPU  (xgboost)")

# sklearn models (always CPU)
print("  🐌 USING CPU  (random_forest)")
```

---

## Expected Terminal Output

### Case 1 — GPU fully working

```
✅ GPU fully functional!

  🚀 USING GPU  (xgboost)
  🐌 USING CPU  (random_forest)
  🐌 USING CPU  (gradient_boosting)
  ...
```

### Case 2 — GPU not configured (most common first run)

```
🚨 GPU NOT PROPERLY CONFIGURED:

  - PyTorch installed but CUDA not available
  - XGBoost installed but NOT GPU-enabled

✅ FIX BY INSTALLING:

  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install xgboost==2.0.3
```

---

## How to Confirm GPU is Actually Being Used

After installing correct builds and running training, open a second terminal and run:

```
nvidia-smi -l 1
```

While training runs, you should see:
- Memory usage increase (500 MB+)
- GPU Util > 10% during XGBoost training

If GPU Util stays at 0%, the CPU-only build is still active.

---

## Install Reference

### PyTorch (CUDA 12.1 build)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### XGBoost (GPU-enabled)

```
pip install xgboost==2.0.3
```

> Plain `pip install xgboost` installs the CPU build. You must use the versioned install above which includes CUDA support.

---

## Risks

| Risk | Mitigation |
|---|---|
| CUDA version mismatch | Validator catches it via actual fit test, not just import check |
| Silent CPU fallback | `require_gpu=True` mode raises hard error |
| XGBoost config check unreliable | Actual `model.fit()` on `device="cuda"` is the ground truth test |
| sklearn models never use GPU | Documented clearly — no false expectations |

---

## Definition of Done

- `validate_gpu_or_exit()` runs at start of `train_all()`
- Failed GPU checks print exact install commands
- `FORCE_GPU = True` causes hard stop if GPU unavailable
- Each model prints which device it is using
- `nvidia-smi -l 1` shows GPU memory usage during XGBoost training

