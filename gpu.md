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


---

## Colab GPU Acceleration — cuML Reference

These are Colab's recommended snippets for accelerating sklearn with GPU via `cuml.accel`. Requires a Colab GPU runtime (T4 or better). The magic `%load_ext cuml.accel` must come before any sklearn imports — it patches sklearn to route supported operations to RAPIDS cuML on the GPU transparently.

> Note: `cuml.accel` is only available in Colab GPU environments and RAPIDS installations. It does not work on local CPU-only machines or Render.

---

### K-Means Segmentation

```python
%load_ext cuml.accel
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

n_samples = 1000
n_features = 2
n_clusters = 3

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
kmeans = KMeans(n_clusters=n_clusters, max_iter=100)
kmeans.fit(X)
labels = kmeans.labels_
print(silhouette_score(X, labels))
```

---

### K-Nearest Neighbors Classifier

```python
%load_ext cuml.accel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=100000, n_features=100, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
```

> Relevant to this project: KNN is one of the 7 trained models. On CPU it's the slowest at inference time (scans all training rows). GPU acceleration via cuML would make it viable for larger datasets.

---

### Logistic Regression Classifier

```python
%load_ext cuml.accel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000000, n_features=200, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

### PCA Dimensionality Reduction

```python
%load_ext cuml.accel
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
```

> Potential use in this project: PCA could reduce the 24 feature dimensions before training, removing correlated features like `feels_like` + `temp_lag_1h` that carry redundant information.

---

### Random Forest Classification

```python
%load_ext cuml.accel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=100000, n_features=100, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=1.0, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

> Directly applicable: this project's best model is RandomForest. Adding `%load_ext cuml.accel` in Colab before training would accelerate it on GPU without any code changes.

---

### UMAP Dimensionality Reduction

```python
%load_ext cuml.accel
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import umap

X, y = make_classification(n_samples=100000, n_features=20, n_classes=5, n_informative=5, random_state=0)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

umap_model = umap.UMAP(n_neighbors=15, n_components=2, random_state=42, min_dist=0.0)
X_train_umap = umap_model.fit_transform(X_train)

plt.figure(figsize=(10, 8))
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap='Spectral', s=10)
plt.colorbar(label="Activity")
plt.title("UMAP projection")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.show()
```

> UMAP could be used to visualise the 24-feature weather dataset in 2D — useful for spotting seasonal clusters or anomalous weather events in the training data.

---

### How to Use in This Project (Colab)

To accelerate `weather_analysis.ipynb` training on Colab GPU:

1. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
2. Add at the very top of the training cell, before any sklearn import:

```python
%load_ext cuml.accel
```

3. Run normally — cuML intercepts RandomForest, KNN, Ridge, and PCA calls and routes them to GPU. No other code changes needed.

The `%load_ext cuml.accel` approach only works in Colab/RAPIDS environments. The local `ml/train_all.py` pipeline uses XGBoost `device="cuda"` and PyTorch CUDA for GPU acceleration instead — those are the correct paths for local GTX 1650 usage.
