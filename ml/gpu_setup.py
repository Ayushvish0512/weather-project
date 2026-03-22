"""
ml/gpu_setup.py

GPU availability checker and environment enforcer.

Usage:
    from ml.gpu_setup import validate_gpu, detect_gpu, GPU_STATUS

    # At pipeline start — warns but continues if GPU missing
    validate_gpu(require_gpu=False)

    # Hard fail if GPU is required
    validate_gpu(require_gpu=True)

    # Get device string for model params
    device = detect_gpu()   # returns "cuda" or "cpu"

Run standalone to diagnose your GPU setup:
    python ml/gpu_setup.py
"""

import subprocess
import sys

# ── Public state set after validate_gpu() runs ──────────────────────────────
GPU_STATUS = {
    "nvidia_smi":    False,
    "torch_cuda":    False,
    "xgboost_cuda":  False,
    "device":        "cpu",
}

INSTALL_COMMANDS = [
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "pip install xgboost==2.0.3",
]


# ── Internal checks ──────────────────────────────────────────────────────────

def _check_nvidia_smi() -> tuple[bool, str]:
    try:
        subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.DEVNULL)
        return True, None
    except Exception:
        return False, "NVIDIA GPU not detected (nvidia-smi failed)"


def _check_torch_cuda() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, f"PyTorch CUDA OK — {name}"
        return False, "PyTorch installed but CUDA not available (CPU-only build?)"
    except ImportError:
        return False, "PyTorch not installed"


def _check_xgboost_cuda() -> tuple[bool, str]:
    try:
        import json
        import warnings
        import numpy as np
        from xgboost import XGBRegressor

        X = np.random.rand(10, 3).astype("float32")
        y = np.random.rand(10).astype("float32")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
            m.fit(X, y)

        cfg = json.loads(m.get_booster().save_config())
        device = cfg.get("learner", {}).get("generic_param", {}).get("device", "cpu")

        if device == "cuda":
            return True, "XGBoost CUDA OK"
        return False, "XGBoost installed but running on CPU (not a CUDA build)"
    except Exception as e:
        return False, f"XGBoost GPU test failed: {e}"


# ── Public API ───────────────────────────────────────────────────────────────

def detect_gpu() -> str:
    """
    Fast device check — returns 'cuda' or 'cpu'.
    Does NOT print anything. Use validate_gpu() for full diagnostics.
    """
    try:
        import json, warnings, numpy as np
        from xgboost import XGBRegressor
        X = np.ones((4, 2), dtype="float32")
        y = np.ones(4, dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
            m.fit(X, y)
        cfg = json.loads(m.get_booster().save_config())
        device = cfg.get("learner", {}).get("generic_param", {}).get("device", "cpu")
        return "cuda" if device == "cuda" else "cpu"
    except Exception:
        return "cpu"


def validate_gpu(require_gpu: bool = False) -> dict:
    """
    Run all GPU checks, print results, and update GPU_STATUS.

    Args:
        require_gpu: If True, raises RuntimeError when GPU is not available.

    Returns:
        GPU_STATUS dict with keys: nvidia_smi, torch_cuda, xgboost_cuda, device
    """
    print("\n── GPU Environment Check ──────────────────────────────")

    issues = []
    info   = []

    # Check 1: nvidia-smi
    ok, msg = _check_nvidia_smi()
    GPU_STATUS["nvidia_smi"] = ok
    if ok:
        info.append("  ✅ nvidia-smi OK")
    else:
        issues.append(msg)
        print(f"  ❌ {msg}")

    # Check 2: PyTorch CUDA
    ok, msg = _check_torch_cuda()
    GPU_STATUS["torch_cuda"] = ok
    if ok:
        info.append(f"  ✅ {msg}")
    else:
        issues.append(msg)
        print(f"  ❌ {msg}")

    # Check 3: XGBoost CUDA (actual fit test — ground truth)
    ok, msg = _check_xgboost_cuda()
    GPU_STATUS["xgboost_cuda"] = ok
    if ok:
        info.append(f"  ✅ {msg}")
    else:
        issues.append(msg)
        print(f"  ❌ {msg}")

    # Print passing checks
    for line in info:
        print(line)

    if issues:
        print("\n🚨 GPU NOT PROPERLY CONFIGURED")
        print("   The following issues were found:\n")
        for issue in issues:
            print(f"   • {issue}")

        print("\n   ✅ Fix by running:\n")
        for cmd in INSTALL_COMMANDS:
            print(f"   {cmd}")
        print()

        GPU_STATUS["device"] = "cpu"

        if require_gpu:
            raise RuntimeError(
                "GPU required (FORCE_GPU=True) but not available. "
                "Run the install commands above and retry."
            )
        else:
            print("   ⚠️  Continuing on CPU. Set FORCE_GPU=True to hard-fail instead.\n")
    else:
        GPU_STATUS["device"] = "cuda"
        print("\n✅ GPU fully functional — training will use CUDA\n")

    print("───────────────────────────────────────────────────────\n")
    return GPU_STATUS


def device_label(model_name: str, device: str) -> str:
    """Return a printable device label for a model."""
    # sklearn models are always CPU — don't mislead
    CPU_ONLY = {"random_forest", "gradient_boosting", "decision_tree",
                "linear_regression", "ridge", "knn"}
    if model_name in CPU_ONLY:
        return "🐌 CPU  (sklearn — always CPU)"
    return "🚀 GPU (CUDA)" if device == "cuda" else "🐌 CPU"


# ── Standalone diagnostic ────────────────────────────────────────────────────

if __name__ == "__main__":
    status = validate_gpu(require_gpu=False)
    print("GPU_STATUS:", status)

    if status["device"] == "cuda":
        print("\nYour environment is GPU-ready.")
        print("Run: python ml/train_all.py")
    else:
        print("\nYour environment is NOT GPU-ready.")
        print("Install the packages above, then re-run: python ml/gpu_setup.py")
        print("\nTo verify GPU is actually being used during training:")
        print("  nvidia-smi -l 1   (watch GPU memory + utilisation rise)")
