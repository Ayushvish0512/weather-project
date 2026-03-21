import os
import sys
import joblib
import httpx
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

router = APIRouter(prefix="/webhook", tags=["webhook"])

MODEL_PATH = str(ROOT / "ml" / "model.pkl")

# In-memory store for registered webhook URLs.
_webhook_urls: list[str] = []

_default = os.getenv("WEBHOOK_URL")
if _default:
    _webhook_urls.append(_default)


class WebhookConfig(BaseModel):
    url: str


def _build_prediction(hours_ahead: int = 1) -> dict | None:
    try:
        import glob
        import pandas as pd
        from ml.preprocess import engineer_features, FEATURE_COLS

        model = joblib.load(MODEL_PATH)
        files = sorted(glob.glob(str(ROOT / "data" / "weather_*.csv")))
        if not files:
            return None

        df = pd.read_csv(files[-1]).tail(30).copy()
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        df = df.sort_values("recorded_at").reset_index(drop=True)
        df = engineer_features(df)
        df.dropna(subset=FEATURE_COLS, inplace=True)
        if df.empty:
            return None

        features = df[FEATURE_COLS].iloc[-1].values.astype("float32").reshape(1, -1)
        future_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_ahead)
        temp = model.predict(features)[0]
        return {
            "predicted_temperature": round(float(temp), 2),
            "hours_ahead": hours_ahead,
            "prediction_for": future_time.isoformat() + "Z",
            "sent_at": datetime.utcnow().isoformat() + "Z"
        }
    except Exception:
        return None


async def dispatch_to_webhooks(hours_ahead: int = 1):
    """Send next-hour prediction to all registered webhook URLs."""
    if not _webhook_urls:
        return {"sent": 0, "message": "No webhooks registered"}

    payload = _build_prediction(hours_ahead)
    if payload is None:
        return {"sent": 0, "message": "Model not trained yet"}

    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        for url in _webhook_urls:
            try:
                resp = await client.post(url, json=payload)
                results.append({"url": url, "status": resp.status_code})
            except Exception as e:
                results.append({"url": url, "error": str(e)})

    return {"sent": len(results), "results": results, "payload": payload}


# --- Endpoints ---

@router.post("/register")
def register_webhook(config: WebhookConfig):
    """Register a URL to receive hourly weather predictions."""
    if config.url in _webhook_urls:
        return {"message": "URL already registered", "url": config.url}
    _webhook_urls.append(config.url)
    return {"message": "Webhook registered", "url": config.url}


@router.delete("/unregister")
def unregister_webhook(config: WebhookConfig):
    """Remove a previously registered webhook URL."""
    if config.url not in _webhook_urls:
        raise HTTPException(status_code=404, detail="URL not found")
    _webhook_urls.remove(config.url)
    return {"message": "Webhook removed", "url": config.url}


@router.get("/list")
def list_webhooks():
    """List all registered webhook URLs."""
    return {"webhooks": _webhook_urls}


@router.post("/send")
async def send_now(hours_ahead: int = 1):
    """Manually trigger a prediction push to all registered webhooks."""
    result = await dispatch_to_webhooks(hours_ahead)
    return result
