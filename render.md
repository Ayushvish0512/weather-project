# Render Deployment Guide

How to deploy the Weather ML FastAPI to Render and get live temperature predictions.

---

## What Gets Deployed

```
Your machine                          Render (cloud)
─────────────────                     ──────────────────────────
ml/train_all.py  →  model.pkl   →    FastAPI (app/main.py)
data/bootstrap.py → weather_*.csv →  (CSVs committed to Git)
```

Render runs the FastAPI server only. All training and data download happens on your machine. You push the trained `model.pkl` and CSV files to Git, Render pulls them.

---

## Why This Approach

Render free tier has:
- No persistent disk (files written at runtime are lost on restart)
- ~512 MB RAM — not enough to train models
- No GPU

So the split is:
- Train locally → push `model.pkl` to Git
- Render serves predictions using the committed model
- CSVs are committed to Git so feature engineering works at prediction time

---

## Step 1 — Prepare Locally

### 1a. Train the model

```powershell
.\venv\Scripts\python.exe ml/train_all.py
```

This produces `ml/model.pkl` — the best model across all 7 algorithms.

### 1b. Make sure CSVs are up to date

```powershell
.\venv\Scripts\python.exe data/bootstrap.py
```

The latest CSV (e.g. `data/weather_2026-01-01_2026-03-22.csv`) is needed on Render so `get_latest_features()` can engineer lag/rolling features for prediction.

### 1c. Commit model and CSVs to Git

```powershell
git add ml/model.pkl data/weather_*.csv
git commit -m "update model and data"
git push
```

> If `model.pkl` is in `.gitignore`, remove that line — Render needs it.

---

## Step 2 — Render Setup

A `render.yaml` file is included in the repo. Render reads it automatically and pre-fills the build command, start command, region, and non-secret env vars. You only need to set `DATABASE_URL` and optionally `WEBHOOK_URL` manually.

### 2a. Create a Web Service on Render

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo
3. Render detects `render.yaml` and pre-fills everything — review and click Deploy

If you prefer to set fields manually instead of using `render.yaml`:

| Field | Value |
|---|---|
| Name | `weather-ml` (or anything) |
| Region | Singapore or closest to India |
| Branch | `main` |
| Runtime | Python 3 |
| Build Command | `pip install -r render_requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

**Build Command** — installs only the minimal inference dependencies:
```
pip install -r render_requirements.txt
```

**Start Command** — launches the FastAPI server on the port Render assigns:
```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

> Do NOT use `--reload` on Render — that's for local dev only. Render sets `$PORT` automatically, do not hardcode a port number.

### 2b. Set Environment Variables on Render

Go to your service → Environment → Add these:

| Key | Value | Required |
|---|---|---|
| `DATABASE_URL` | Internal PostgreSQL URL from Render DB | ✅ Yes |
| `MODEL_VERSION` | `v1` | ✅ Yes |
| `WEATHER_CITY` | `Gurgaon` | Optional |
| `WEBHOOK_URL` | `https://n8n-29o4.onrender.com/webhook/weather` | Optional |

> `OPENWEATHER_API_KEY` is only needed if `data/collect.py` runs on Render. For this setup it's not required.

Where to find these on Render:

```
Render Dashboard
  → Your Web Service
    → Environment (left sidebar)
      → Add Environment Variable
```

For `DATABASE_URL` specifically:
```
Render Dashboard
  → Your PostgreSQL service
    → Connections tab
      → Internal Database URL   ← copy this (not External URL)
```

The Internal URL only works between Render services in the same region. It's faster and doesn't count against bandwidth limits.

---

## Step 3 — How Prediction Works on Render

When a request hits `GET /predict/next-hour`:

```
Request arrives
      ↓
app/predict.py: load_model()
      → loads ml/model.pkl from disk (committed to Git, always present)
      ↓
app/predict.py: get_latest_features()
      → reads last 30 rows from data/weather_YYYY-MM-DD.csv
      → engineers 24 features (lag, rolling, time, derived)
      → returns feature vector for current hour
      ↓
model.predict(features)
      → returns predicted temperature in °C
      ↓
insert_prediction(datetime, temp, model_version)
      → stores in weather_predictions table on Render PostgreSQL
      ↓
Returns JSON response
```

### What "current time" means

The prediction is always for the **next full hour** from UTC now:

```python
prediction_for = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
```

So if you call the API at 14:35 UTC, it predicts temperature at 15:00 UTC (= 20:30 IST).

To get the current hour's prediction, call at any time — the model uses the last available CSV row's features, which represent conditions right now.

---

## Step 4 — Test the Deployed API

Replace `your-service.onrender.com` with your actual Render URL.

### Health check
```
GET https://your-service.onrender.com/
```
Expected: `{"status": "Weather ML API is running"}`

### Predict next hour
```
GET https://your-service.onrender.com/predict/next-hour
```
Expected:
```json
{
  "prediction_for": "2026-03-22T15:00:00Z",
  "predicted_temp_c": 28.4,
  "summary": "Warm, Clear",
  "model_version": "v1"
}
```

### Predict next 6 hours
```
GET https://your-service.onrender.com/predict/hours?hours=6
```

### Predict all remaining hours today
```
GET https://your-service.onrender.com/predict/today
```

### Swagger UI (interactive docs)
```
https://your-service.onrender.com/docs
```

---

## Step 5 — Weekly Update Cycle

Every Sunday (or whenever you want fresh predictions):

```powershell
# 1. Download new data up to yesterday
.\venv\Scripts\python.exe data/bootstrap.py

# 2. Retrain all models, pick best
.\venv\Scripts\python.exe ml/train_all.py

# 3. Commit updated model + latest CSV
git add ml/model.pkl data/weather_*.csv
git commit -m "weekly retrain $(Get-Date -Format yyyy-MM-dd)"
git push
```

Render auto-deploys on push. The new `model.pkl` is live within ~2 minutes.

---

## Render Free Tier Constraints

| Constraint | Impact | Workaround |
|---|---|---|
| Sleeps after 15 min inactivity | First request after sleep takes 30–60s cold start | Acceptable — n8n webhook keeps it warm |
| ~512 MB RAM | Cannot train models on Render | Train locally, push model.pkl |
| No persistent disk | Files written at runtime are lost on restart | All needed files committed to Git |
| No GPU | Prediction only needs CPU — model.pkl is already trained | No impact |
| 750 free hours/month | ~31 days of continuous uptime | One service stays free |

---

## Troubleshooting

**`503 — Model not trained`**
→ `ml/model.pkl` is missing from the repo. Run `ml/train_all.py` locally and commit it.

**`503 — No CSV data`**
→ No `data/weather_*.csv` files committed. Run `data/bootstrap.py` locally and commit the CSVs.

**`503 — Not enough rows to compute lag features`**
→ The latest CSV has fewer than 30 rows. Commit a more complete CSV.

**Predictions are stale (same value every hour)**
→ The CSV committed to Git is old. Run `data/bootstrap.py` and push updated CSVs.

**Database connection error**
→ Check `DATABASE_URL` in Render environment variables. Use the Internal URL (not External) for Render-to-Render connections.

**Cold start timeout**
→ Normal on free tier. First request after 15 min idle takes up to 60s. Subsequent requests are fast.

---

## File Checklist Before Deploying

```
✅ ml/model.pkl               — trained model (committed to Git)
✅ data/weather_*.csv         — historical data for feature engineering
✅ render_requirements.txt    — minimal Render dependencies (no torch/sklearn/xgboost)
✅ app/main.py                — FastAPI entry point
✅ app/predict.py             — prediction endpoints
✅ app/webhook.py             — webhook endpoints
✅ db/postgres.py             — DB connection using DATABASE_URL from env
✅ ml/preprocess.py           — feature engineering (needed at prediction time)
❌ .env                       — NOT committed — use Render environment variables instead
❌ venv/                      — NOT committed — Render builds its own
```
