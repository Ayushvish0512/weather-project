# Weather Prediction ML Project — Gurgaon, India

---

# Part 1 — How to Use This Repo

## Quick Start

### Step 1 — Create virtual environment

```powershell
py -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

> Always use `.\venv\Scripts\python.exe` — never `python` or `py` directly after setup.

---

### Step 2 — Configure environment

Copy `.env.example` to `.env` and fill in your values:

```env
DATABASE_URL=postgresql://user:password@host/dbname
OPENWEATHER_API_KEY=your_key_here
WEATHER_CITY=Gurgaon
MODEL_VERSION=v1
WEBHOOK_URL=https://n8n-29o4.onrender.com/webhook/weather
```

---

### Step 3 — Initialize database tables

```powershell
.\venv\Scripts\python.exe db/postgres.py
```

Creates three tables: `weather_raw`, `weather_predictions`, `model_metrics`.

---

### Step 4 — Download historical weather data

```powershell
.\venv\Scripts\python.exe data/bootstrap.py
```

Downloads year-by-year CSVs from Open-Meteo archive (2020 → present) into `data/`.
Already downloaded files are skipped automatically on re-run.

To extend the date range, edit `data/bootstrap.py`:

```python
HISTORY_START = date(2020, 1, 1)   # NEVER change — fixed baseline
HISTORY_END   = date(2026, 3, 13)  # extend forward as needed
```

---

### Step 5 — Train a single model (quick)

```powershell
.\venv\Scripts\python.exe ml/train.py
```

Trains RandomForest on all CSVs, saves `ml/model.pkl` and `ml/model_v1.pkl`.

---

### Step 6 — Train all 7 models and compare (recommended)

```powershell
.\venv\Scripts\python.exe ml/train_all.py
```

Trains 7 algorithms with epoch-style progress printed to terminal. Saves each `.pkl` file, then auto-sets the best performer as `model.pkl`.

**What you'll see in the terminal:**

```
Dataset: 43000 train / 10800 test rows
Epochs: 100  |  Total trees: 500
==================================================

  [random_forest] 100 epochs × 5 trees = 500 total
    Epoch   1/100 — trees:    5  MAE: 0.3102°C
    Epoch  10/100 — trees:   50  MAE: 0.2741°C
    Epoch  50/100 — trees:  250  MAE: 0.2589°C
    Epoch 100/100 — trees:  500  MAE: 0.2938°C

  [gradient_boosting] 100 epochs × 5 trees = 500 total
    Epoch   1/100 — trees:    5  MAE: 0.4201°C
    ...
    Epoch 100/100 — trees:  500  MAE: 0.2036°C

  [xgboost] 500 rounds — printing every 50
    Round   50/500  MAE: 0.3201°C
    Round  500/500  MAE: 0.3097°C

  [decision_tree] single pass (no epochs)
    MAE: 0.3988°C

  [linear_regression] single pass (no epochs)
    MAE: 1.3079°C

  [ridge] single pass (no epochs)
    MAE: 1.3080°C

  [knn] single pass (no epochs)
    MAE: 1.5433°C

==================================================
Model                     MAE
-----------------------------------
gradient_boosting      0.2036°C   ← best → saved as model.pkl
random_forest          0.2938°C
xgboost                0.3097°C
decision_tree          0.3988°C
linear_regression      1.3079°C
ridge                  1.3080°C
knn                    1.5433°C
```

---

### Step 7 — Generate a prediction

```powershell
.\venv\Scripts\python.exe ml/predict.py
```

Predicts next-hour temperature using latest CSV data, stores result in DB.

---

### Step 8 — Run the FastAPI server

```powershell
.\venv\Scripts\uvicorn.exe app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger UI.

---

### Step 9 — Evaluate predictions vs actuals

```powershell
.\venv\Scripts\python.exe ml/evaluate.py
```

JOINs stored predictions against actual weather in `weather_raw`, computes MAE/RMSE, stores in `model_metrics`.

---

### Step 10 — Weekly retraining (production cycle)

Run every Sunday to keep the model fresh:

```powershell
.\venv\Scripts\python.exe data/bootstrap.py
.\venv\Scripts\python.exe ml/train_all.py
.\venv\Scripts\python.exe ml/predict.py
.\venv\Scripts\python.exe ml/evaluate.py
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/predict/next-hour` | Predict next hour temp + weather summary |
| GET | `/predict/hours?hours=6` | Forecast for next N hours (max 24) |
| GET | `/predict/today` | All remaining hours of today (UTC) |
| POST | `/train` | Retrain model from CSVs via API |
| POST | `/evaluate` | Evaluate stored predictions vs actuals |
| POST | `/webhook/register` | Register a URL to receive predictions |
| DELETE | `/webhook/unregister` | Remove a registered webhook URL |
| GET | `/webhook/list` | List all registered webhook URLs |
| POST | `/webhook/send?hours_ahead=1` | Manually push prediction to all webhooks |

**Example response — `GET /predict/hours?hours=3`:**

```json
{
  "location": "Gurgaon, IN",
  "generated_at": "2026-03-20T14:00:00Z",
  "model_version": "v1",
  "forecast": [
    { "hour": 1, "prediction_for": "2026-03-20T15:00:00Z", "predicted_temp_c": 18.4, "summary": "Cool, Clear" },
    { "hour": 2, "prediction_for": "2026-03-20T16:00:00Z", "predicted_temp_c": 17.9, "summary": "Cool, Clear" },
    { "hour": 3, "prediction_for": "2026-03-20T17:00:00Z", "predicted_temp_c": 17.2, "summary": "Cool, Clear, Breezy" }
  ]
}
```

---

## Model Files Reference

After running `ml/train_all.py`, these files are saved in `ml/`:

```
model.pkl                    ← always the best performing model (auto-updated)
model_random_forest.pkl
model_gradient_boosting.pkl
model_xgboost.pkl
model_decision_tree.pkl
model_linear_regression.pkl
model_ridge.pkl
model_knn.pkl
model_v1.pkl                 ← versioned snapshot
```

To switch which model is used for predictions, change `MODEL_PATH` in `ml/predict.py`:

```python
MODEL_PATH = str(ROOT / "ml" / "model_xgboost.pkl")
```

---

## Actual Training Results

Trained on 54,000 rows (2020–2026), 24 features, 100 epochs, 500 trees:

```
Model                          MAE
-----------------------------------
gradient_boosting           0.2036°C   ← best
random_forest               0.2938°C
xgboost                     0.3097°C
decision_tree               0.3988°C
linear_regression           1.3079°C
ridge                       1.3080°C
knn                         1.5433°C
```

MAE = Mean Absolute Error in °C. Lower is better.
`0.2036°C` means the model is wrong by ~0.2 degrees on average.

---

# Part 2 — Product Requirements Document (PRD)

---

## 1. Project Goal

Build a production-grade ML system that:

- Collects and stores historical weather data (Open-Meteo archive API)
- Trains multiple ML models to predict hourly temperature for Gurgaon, India
- Serves predictions via FastAPI with webhook push support
- Stores predictions in PostgreSQL and evaluates accuracy over time
- Retrains weekly to improve accuracy (pseudo-reinforcement loop)
- Deploys on Render free tier

---

## 2. System Architecture

```
Open-Meteo Archive API
        ↓
data/bootstrap.py  →  CSV files in data/
        ↓
ml/train_all.py    →  model_*.pkl files in ml/
        ↓
ml/predict.py      →  weather_predictions (DB)
        ↓
ml/evaluate.py     →  model_metrics (DB)
        ↓
FastAPI + Webhook  →  n8n / external consumers
```

### Data Flow

```
bootstrap.py  → downloads CSVs only (no DB write — training data stays in CSV)
train_all.py  → merges all CSVs → engineers 24 features → trains 7 models → saves .pkl
predict.py    → loads model.pkl → engineers features from last 30 CSV rows → predicts → saves to DB
evaluate.py   → JOINs weather_predictions vs weather_raw → computes MAE/RMSE → saves to DB
```

Training data lives in CSV files only. The database stores predictions and metrics only.

---

## 3. Project Folder Structure

```
weather/
│
├── app/
│   ├── main.py          # FastAPI app + hourly webhook background scheduler
│   ├── predict.py       # /predict/*, /train, /evaluate endpoints
│   └── webhook.py       # /webhook/register, /send, /list, /unregister
│
├── ml/
│   ├── train.py         # Quick single-model train (RandomForest, 100 trees)
│   ├── train_all.py     # Full 7-model training with epoch logging (100 epochs, 500 trees)
│   ├── preprocess.py    # Feature engineering — 24 features (lag, rolling, derived, time)
│   ├── predict.py       # Load model → engineer features → predict → store in DB
│   ├── evaluate.py      # JOIN predictions vs actuals → MAE/RMSE → store in DB
│   ├── model.pkl        # Best model (auto-updated after train_all.py)
│   └── model_*.pkl      # Per-algorithm model snapshots
│
├── data/
│   ├── bootstrap.py     # Download Open-Meteo archive → year-by-year CSV files
│   ├── collect.py       # Live hourly fetch from OpenWeather API → DB
│   └── weather_*.csv    # Downloaded historical data (2020–present)
│
├── db/
│   └── postgres.py      # Connection pool + all insert/query helpers
│
├── .env                 # Environment variables (never commit)
├── .env.example         # Template for .env
├── requirements.txt
└── README.md
```

---

## 4. Data Source — Open-Meteo Archive API

### URL Structure

```
https://archive-api.open-meteo.com/v1/archive
  ?latitude=28.4595
  &longitude=77.0266
  &start_date=YYYY-MM-DD
  &end_date=YYYY-MM-DD
  &hourly=temperature_2m,apparent_temperature,relative_humidity_2m,
          dew_point_2m,pressure_msl,cloudcover,visibility,windspeed_10m,
          winddirection_10m,windgusts_10m,precipitation,rain,weathercode
```

Location is fixed to Gurgaon, India (lat 28.4595, lon 77.0266).

### Columns Returned by the API

| API Column | Saved As | Description | Unit |
|---|---|---|---|
| `time` | `recorded_at` | Hourly timestamp | datetime |
| `temperature_2m` | `temperature` | Air temperature at 2m height | °C |
| `apparent_temperature` | `feels_like` | Feels-like / wind chill | °C |
| `relative_humidity_2m` | `humidity` | Relative humidity at 2m | % |
| `dew_point_2m` | `dew_point` | Dew point temperature | °C |
| `pressure_msl` | `pressure` | Sea-level atmospheric pressure | hPa |
| `cloudcover` | `cloudcover` | Total cloud cover | % |
| `visibility` | `visibility` | Horizontal visibility | meters |
| `windspeed_10m` | `wind_speed` | Wind speed at 10m | km/h |
| `winddirection_10m` | `wind_direction` | Wind direction at 10m | degrees |
| `windgusts_10m` | `wind_gusts` | Peak wind gust speed | km/h |
| `precipitation` | `precipitation` | Total precipitation (rain + snow) | mm |
| `rain` | `rain` | Rain-only component | mm |
| `weathercode` | `weather_main` | WMO weather interpretation code | int |

> Note: The Open-Meteo hourly archive API does NOT provide `temp_min` / `temp_max` as separate columns.
> In the hourly archive, `temperature_2m` is the actual temperature at that hour.
> Daily min/max are derived features computed in `ml/preprocess.py` using `cummin()` / `cummax()` within each calendar day.
> (The `temp_min`/`temp_max` fields in OpenWeatherMap's current weather API are city-level deviations, not daily extremes — they are not used here.)

### weathercode Reference

| Code | Meaning |
|---|---|
| 0 | Clear sky |
| 1–3 | Mainly clear / partly cloudy / overcast |
| 45, 48 | Fog |
| 51–67 | Drizzle / rain (light to heavy) |
| 71–77 | Snow |
| 80–82 | Rain showers |
| 95+ | Thunderstorm |

### Bootstrap Logic

`data/bootstrap.py` downloads in yearly chunks and skips files that already exist:

| Scenario | Behaviour |
|---|---|
| `data/` folder empty | Downloads from `HISTORY_START` (2020-01-01) to `HISTORY_END` |
| CSV exists but behind | Skips existing files, downloads only missing years |
| CSV is current | Skips — nothing to do |

```python
HISTORY_START = date(2020, 1, 1)   # NEVER change — fixed baseline
HISTORY_END   = date(2026, 3, 13)  # extend forward as needed
```

---

## 5. Feature Engineering — 24 Features

All features are computed in `ml/preprocess.py`. The model is trained on these 24 columns.

### Time Features (5)

| Feature | How computed | Why useful |
|---|---|---|
| `hour` | `dt.hour` | Temperature follows a strong daily cycle |
| `day_of_week` | `dt.dayofweek` | Weekly patterns (urban heat, traffic) |
| `month` | `dt.month` | Seasonal variation |
| `season` | `month % 12 // 3` | 0=winter, 1=spring, 2=summer, 3=autumn |
| `is_daytime` | `hour.between(6, 18)` | Day/night temperature split |

### Raw Weather Features (11)

`humidity`, `dew_point`, `pressure`, `cloudcover`, `wind_speed`, `wind_direction`, `wind_gusts`, `feels_like`, `precipitation`, `rain`, `weather_main`

### Derived Features (3)

| Feature | Formula | Why useful |
|---|---|---|
| `humidity_pressure_ratio` | `humidity / pressure` | Storm / instability indicator |
| `daily_temp_max` | `cummax()` within each calendar day | Daily range context |
| `daily_temp_min` | `cummin()` within each calendar day | Daily range context |

> `daily_temp_max` and `daily_temp_min` are computed from the hourly data using expanding window within each day — not from a separate API field.

### Lag Features (3) — most powerful for time-series

| Feature | Shift | Why useful |
|---|---|---|
| `temp_lag_1h` | `temperature.shift(1)` | Strongest predictor — temperature changes gradually |
| `temp_lag_3h` | `temperature.shift(3)` | Short-term trend direction |
| `temp_lag_24h` | `temperature.shift(24)` | Same hour yesterday — captures daily pattern |

### Rolling Statistics (2)

| Feature | Window | Why useful |
|---|---|---|
| `temp_rolling_mean_6h` | 6-hour rolling mean | Smoothed recent trend |
| `temp_rolling_std_6h` | 6-hour rolling std dev | Volatility / instability indicator |

### Target Variable

```
y = temperature (°C) at the next hour
```

---

## 6. ML Models — Training Strategy

### Algorithms Trained (7)

| Model | Type | Epoch concept |
|---|---|---|
| GradientBoosting | Ensemble regression | Simulated via `warm_start` + incremental `n_estimators` |
| RandomForest | Ensemble regression | Simulated via `warm_start` + incremental `n_estimators` |
| XGBoost | Gradient boosting | Real per-round logging via `TrainingCallback` |
| DecisionTree | Single tree | Single pass — no epochs |
| LinearRegression | Linear | Single pass — no epochs |
| Ridge | Regularized linear | Single pass — no epochs |
| KNN | Instance-based | Single pass — no epochs |

### Epochs vs n_estimators — Important Distinction

This project uses tree-based models, not neural networks. There are no traditional epochs.

| Concept | Neural Network | Tree Models (this project) |
|---|---|---|
| Training unit | epoch (full data pass) | one tree added per step |
| Setting | `epochs=100` | `n_estimators=500` |
| Equivalent | 100 full passes | 500 trees built |
| Overfitting control | dropout, early stopping | `max_depth`, `max_samples` |

In `ml/train_all.py`, `TREES_TOTAL=500` is split across `EPOCHS=100` iterations (5 trees per epoch). This gives you epoch-style terminal output showing MAE improving as more trees are added — it's the same idea as epochs, just implemented differently for tree models.

```python
EPOCHS      = 100   # how many progress checkpoints to print
TREES_TOTAL = 500   # total trees built (= training strength)
```

To increase training strength, raise `TREES_TOTAL`:

```python
TREES_TOTAL = 1000  # stronger model, takes longer
```

If you ever switch to a neural network (LSTM, etc.):

```python
model.fit(X_train, y_train, epochs=100)  # real epochs — only for neural networks
```

### Train/Test Split

Always chronological — never shuffle time-series data:

```python
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]   # 80% train, 20% test
```

### RAM Optimisation

The training pipeline is designed to avoid RAM crashes on low-memory machines:

| Technique | What it does |
|---|---|
| `float32` arrays | Halves data memory vs default `float64` |
| One model at a time | Each model is trained, saved, then `del`-ed before the next |
| `gc.collect()` per epoch | Forces Python to release memory after each epoch |
| `max_samples=0.8` | Each tree sees 80% of rows — less RAM per tree |
| `max_features=0.8` | Each split uses 80% of features |
| `tree_method="hist"` | XGBoost histogram binning — much lower RAM than exact method |
| `n_jobs=-1` | All CPU cores for speed (set to `1` only if RAM is critically low) |

---

## 7. Database Schema

Only predictions and metrics go to the database. Training data stays in CSV files.

### weather_predictions

```sql
CREATE TABLE weather_predictions (
    id SERIAL PRIMARY KEY,
    prediction_for TIMESTAMP,
    predicted_temp FLOAT,
    model_version TEXT
);
```

### model_metrics

```sql
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    evaluated_at TIMESTAMP,
    mae FLOAT,
    rmse FLOAT,
    model_version TEXT
);
```

### weather_raw (for live data collection via `data/collect.py`)

```sql
CREATE TABLE weather_raw (
    id SERIAL PRIMARY KEY,
    recorded_at TIMESTAMP UNIQUE,
    temperature FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    wind_speed FLOAT,
    weather_main TEXT
);
```

### db/postgres.py helpers

```python
from db.postgres import (
    init_db,
    insert_prediction,
    insert_metrics,
    insert_weather,
    insert_weather_bulk,
    fetch_prediction_vs_actual,
    get_latest_recorded_at,
)
```

Connection pooling via `psycopg2.pool.SimpleConnectionPool` (1–5 connections).
`.env` is loaded using absolute path relative to `__file__` so it works from any working directory.

---

## 8. Evaluation — How Accuracy is Measured

`ml/evaluate.py` JOINs stored predictions with actual weather:

```sql
SELECT
    p.prediction_for,
    p.predicted_temp,
    r.temperature AS actual_temp
FROM weather_predictions p
JOIN weather_raw r ON p.prediction_for = r.recorded_at;
```

Then computes:

```python
errors = predicted - actual
mae  = mean(abs(errors))      # average error in °C
rmse = sqrt(mean(errors²))    # penalises large errors more
```

Both stored in `model_metrics` with `model_version` so you can compare across retraining runs:

```sql
SELECT model_version, AVG(mae) FROM model_metrics GROUP BY model_version;
```

---

## 9. Webhook System

Predictions are automatically pushed to registered URLs every hour via a FastAPI background scheduler.

### Register the n8n webhook

```
POST /webhook/register
Body: { "url": "https://n8n-29o4.onrender.com/webhook/weather" }
```

Or set `WEBHOOK_URL` in `.env` to pre-register on startup.

### Payload sent to webhook

```json
{
  "predicted_temperature": 18.47,
  "hours_ahead": 1,
  "prediction_for": "2026-03-20T21:00:00Z",
  "sent_at": "2026-03-20T20:00:00Z"
}
```

### Manual trigger

```
POST /webhook/send?hours_ahead=1
```

### Background scheduler

`app/main.py` runs an `asyncio` background task that fires `dispatch_to_webhooks()` every 3600 seconds. It starts automatically when the FastAPI server starts.

---

## 10. Deployment — Render Free Tier

### Services

1. PostgreSQL (already running at Render)
2. FastAPI Web Service

### Environment Variables on Render

```
DATABASE_URL        = Render internal PostgreSQL URL
OPENWEATHER_API_KEY = your key
MODEL_VERSION       = v1
WEBHOOK_URL         = https://n8n-29o4.onrender.com/webhook/weather
```

### Constraints and Workarounds

| Constraint | Workaround |
|---|---|
| Free tier sleeps after 15 min inactivity | Cold start ~30–60s — acceptable for this use case |
| Limited RAM on Render | Train locally, upload `model.pkl` via Git |
| `n_estimators` limit on Render | Keep ≤ 200 trees when running on Render |
| DB latency (Render free tier) | Only predictions/metrics go to DB, not training data |

---

## 11. Model Versioning

Every training run saves a versioned snapshot:

```
model_v1.pkl   ← first training
model_v2.pkl   ← after more data
model_v3.pkl   ← after feature improvements
model.pkl      ← always the current best
```

`model_version` column in `weather_predictions` and `model_metrics` tracks which model made each prediction:

```sql
SELECT model_version, AVG(mae) FROM model_metrics GROUP BY model_version ORDER BY AVG(mae);
```

---

## 12. Weekly Retraining Cycle

```
Every Sunday
      ↓
data/bootstrap.py   → fill any data gaps in CSVs
      ↓
ml/train_all.py     → retrain all 7 models, pick best → model.pkl updated
      ↓
ml/predict.py       → generate and store next prediction
      ↓
ml/evaluate.py      → compare vs actuals, log MAE/RMSE
      ↓
If new model is better → model.pkl updated automatically
```

---

## 13. Development Roadmap

### Phase 1 — Data Pipeline
- [x] `data/bootstrap.py` — chunked yearly download, skip existing, gap fill
- [x] `data/collect.py` — live hourly OpenWeather fetch → DB
- [x] `db/postgres.py` — connection pool, bulk insert, query helpers

### Phase 2 — ML Model
- [x] `ml/preprocess.py` — 24 features (lag, rolling, derived, time)
- [x] `ml/train.py` — quick single-model train (RandomForest, 100 trees)
- [x] `ml/train_all.py` — 7 models, 100 epochs, 500 trees, RAM-safe, epoch logging
- [x] `ml/predict.py` — load model, engineer features from last 30 rows, predict, store

### Phase 3 — Evaluation
- [x] `ml/evaluate.py` — JOIN predictions vs actuals, MAE/RMSE, store metrics

### Phase 4 — API
- [x] `GET /predict/next-hour`, `/predict/hours`, `/predict/today`
- [x] `POST /train`, `/evaluate`
- [x] Webhook system with register/unregister/send/list
- [x] Hourly background scheduler
- [ ] Deploy to Render

### Phase 5 — Automation
- [ ] Daily cron: `data/collect.py`
- [ ] Weekly cron (Sunday): full retrain pipeline

---

## 14. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| RAM crash during training | `float32` arrays, one model at a time, `gc.collect()` per epoch |
| Render free tier RAM limit | Train locally, upload `model.pkl` via Git |
| Open-Meteo rate limits | Chunked yearly downloads, skip existing files |
| Model drift (seasonal) | Weekly retraining with fresh data |
| DB latency (Render free tier) | Only predictions/metrics go to DB, not training data |
| `prediction_for` JOIN mismatch | Truncate to hour on both sides before insert |
| Windows RAM pressure during training | `n_jobs=-1` for speed; fall back to `n_jobs=1` only if crashing |

---

## 15. Definition of Done

The project is complete when:

- Predictions are generated and stored automatically every hour
- MAE tracked per model version over time
- Weekly retraining runs without manual intervention
- API is publicly accessible on Render
- Webhook pushes next-hour prediction to n8n every hour
