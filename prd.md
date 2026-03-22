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

`data/bootstrap.py` downloads in yearly chunks. `HISTORY_END` is always computed at runtime as yesterday — no manual edits ever needed.

```python
HISTORY_START = date(2020, 1, 1)                    # NEVER change — fixed baseline
HISTORY_END   = date.today() - timedelta(days=1)    # always yesterday, auto-updates
```

The download range is split into yearly chunks:

```
2020-01-01 → 2020-12-31   (full year)
2021-01-01 → 2021-12-31   (full year)
...
2026-01-01 → 2026-03-21   (partial — current year, end = yesterday)
```

---

### File Naming Convention

Each chunk is saved as:

```
data/weather_{chunk_start}_{chunk_end}.csv
```

Examples:
```
data/weather_2020-01-01_2020-12-31.csv   ← full year, never changes
data/weather_2021-01-01_2021-12-31.csv   ← full year, never changes
data/weather_2026-01-01_2026-03-21.csv   ← partial year, end date moves daily
```

The current year's file has a moving end date in its filename. Every day you run bootstrap, yesterday becomes the new end date, so the filename changes.

---

### Skip vs Re-download Decision Logic

For each chunk, `download_year(start, end)` runs this logic:

```
Step 1 — Build expected filename:
    csv_path = f"data/weather_{start}_{end}.csv"

Step 2 — Check for stale file (same start, different end date):
    stale = any file matching "weather_{start}_*.csv" that is NOT csv_path
    → If found: DELETE the stale file, then proceed to download

Step 3 — Check if current file already exists:
    → If csv_path exists: SKIP (already up to date)
    → If not: DOWNLOAD from Open-Meteo API
```

---

### Scenario Walkthrough

**Scenario A — First run ever (empty `data/` folder)**
```
No files exist → all chunks download from scratch
2020-01-01_2020-12-31.csv  ← downloaded
2021-01-01_2021-12-31.csv  ← downloaded
...
2026-01-01_2026-03-21.csv  ← downloaded (partial year, ends yesterday)
```

**Scenario B — Re-run same day**
```
All files exist with correct end dates → all skipped
Already exists, skipping: weather_2020-01-01_2020-12-31.csv
Already exists, skipping: weather_2026-01-01_2026-03-21.csv
```

**Scenario C — Re-run next day (most important case)**
```
Full year files (2020–2025) → unchanged filenames → skipped
Current year file:
  Stale:    weather_2026-01-01_2026-03-21.csv  ← DELETED
  Expected: weather_2026-01-01_2026-03-22.csv  ← DOWNLOADED (new end date)
```

**Scenario D — Re-run after a week away**
```
Full year files → skipped
Current year file:
  Stale:    weather_2026-01-01_2026-03-21.csv  ← DELETED
  Expected: weather_2026-01-01_2026-03-28.csv  ← DOWNLOADED (catches up all missing days)
```

**Scenario E — New year rolls over (Jan 1)**
```
2025-01-01_2025-12-31.csv  ← now a complete year, never changes again
2026-01-01_2026-01-01.csv  ← new partial year file created (just 1 day)
```

---

### Why "yesterday" and not "today"

Open-Meteo archive API has a ~1 day lag. Today's data is not yet available or is incomplete. Using `date.today() - timedelta(days=1)` ensures every request returns complete hourly data for all 24 hours of the requested end date.

Requesting today would return partial data (only hours up to the current UTC time), which would create NaN rows in the CSV and corrupt lag features during training.

---

### Summary Table

| Scenario | Full year files (2020–2025) | Current year file |
|---|---|---|
| First run | Downloaded | Downloaded |
| Same day re-run | Skipped | Skipped |
| Next day re-run | Skipped | Stale deleted → re-downloaded |
| After N days away | Skipped | Stale deleted → re-downloaded (catches up) |
| New year (Jan 1) | Previous year now complete, skipped forever | New partial file created |

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

---

### Single-Pass Models — Why No Epochs, Are They Working?

The four models that print `single pass (no epochs)` are not broken or undertrained. This is exactly how they are supposed to work.

```
[decision_tree]      MAE: 0.2504°C   single pass
[linear_regression]  MAE: 0.3381°C   single pass
[ridge]              MAE: 0.2856°C   single pass
[knn]                MAE: 0.6482°C   single pass
```

**Why single pass?**

These are not iterative learners. They have no concept of "train a little, then train more." Each one solves its problem in a single mathematical operation over the full dataset:

| Model | What happens in one pass |
|---|---|
| DecisionTree | Recursively splits the data by the best feature threshold until `max_depth=12` is reached. Done. |
| LinearRegression | Solves a system of equations (ordinary least squares) — one matrix operation. Done. |
| Ridge | Same as LinearRegression but adds an L2 penalty term to prevent overfitting. One solve. Done. |
| KNN | Does nothing during training — just stores the data. Prediction time: finds the 5 nearest neighbours. |

Running them for multiple epochs would not improve them — it would just repeat the same computation and return the same result every time.

**Are they doing their job?**

Yes. Their MAE results are real and meaningful:

| Model | MAE | Verdict |
|---|---|---|
| decision_tree | 0.2504°C | Decent — single tree captures non-linear patterns but overfits more than ensembles |
| linear_regression | 0.3381°C | Expected — temperature is non-linear, linear models hit a ceiling |
| ridge | 0.2856°C | Slightly better than linear — L2 regularisation reduces variance |
| knn | 0.6482°C | Worst — KNN struggles with 24 features (curse of dimensionality) and 227k rows |

None of these will win against RandomForest or XGBoost on this dataset. That is expected and fine. They exist in the pipeline for two reasons:

1. **Baseline comparison** — they prove the ensemble models are actually earning their complexity. If RandomForest only beat LinearRegression by 0.01°C, it wouldn't be worth the RAM and training time.
2. **Automatic best-model selection** — `train_all_fixed.py` picks the lowest MAE across all 7 models. If for some reason the ensemble models overfit badly on a future dataset, a simpler model could legitimately win.

**Why are they always CPU?**

sklearn does not support GPU acceleration. This is a library design decision — sklearn is optimised for CPU parallelism via `n_jobs=-1`. There is no CUDA path, no workaround, and no version of sklearn that changes this. GPU support for these model types would require switching to cuML (NVIDIA RAPIDS), which has significant installation complexity and is not worth it for models that are already fast on CPU at this data size.

**Should you remove them?**

No. They train in under 1 second each, cost almost no RAM, and give you a meaningful baseline. Removing them would only save ~3 seconds of total training time while losing the comparison value.

---

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


---

## Problems

Known challenges, limitations, and technical debt in this project. Grouped by category with root cause, current impact, and recommended fix.

---

### 1. Data Quality

**`weathercode` treated as a continuous number**

- Root cause: `weathercode` is a WMO categorical code (0 = clear, 95 = thunderstorm). It's stored and fed to the model as a raw integer. The model interprets 95 as "greater than" 3, implying thunderstorm is just a "bigger" version of cloudy — which is mathematically wrong.
- Impact: Tree models partially work around this by splitting on thresholds, but linear models and KNN are actively misled. The feature contributes noise instead of signal.
- Fix: One-hot encode `weathercode` into buckets in `ml/preprocess.py`:
  ```python
  bins = [0, 3, 45, 67, 77, 82, 99]
  labels = ["clear", "cloudy", "fog", "rain", "snow", "showers", "storm"]
  df["weather_bucket"] = pd.cut(df["weather_main"], bins=bins, labels=labels)
  df = pd.get_dummies(df, columns=["weather_bucket"])
  ```
- Status: not fixed — `weather_main` is currently passed as raw int

---

**Lag features break at CSV file boundaries**

- Root cause: `temp_lag_1h = temperature.shift(1)` is computed per-file when files are loaded individually. The last row of `weather_2023.csv` and the first row of `weather_2024.csv` are 1 hour apart in real time, but `shift(1)` across the concatenated DataFrame doesn't know this — it works correctly only because `load_raw_data()` concatenates all CSVs before engineering. However, if a file is missing or has a gap, the shift silently produces a NaN which gets dropped.
- Impact: A few rows lost per year boundary — negligible for training accuracy, but worth knowing when debugging row count mismatches.
- Fix: After concat and sort, validate that `recorded_at` has no gaps larger than 1 hour before computing lags. Log a warning if gaps are found.
- Status: not fixed — gaps are silently dropped via `dropna`

---

**Missing or null hours in archive data**

- Root cause: Open-Meteo archive occasionally has null values for specific hours, especially `precipitation` and `visibility`. The current `dropna(subset=["temperature", "humidity", "pressure"])` only drops rows missing the three core columns — other nulls propagate into features.
- Impact: Null values in `wind_gusts` or `dew_point` become `NaN` in derived features, which some models handle differently (tree models ignore, linear models fail silently or produce NaN predictions).
- Fix: Add forward-fill for weather columns before feature engineering:
  ```python
  df[numeric_cols] = df[numeric_cols].ffill().bfill()
  ```
- Status: partial — only critical columns are checked

---

### 2. Model Drift

**Extreme seasonality in Gurgaon**

- Root cause: Gurgaon has one of the widest temperature ranges in India — ~5°C in January, ~45°C in May/June. A model trained on a full year handles this, but if weekly retraining is skipped for several weeks during a seasonal transition, the model's recent-data weighting drifts.
- Impact: Predictions can be 3–5°C off during rapid seasonal transitions (Feb→Mar, Oct→Nov).
- Fix: Weight recent data more heavily during training using `sample_weight`:
  ```python
  # Give last 90 days 3x weight
  weights = np.where(df["recorded_at"] > cutoff_90d, 3.0, 1.0)
  model.fit(X_train, y_train, sample_weight=weights)
  ```
- Status: not implemented — all rows weighted equally

---

**Year-over-year climate shift**

- Root cause: 2024 and 2025 summers in Gurgaon were measurably hotter than 2020–2022. A model trained equally on all years may underpredict peak summer temperatures.
- Impact: Systematic underprediction during heatwaves — the model has never seen those temperatures in training.
- Fix: Track mean error (bias) per month in `model_metrics`. If June bias > +2°C, trigger retraining with higher weight on recent summers.
- Status: not implemented — only MAE/RMSE tracked, no bias per season

---

### 3. Prediction Quality

**Multi-hour forecast uses stale lag features**

- Root cause: `/predict/hours?hours=6` calls `get_latest_features()` once and reuses the same feature vector for all 6 hours. The lag features (`temp_lag_1h`, `temp_lag_3h`, `temp_rolling_mean_6h`) are computed from the last real CSV row — they don't update as the forecast rolls forward.
- Impact: Hour 1 prediction is accurate. Hours 2–6 are essentially the same prediction repeated with only the `hour` field changing. The further ahead, the less meaningful the result.
- Fix: Implement autoregressive forecasting — feed each prediction back as the next hour's lag:
  ```python
  predicted_temps = []
  lag_window = list(df["temperature"].tail(24))  # seed with real data

  for h in range(1, hours + 1):
      features = build_features_from_lag(lag_window, hour=(now + timedelta(hours=h)).hour)
      temp = model.predict(features)[0]
      predicted_temps.append(temp)
      lag_window.append(temp)   # use prediction as next lag input
      lag_window.pop(0)
  ```
- Status: not fixed — all forecast hours use identical lag features

---

**No prediction confidence interval**

- Root cause: All models return a single point estimate. There's no measure of uncertainty.
- Impact: A prediction of "28.4°C" with no range is misleading — the model could be ±0.2°C or ±4°C depending on conditions, and the API consumer has no way to know.
- Fix: Use `RandomForest` individual tree predictions to estimate spread:
  ```python
  tree_preds = np.array([t.predict(X) for t in model.estimators_])
  mean = tree_preds.mean(axis=0)
  std  = tree_preds.std(axis=0)
  # Return: predicted=28.4, confidence_low=27.8, confidence_high=29.0
  ```
- Status: not implemented

---

### 4. Evaluation Gap

**`ml/evaluate.py` returns zero rows if `data/collect.py` isn't running**

- Root cause: Evaluation JOINs `weather_predictions` (predictions stored by the API) with `weather_raw` (actuals stored by `data/collect.py`). If the live collection cron job isn't running, `weather_raw` stays empty and the JOIN returns nothing.
- Impact: You can generate predictions all day but have no way to measure their accuracy unless the live collector is also running. On Render free tier, the service sleeps — so the cron never fires.
- Fix: Either (a) run `data/collect.py` as a separate Render cron job, or (b) backfill actuals from Open-Meteo archive after the fact:
  ```python
  # Fetch yesterday's actuals and insert into weather_raw for evaluation
  .\venv\Scripts\python.exe data/bootstrap.py  # already handles gap fill
  ```
- Status: evaluation is non-functional unless collect.py is running continuously

---

**No bias tracking — only absolute error**

- Root cause: `model_metrics` stores MAE and RMSE only. These are symmetric — a model that's always 2°C too high looks the same as one that's randomly ±2°C.
- Impact: Systematic seasonal bias goes undetected. You might retrain and get the same MAE but the model is still always wrong in the same direction.
- Fix: Add `mean_error` (signed) to `model_metrics`:
  ```python
  mean_error = float(np.mean(predicted - actual))  # positive = model runs hot
  ```
  And track per-month breakdown to catch seasonal drift early.
- Status: not implemented

---

### 5. Infrastructure

**Render free tier sleeps — hourly webhook scheduler dies**

- Root cause: Render free tier web services sleep after 15 minutes of inactivity. The `asyncio` background task in `app/main.py` that fires webhooks every hour is killed when the process sleeps.
- Impact: Webhook pushes are unreliable. The n8n workflow won't receive data during sleep periods, which could be most of the day if the API isn't being actively called.
- Fix options:
  - Use an external cron service (cron-job.org, GitHub Actions scheduled workflow) to ping `/webhook/send` every hour — this also keeps the service awake
  - Upgrade to Render paid tier (no sleep)
  - Move webhook dispatch to a separate Render cron job service
- Status: known limitation — no fix implemented

---

**`model.pkl` committed to Git — binary blob history grows**

- Root cause: `model.pkl` and all `model_*.pkl` files are tracked in Git. Each weekly retrain produces a new binary version. Git stores the full file each time (binary files don't diff).
- Impact: After 6 months of weekly retraining, the repo history could contain 200+ MB of `.pkl` blobs. `git clone` becomes slow.
- Fix: Add to `.gitignore`:
  ```
  ml/*.pkl
  ```
  Then store model files on Render Disk, S3, or use DVC (Data Version Control) for proper ML artifact versioning.
- Status: `.pkl` files are currently tracked in Git

---

**Webhook URL list is in-memory — lost on every restart**

- Root cause: `_webhook_urls` in `app/webhook.py` is a Python list. Any URL registered via `POST /webhook/register` is lost when the process restarts or Render redeploys.
- Impact: Only the `WEBHOOK_URL` from `.env` survives restarts. Any dynamically registered URLs (e.g. from a frontend or n8n workflow) are silently dropped.
- Fix: Persist registered URLs in the database:
  ```sql
  CREATE TABLE webhook_subscriptions (
      id SERIAL PRIMARY KEY,
      url TEXT UNIQUE NOT NULL,
      created_at TIMESTAMP DEFAULT NOW()
  );
  ```
  Load on startup, write on register, delete on unregister.
- Status: in-memory only — not persisted

---

### Priority Summary

| Problem | Impact | Effort | Fix Now? |
|---|---|---|---|
| `weathercode` one-hot encoding | Medium — misleads linear models | Low | Yes |
| Multi-hour forecast stale lags | High — hours 2–24 are unreliable | Medium | Yes |
| Evaluation needs collect.py running | High — can't measure accuracy | Low | Yes |
| Webhook URLs lost on restart | Medium — n8n URL survives via .env | Low | Yes |
| Render sleep kills webhook scheduler | Medium — missed hourly pushes | Low | Yes (external cron) |
| No confidence interval | Low — nice to have | Medium | Later |
| Bias tracking per season | Low — diagnostic improvement | Low | Later |
| Sample weighting for recent data | Medium — better seasonal accuracy | Medium | Later |
| `.pkl` files in Git | Low — repo size issue over time | Low | Later |
