# Weather Prediction ML Project – Architecture & Progress Log

## 1. Project Goal

Build a machine learning system that:

* Collects historical weather data
* Stores it in PostgreSQL (hosted on Render free tier)
* Trains a model to predict future weather (temperature, humidity, rainfall)
* Compares predictions with real data from OpenWeather API
* Retrains weekly to improve accuracy (pseudo‑reinforcement loop)
* Deploys prediction API using FastAPI on Render free tier

---

## 2. Current Infrastructure Status

### Backend & Storage

* **PostgreSQL**: Hosted on Render (Free Tier)
* Database connected successfully from local environment
* Tables:

  * `weather_raw` — ground truth actual data (UNIQUE on `recorded_at`)
  * `weather_predictions` — model guesses
  * `model_metrics` — error tracking per model version

### Data Source

* **Primary Source**: OpenWeather API
* Data fields collected:

  * datetime
  * temperature
  * humidity
  * pressure
  * wind speed
  * weather condition

---

## 3. System Architecture Overview

```
OpenWeather API
      ↓
data/collect.py  →  weather_raw (actual data)
                          ↓
                    ml/train.py  →  model_vN.pkl
                          ↓
                    ml/predict.py  →  weather_predictions
                          ↓
                    ml/evaluate.py  →  model_metrics
                          ↓
              FastAPI + Webhook push (next-hour prediction)
```

### Data Flow (Simple View)

```
OpenWeather API
      ↓
weather_raw  (actual data)
      ↓
Train Model
      ↓
weather_predictions (future predictions)
      ↓
Later fetch actual again
      ↓
model_metrics (error calculation)
```

This gives you:
- training data
- prediction history
- accuracy tracking

---

## 4. Database Responsibilities

| Table                | Purpose                        |
|----------------------|--------------------------------|
| `weather_raw`        | Ground truth data              |
| `weather_predictions`| What the model guessed         |
| `model_metrics`      | How wrong the model was        |

This separation is important because:
- training uses only actual data
- evaluation compares predicted vs actual

---

## 5. Dataset Reference (Open-Meteo Columns)

| Column | Description | Unit | ML Use |
|---|---|---|---|
| `time` | Hourly timestamp | datetime | Extract hour, day, month, season |
| `temperature_2m` | Air temp at 2m height | °C | Primary prediction target |
| `apparent_temperature` | Feels-like temp | °C | Comfort/heat index feature |
| `relative_humidity_2m` | Moisture in air | % (0–100) | Rain/fog prediction feature |
| `dew_point_2m` | Condensation threshold | °C | Fog detection feature |
| `pressure_msl` | Sea-level air pressure | hPa | Storm detection (low = bad weather) |
| `cloudcover` | Sky cloud coverage | % (0–100) | Fog/solar prediction feature |
| `visibility` | Visible distance | meters | Often missing — drop or fill |
| `windspeed_10m` | Wind speed at 10m | km/h | Storm/wind energy feature |
| `winddirection_10m` | Wind direction | degrees (0–360) | Storm prediction feature |
| `windgusts_10m` | Peak gust speed | km/h | Storm prediction feature |
| `precipitation` | Total precipitation | mm | Flood forecasting target |
| `rain` | Rain-only component | mm | Rain prediction target |
| `weathercode` | Encoded weather type | int | Classification target |

### weathercode reference

| Code | Meaning |
|---|---|
| 0 | Clear sky |
| 1–3 | Partly/mostly cloudy |
| 51–67 | Rain (drizzle to heavy) |
| 71–77 | Snow |
| 95+ | Thunderstorm |

---

## 6. Machine Learning Strategy

### Model Type

Supervised learning — we have labeled historical data. Reinforcement learning is not suitable for time-series prediction directly.

### Prediction Targets

| Goal | Target column |
|---|---|
| Predict temperature | `temperature_2m` |
| Predict rain | `precipitation` or `rain` |
| Classify weather type | `weathercode` |

### Feature Inputs (X)

```
relative_humidity_2m, dew_point_2m, pressure_msl,
cloudcover, windspeed_10m, winddirection_10m, windgusts_10m
```

Plus time-engineered features extracted from `time`:

```python
df["hour"]        = pd.to_datetime(df["time"]).dt.hour
df["day_of_week"] = pd.to_datetime(df["time"]).dt.dayofweek
df["month"]       = pd.to_datetime(df["time"]).dt.month
```

### Data Cleaning Notes

- `visibility` is often missing — drop or fill with median
- All numeric columns coerced with `pd.to_numeric(errors="coerce")`, NaN rows dropped
- No random shuffle — use chronological train/test split:

```python
train = df[df["time"] < "2024-01-01"]
test  = df[df["time"] >= "2024-01-01"]
```

### Algorithms

| Type | Algorithm |
|---|---|
| Regression | Linear Regression (baseline), Random Forest, XGBoost |
| Classification | Random Forest, Logistic Regression, Neural Network |
| Time-series (advanced) | ARIMA, LSTM, Prophet |

### Learned Patterns (intuition)

- High humidity + low pressure → rain
- High cloudcover → overcast/rain
- Pressure drop + strong gusts → storm approaching

---

### ML Pipeline Implementation Guide

Three separate pipelines are built and compared:

1. Regression — predict temperature or precipitation (numeric output)
2. Classification — predict weathercode (category output)
3. Time-series — forecast future values sequentially

#### Common Setup

```python
df["time"] = pd.to_datetime(df["time"])
df["hour"]  = df["time"].dt.hour
df["day"]   = df["time"].dt.day
df["month"] = df["time"].dt.month
df = df.drop(columns=["visibility"]).dropna()
```

Chronological split (never shuffle time-series data):

```python
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
```

#### Regression (predict temperature)

```python
# Features & target
X = df.drop(columns=["temperature_2m", "time"])
y = df["temperature_2m"]

# Train all three
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

models = [LinearRegression(), RandomForestRegressor(), XGBRegressor()]
for m in models:
    m.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_absolute_error
for m in models:
    print(m.__class__.__name__, mean_absolute_error(y_test, m.predict(X_test)))
```

#### Classification (predict weathercode)

```python
X = df.drop(columns=["weathercode", "time"])
y = df["weathercode"]

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(),
    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
]
for m in models:
    m.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for m in models:
    print(m.__class__.__name__, accuracy_score(y_test, m.predict(X_test)))
```

Note: normalize features before MLPClassifier using `StandardScaler`.

#### Time-Series (advanced)

ARIMA — classical, good for temperature/pressure:

```python
from statsmodels.tsa.arima.model import ARIMA
model_fit = ARIMA(df["temperature_2m"], order=(5,1,0)).fit()
forecast = model_fit.forecast(steps=10)
```

Prophet — easy, handles seasonality well:

```python
from prophet import Prophet
prophet_df = df[["time", "temperature_2m"]].rename(columns={"time":"ds","temperature_2m":"y"})
m = Prophet()
m.fit(prophet_df)
forecast = m.predict(m.make_future_dataframe(periods=24))
```

LSTM — deep learning, best for complex patterns (requires sequence reshaping):

```python
X_lstm = np.array(X).reshape((X.shape[0], 1, X.shape[1]))
model = Sequential([LSTM(50, activation='relu', input_shape=(1, X.shape[1])), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X_lstm, y, epochs=10)
```

#### Model Comparison Metrics

| Pipeline | Metric |
|---|---|
| Regression | MAE, RMSE |
| Classification | Accuracy, Confusion Matrix |
| Time-series | MAE, RMSE on forecast |

#### Recommended Progression

Start simple → validate → improve:

```
LinearRegression / LogisticRegression
        ↓
  RandomForest
        ↓
   XGBoost
        ↓
 LSTM / Prophet
```

#### Pro Tips

- Add lag features to capture temporal patterns: `df["temp_lag1"] = df["temperature_2m"].shift(1)`
- Handle class imbalance in `weathercode` (rare storm codes vs common clear-sky codes)
- Normalize inputs for neural networks with `StandardScaler`

---

## 6. Project Folder Structure

```
weather-ml/
│
├── app/
│   ├── main.py                # FastAPI app + hourly webhook scheduler
│   ├── predict.py             # /predict, /train, /evaluate endpoints
│   └── webhook.py             # Webhook register/send endpoints
│
├── ml/
│   ├── train.py               # Model training script
│   ├── predict.py             # Generate + store next-hour prediction
│   ├── preprocess.py          # Data cleaning and feature engineering
│   ├── evaluate.py            # JOIN predictions vs actuals → model_metrics
│   ├── model.pkl              # Latest trained model (symlink/copy)
│   └── model_vN.pkl           # Versioned model snapshots
│
├── data/
│   ├── collect.py             # Fetch from OpenWeather → weather_raw (hourly)
│   └── bootstrap.py           # One-time bulk seed from Open-Meteo archive
│
├── db/
│   └── postgres.py            # Connection pool + insert/query helpers
│
├── requirements.txt
└── README.md
```

---

## 7. Database Schema

### Table: weather_raw

Stores actual weather fetched from API. `recorded_at` is UNIQUE to prevent duplicates.

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

### Table: weather_predictions

Stores model predictions. Joined with `weather_raw` on `prediction_for = recorded_at` for evaluation.

```sql
CREATE TABLE weather_predictions (
    id SERIAL PRIMARY KEY,
    prediction_for TIMESTAMP,
    predicted_temp FLOAT,
    model_version TEXT
);
```

### Table: model_metrics

Tracks accuracy over time per model version.

```sql
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    evaluated_at TIMESTAMP,
    mae FLOAT,
    rmse FLOAT,
    model_version TEXT
);
```

---

## 8. File-by-File Execution Chain

```
data/collect.py
      ↓
ml/train.py
      ↓
ml/predict.py
      ↓
ml/evaluate.py
```

### Step 1 – data/collect.py

Runs hourly or manually.

```
API → JSON → INSERT INTO weather_raw
```

Key rule: `recorded_at` must be UNIQUE — prevents duplicate rows when the script runs twice (`ON CONFLICT DO NOTHING`).

### Step 0 (Bootstrap) – data/bootstrap.py

Handles two scenarios automatically:

| Scenario | Behaviour |
|---|---|
| DB is empty | Full seed from `2020-01-01` to yesterday |
| DB is behind | Gap fill from `last_recorded_at + 1 day` to yesterday |
| DB is current | No-op |

Called automatically by `ml/train.py` before every training run, or manually:

```bash
python data/bootstrap.py
```

Dynamic URL logic:
- `HISTORY_START = 2020-01-01` — fixed, never changes
- `end_date` = yesterday (always `date.today() - 1`)
- `start_date` = `HISTORY_START` if DB empty, else `last_recorded_at.date() + 1 day`
- Lat/lon fixed to Gurgaon (28.4595, 77.0266)

Cleaning steps:
- Rename Open-Meteo fields to match `weather_raw` schema
- Parse `time` → `recorded_at` as datetime
- Coerce all numeric columns with `pd.to_numeric(errors="coerce")`
- Drop rows with any NaN

### Step 2 – ml/train.py

```
SELECT * FROM weather_raw
      ↓
feature engineering
      ↓
train regression model
      ↓
save model_vN.pkl + model.pkl
```

Example training dataset:

| time  | temp | humidity | pressure |
|-------|------|----------|----------|
| 10:00 | 30   | 55       | 1012     |
| 11:00 | 32   | 52       | 1011     |

### Step 3 – ml/predict.py

Loads `model.pkl`, fetches latest weather, predicts next-hour temperature, stores it:

```
INSERT INTO weather_predictions
```

Example row:

| prediction_for      | predicted_temp | model_version |
|---------------------|----------------|---------------|
| 2026-03-20 15:00    | 34.2           | v1            |

### Step 4 – ml/evaluate.py

Runs after actual data becomes available. JOINs predictions with actuals:

```sql
SELECT
    p.prediction_for,
    p.predicted_temp,
    r.temperature
FROM weather_predictions p
JOIN weather_raw r ON p.prediction_for = r.recorded_at;
```

Python computes `error = predicted - actual` and stores MAE & RMSE in `model_metrics`.

---

## 9. ML Training Workflow

### Feature Engineering

Convert datetime into model-friendly features:

* hour
* day_of_week
* month

### Model Features

```
X = [hour, day_of_week, month, relative_humidity_2m, dew_point_2m,
     pressure_msl, cloudcover, windspeed_10m, winddirection_10m, windgusts_10m]
y = temperature_2m  (or weathercode for classification)
```

### Save Model

```python
joblib.dump(model, "ml/model_v1.pkl")
joblib.dump(model, "ml/model.pkl")  # latest
```

---

## 10. Weekly Retraining Strategy

This simulates reinforcement learning by:

* Collecting new data every day
* Retraining every week
* Comparing performance with previous model

### Cron Schedule

```
Daily:
    collect.py

Every Sunday:
    train.py
    predict.py
    evaluate.py
```

Workflow:

```
Sunday Cron Job
      ↓
Fetch new data
      ↓
Retrain model
      ↓
Evaluate accuracy
      ↓
Replace model if improved
```

---

## 11. Model Evaluation Metrics

We track:

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* Accuracy trend over time per model version

This allows answering: *Which model version was most accurate?*

---

## 12. FastAPI Endpoints

### Predict Weather

```
GET /predict?hours_ahead=3
```

Returns:

```json
{
  "prediction_for": "2026-03-20T15:00:00",
  "predicted_temperature": 31.2,
  "model_version": "v1",
  "hours_ahead": 3
}
```

### Trigger Training

```
POST /train
```

### Trigger Evaluation

```
POST /evaluate
```

### Webhook – Register URL

```
POST /webhook/register
Body: { "url": "https://your-service.com/hook" }
```

### Webhook – Send Now

```
POST /webhook/send?hours_ahead=1
```

Pushes next-hour prediction to all registered URLs immediately.

### Webhook – List

```
GET /webhook/list
```

### Webhook – Unregister

```
DELETE /webhook/unregister
Body: { "url": "https://your-service.com/hook" }
```

### Webhook Payload (sent automatically every hour)

```json
{
  "predicted_temperature": 28.4,
  "hours_ahead": 1,
  "prediction_for": "2026-03-20T15:00:00Z",
  "sent_at": "2026-03-20T14:00:00Z"
}
```

---

## 13. db/postgres.py Responsibilities

Contains connection pooling, insert helpers, and query helpers so all other scripts call simple functions:

```python
from db.postgres import insert_weather, insert_prediction, insert_metrics
from db.postgres import fetch_all_weather, fetch_prediction_vs_actual
```

No raw SQL scattered across scripts.

---

## 14. Deployment Plan on Render

### Services

1. **PostgreSQL** (already running)
2. **FastAPI Web Service**

### Environment Variables

* `DATABASE_URL` — Render PostgreSQL connection string
* `OPENWEATHER_API_KEY` — OpenWeather API key
* `WEATHER_CITY` — City to collect weather for (default: London)
* `MODEL_VERSION` — Current model version tag (default: v1)
* `WEBHOOK_URL` — Optional default webhook URL pre-registered on startup

### Render Constraints

* Free tier sleeps after inactivity
* Cold start delay ~30–60 seconds
* Limited RAM → Avoid heavy models like large LSTM

---

## 15. Accuracy Verification with Real Weather

```
ml/predict.py → INSERT INTO weather_predictions
                          ↓
              (wait for actual data to arrive)
                          ↓
ml/evaluate.py → JOIN predictions with weather_raw
                          ↓
              INSERT INTO model_metrics
```

This creates a continuous feedback loop.

---

## 16. Versioning Strategy

Each model saved with version tag:

```
model_v1.pkl
model_v2.pkl
```

`model.pkl` always points to the latest. `model_version` column in DB tracks which model made each prediction, enabling comparison across versions.

---

## 17. Data Lifecycle Example

| Day | weather_raw | weather_predictions | model_metrics |
|-----|-------------|---------------------|---------------|
| 1   | 24 rows     | 0                   | 0             |
| 3   | 72 rows     | 48 rows             | 24 rows       |
| 30  | 720+ rows   | meaningful ML       | trend visible |

At 720+ training samples, ML predictions become statistically meaningful.

---

## 18. Development Roadmap

### Phase 1 – Data Pipeline
* [x] `db/postgres.py` — connection pool + helpers (including `insert_weather_bulk`)
* [x] `data/bootstrap.py` — dynamic date-range sync (full seed or gap fill); auto-called by `train.py`
* [x] `data/collect.py` — fetch + store with dedup

### Phase 2 – ML Model
* [x] `ml/preprocess.py` — feature engineering
* [x] `ml/train.py` — train + save versioned model
* [x] `ml/predict.py` — predict next hour + store

### Phase 3 – Evaluation System
* [x] `ml/evaluate.py` — JOIN + compute MAE/RMSE + store

### Phase 4 – API Deployment
* [x] FastAPI `/predict`, `/train`, `/evaluate` endpoints
* [x] Webhook system with register/send/list/unregister
* [x] Hourly background scheduler for automatic webhook push
* [ ] Deploy to Render

### Phase 5 – Automation
* [ ] Daily cron: `collect.py`
* [ ] Weekly cron (Sunday): `train.py` → `predict.py` → `evaluate.py`

---

## 19. Strict Implementation Order

```
db/postgres.py
data/collect.py
ml/train.py
ml/predict.py
ml/evaluate.py
```

Skipping this order makes debugging very hard because each step depends on the previous one's output.

---

## 20. Risks & Considerations

* OpenWeather free tier has rate limits
* Render free tier has storage and compute limits
* Model drift may occur due to seasonal weather patterns
* `prediction_for` must align exactly with `recorded_at` for JOIN to work (truncate to hour)

---

## 21. Definition of Project Completion

The system is considered complete when:

* Predictions are generated automatically
* Accuracy metrics are tracked per model version
* Weekly retraining runs without manual intervention
* API is publicly accessible
* Webhook delivers next-hour predictions to registered consumers
