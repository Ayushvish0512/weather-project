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
* Tables planned for:

  * `weather_raw`
  * `weather_predictions`
  * `model_metrics`

### Data Source

* **Primary Source**: OpenWeather API
* Data fields to collect:

  * datetime
  * temperature
  * humidity
  * pressure
  * wind speed
  * weather condition

---

## 3. System Architecture Overview

```
OpenWeather API  →  Data Collector Script  →  PostgreSQL (Render)
                                             ↓
                                       ML Training Pipeline
                                             ↓
                                      Trained Model (.pkl)
                                             ↓
                                  FastAPI Prediction Service
                                             ↓
                               Accuracy Checker vs Real Weather
```

---

## 4. Machine Learning Strategy

### Initial Model Type

We will start with **supervised learning**, not reinforcement learning, because:

* We have labeled historical data
* Reinforcement learning is not suitable for time-series prediction directly

### Recommended Algorithms

* Linear Regression (baseline)
* Random Forest Regressor (primary)
* Optional upgrade: LSTM for time series

---

## 5. Project Folder Structure

```
weather-ml/
│
├── app/
│   ├── main.py                # FastAPI app
│   ├── predict.py             # Prediction endpoint
│
├── ml/
│   ├── train.py               # Model training script
│   ├── preprocess.py          # Data cleaning and feature engineering
│   ├── evaluate.py            # Accuracy comparison with real weather
│   └── model.pkl              # Saved trained model
│
├── data/
│   └── collect.py             # Script to fetch weather from OpenWeather
│
├── db/
│   └── postgres.py            # DB connection utilities
│
├── requirements.txt
└── README.md
```

---

## 6. Database Schema (Planned)

### Table: weather_raw

Stores actual weather fetched from API.

```sql
CREATE TABLE weather_raw (
    id SERIAL PRIMARY KEY,
    recorded_at TIMESTAMP,
    temperature FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    wind_speed FLOAT,
    weather_main TEXT
);
```

### Table: weather_predictions

Stores model predictions.

```sql
CREATE TABLE weather_predictions (
    id SERIAL PRIMARY KEY,
    prediction_time TIMESTAMP,
    predicted_temp FLOAT,
    predicted_humidity FLOAT,
    model_version TEXT
);
```

### Table: model_metrics

Tracks accuracy over time.

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

## 7. ML Training Workflow

### Step 1 – Fetch Historical Data

* Use OpenWeather historical endpoint
* Store daily/hourly weather into PostgreSQL

### Step 2 – Feature Engineering

Convert datetime into model-friendly features:

* hour
* day_of_week
* month

### Step 3 – Train Model

Example features:

```
X = [hour, humidity, pressure, wind_speed]
y = temperature
```

### Step 4 – Save Model

```
joblib.dump(model, "ml/model.pkl")
```

---

## 8. Weekly Retraining Strategy

This simulates reinforcement learning by:

* Collecting new data every day
* Retraining every week
* Comparing performance with previous model

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

## 9. Model Evaluation Metrics

We will track:

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* Accuracy trend over time

This allows monitoring whether weekly retraining is actually improving predictions.

---

## 10. FastAPI Endpoints (Planned)

### Predict Weather

```
GET /predict?hours_ahead=3
```

Returns:

```
{
  "predicted_temperature": 31.2,
  "confidence": 0.82
}
```

### Trigger Training

```
POST /train
```

Used manually or via cron to retrain model.

---

## 11. Deployment Plan on Render

### Services

We will deploy **two services** on Render free tier:

1. **PostgreSQL** (already running)
2. **FastAPI Web Service**

### Render Constraints

* Free tier sleeps after inactivity
* Cold start delay ~30–60 seconds
* Limited RAM → Avoid heavy models like large LSTM

---

## 12. Accuracy Verification with Real Weather

After prediction:

```
Model Prediction → Store in weather_predictions
                     ↓
Later fetch actual weather
                     ↓
Compare and store error in model_metrics
```

This creates a continuous feedback loop.

---

## 13. Development Roadmap

### Phase 1 – Data Pipeline

* [ ] Create OpenWeather fetch script
* [ ] Store hourly data in PostgreSQL

### Phase 2 – ML Model

* [ ] Create preprocessing pipeline
* [ ] Train baseline regression model
* [ ] Save model.pkl

### Phase 3 – Evaluation System

* [ ] Build comparison script
* [ ] Store MAE & RMSE

### Phase 4 – API Deployment

* [ ] Build FastAPI predict endpoint
* [ ] Deploy to Render

### Phase 5 – Automation

* [ ] Weekly retraining cron job
* [ ] Daily data ingestion job

---

## 14. Next Immediate Tasks

The next files to implement in order:

1. `data/collect.py`
2. `db/postgres.py`
3. `ml/preprocess.py`
4. `ml/train.py`

We will start with **data collection**, because model training depends on stored historical data.

---

## 15. Risks & Considerations

* OpenWeather free tier has rate limits
* Render free tier has storage and compute limits
* Model drift may occur due to seasonal weather patterns

---

## 16. Versioning Strategy

Each model will be stored with version tag:

```
model_v1.pkl
model_v2.pkl
```

Database column `model_version` will track which model made each prediction.

---

## 17. Definition of Project Completion

The system is considered complete when:

* Predictions are generated automatically
* Accuracy metrics are tracked
* Weekly retraining runs without manual intervention
* API is publicly accessible
