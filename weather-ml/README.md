# Weather Prediction ML

A machine learning system that collects weather data, trains a prediction model, and serves predictions via FastAPI.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export DATABASE_URL=your_postgres_url
   export OPENWEATHER_API_KEY=your_api_key
   export WEATHER_CITY=London
   ```

3. Initialize the database:
   ```bash
   python db/postgres.py
   ```

## Usage

### Collect weather data
```bash
python data/collect.py
```

### Train the model
```bash
python ml/train.py
```

### Evaluate the model
```bash
python ml/evaluate.py
```

### Run the API
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `GET /predict?hours_ahead=3` — Get temperature prediction
- `POST /train` — Trigger model retraining

## Deployment

Deploy the FastAPI service on Render. Set `DATABASE_URL` and `OPENWEATHER_API_KEY` as environment variables in the Render dashboard.
