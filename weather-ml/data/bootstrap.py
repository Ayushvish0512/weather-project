"""
data/bootstrap.py

Two modes:
  1. Full seed  — DB is empty → fetch from HISTORY_START to yesterday
  2. Gap fill   — DB has data but is behind → fetch from (last_recorded_at + 1h) to yesterday

Called automatically by ml/train.py before training, or run manually.
"""
import requests
import pandas as pd
from datetime import date, timedelta
from db.postgres import insert_weather_bulk, get_latest_recorded_at

BASE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=28.4595&longitude=77.0266"
    "&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,"
    "dew_point_2m,pressure_msl,cloudcover,visibility,windspeed_10m,"
    "winddirection_10m,windgusts_10m,precipitation,rain,weathercode"
)

HISTORY_START = date(2020, 1, 1)   # fixed earliest date — never changes

COLUMN_MAP = {
    "time":                 "recorded_at",
    "temperature_2m":       "temperature",
    "relative_humidity_2m": "humidity",
    "pressure_msl":         "pressure",
    "windspeed_10m":        "wind_speed",
    "weathercode":          "weather_main",
}


def _build_url(start: date, end: date) -> str:
    return f"{BASE_URL}&start_date={start}&end_date={end}"


def _fetch_and_clean(start: date, end: date) -> pd.DataFrame:
    url = _build_url(start, end)
    print(f"Fetching {start} → {end} ...")
    hourly = requests.get(url, timeout=120).json()["hourly"]
    df = pd.DataFrame(hourly)[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    numeric_cols = ["temperature", "humidity", "pressure", "wind_speed", "weather_main"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df


def sync():
    """
    Determine the correct date range and sync weather_raw with Open-Meteo archive.
    - If DB is empty  → full seed from HISTORY_START to yesterday
    - If DB is behind → gap fill from day-after-last-record to yesterday
    - If DB is current → nothing to do
    """
    yesterday = date.today() - timedelta(days=1)
    last_dt = get_latest_recorded_at()   # returns None if table is empty

    if last_dt is None:
        start = HISTORY_START
        print("DB is empty. Running full seed.")
    else:
        last_date = last_dt.date()
        if last_date >= yesterday:
            print(f"DB is up to date (last record: {last_date}). Nothing to sync.")
            return
        start = last_date + timedelta(days=1)
        print(f"DB gap detected. Last record: {last_date}. Filling {start} → {yesterday}.")

    df = _fetch_and_clean(start, yesterday)
    inserted = insert_weather_bulk(df.to_dict(orient="records"))
    print(f"Inserted {inserted} new rows into weather_raw.")


if __name__ == "__main__":
    sync()
