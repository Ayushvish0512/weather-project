"""
data/bootstrap.py
Downloads historical weather data year-by-year and saves as CSV in data/.
Already downloaded files are skipped automatically.
Change HISTORY_END to extend the range.
"""
import os
import time
import requests
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

HISTORY_START = date(2020, 1, 1)          # fixed — never change
HISTORY_END   = date.today() - timedelta(days=1)  # always yesterday — auto-updates

BASE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=28.4595&longitude=77.0266"
    "&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,"
    "dew_point_2m,pressure_msl,cloudcover,visibility,windspeed_10m,"
    "winddirection_10m,windgusts_10m,precipitation,rain,weathercode"
)

COLUMN_MAP = {
    "time":                    "recorded_at",
    "temperature_2m":          "temperature",
    "apparent_temperature":    "feels_like",
    "relative_humidity_2m":    "humidity",
    "dew_point_2m":            "dew_point",
    "pressure_msl":            "pressure",
    "cloudcover":              "cloudcover",
    "visibility":              "visibility",
    "windspeed_10m":           "wind_speed",
    "winddirection_10m":       "wind_direction",
    "windgusts_10m":           "wind_gusts",
    "precipitation":           "precipitation",
    "rain":                    "rain",
    "weathercode":             "weather_main",
}


def download_year(start: date, end: date) -> str:
    csv_path = f"data/weather_{start}_{end}.csv"

    # For the current (partial) year, the end date changes daily.
    # Delete any stale file for the same start date before re-downloading.
    if not os.path.exists(csv_path):
        stale = [
            f for f in os.listdir("data")
            if f.startswith(f"weather_{start}_") and f.endswith(".csv")
        ]
        for s in stale:
            os.remove(f"data/{s}")
            print(f"Removed stale file: {s}")

    if os.path.exists(csv_path):
        print(f"Already exists, skipping: {csv_path}")
        return csv_path

    url = f"{BASE_URL}&start_date={start}&end_date={end}"
    print(f"Downloading {start} → {end} ...", end=" ", flush=True)
    for attempt in range(5):
        resp = requests.get(url, timeout=60)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"\n  Rate limited (429). Waiting {wait}s before retry {attempt + 1}/5...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f"Failed to download {start} → {end} after 5 attempts (rate limited).")

    df = pd.DataFrame(resp.json()["hourly"])
    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])

    numeric_cols = [c for c in df.columns if c != "recorded_at"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["temperature", "humidity", "pressure"], inplace=True)

    df.to_csv(csv_path, index=False)
    print(f"{len(df)} rows → {csv_path}")
    return csv_path


def sync():
    chunk_start = HISTORY_START
    while chunk_start <= HISTORY_END:
        chunk_end = min(chunk_start + relativedelta(years=1) - timedelta(days=1), HISTORY_END)
        download_year(chunk_start, chunk_end)
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(2)   # be polite to the API between chunks
    print("Bootstrap complete.")


if __name__ == "__main__":
    sync()
