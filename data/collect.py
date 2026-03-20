import os
import requests
from datetime import datetime
from db.postgres import insert_weather

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = os.getenv("WEATHER_CITY", "London")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def fetch_current_weather() -> dict:
    params = {"q": CITY, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()


def parse_weather(data: dict) -> dict:
    return {
        "recorded_at": datetime.utcfromtimestamp(data["dt"]),
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "weather_main": data["weather"][0]["main"]
    }


def collect():
    raw = fetch_current_weather()
    record = parse_weather(raw)
    insert_weather(record)  # skips silently on duplicate recorded_at
    print(f"Collected weather for {record['recorded_at']}")
    return record


if __name__ == "__main__":
    collect()
