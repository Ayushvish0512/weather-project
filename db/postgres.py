import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from the project root regardless of where the script is run from
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

# Connection pool (min 1, max 5 connections)
_pool = None


def get_pool():
    global _pool
    if _pool is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise EnvironmentError(
                "DATABASE_URL is not set. Check your .env file exists at the project root."
            )
        _pool = pool.SimpleConnectionPool(1, 5, db_url)
    return _pool


def get_connection():
    return get_pool().getconn()


def release_connection(conn):
    get_pool().putconn(conn)


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS weather_raw (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP UNIQUE,
                    temperature FLOAT,
                    humidity FLOAT,
                    pressure FLOAT,
                    wind_speed FLOAT,
                    weather_main TEXT
                );

                CREATE TABLE IF NOT EXISTS weather_predictions (
                    id SERIAL PRIMARY KEY,
                    prediction_for TIMESTAMP,
                    predicted_temp FLOAT,
                    model_version TEXT
                );

                CREATE TABLE IF NOT EXISTS model_metrics (
                    id SERIAL PRIMARY KEY,
                    evaluated_at TIMESTAMP,
                    mae FLOAT,
                    rmse FLOAT,
                    model_version TEXT
                );
            """)
        conn.commit()
        print("Database tables initialized.")
    finally:
        release_connection(conn)


# --- Insert helpers ---

def insert_weather(record: dict):
    """Insert a row into weather_raw. Skips on duplicate recorded_at."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO weather_raw
                    (recorded_at, temperature, humidity, pressure, wind_speed, weather_main)
                VALUES
                    (%(recorded_at)s, %(temperature)s, %(humidity)s,
                     %(pressure)s, %(wind_speed)s, %(weather_main)s)
                ON CONFLICT (recorded_at) DO NOTHING
            """, record)
        conn.commit()
    finally:
        release_connection(conn)


def insert_weather_bulk(records: list[dict]) -> int:
    """Bulk insert into weather_raw using execute_values (single query). Returns row count."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO weather_raw
                    (recorded_at, temperature, humidity, pressure, wind_speed, weather_main)
                VALUES %s
                ON CONFLICT (recorded_at) DO NOTHING
            """, [(
                r["recorded_at"], r["temperature"], r["humidity"],
                r["pressure"], r["wind_speed"], r["weather_main"]
            ) for r in records], page_size=1000)
        conn.commit()
        return len(records)
    finally:
        release_connection(conn)


def insert_prediction(prediction_for: datetime, predicted_temp: float, model_version: str):
    """Insert a row into weather_predictions."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO weather_predictions (prediction_for, predicted_temp, model_version)
                VALUES (%s, %s, %s)
            """, (prediction_for, predicted_temp, model_version))
        conn.commit()
    finally:
        release_connection(conn)


def insert_metrics(mae: float, rmse: float, model_version: str):
    """Insert a row into model_metrics."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_metrics (evaluated_at, mae, rmse, model_version)
                VALUES (%s, %s, %s, %s)
            """, (datetime.utcnow(), mae, rmse, model_version))
        conn.commit()
    finally:
        release_connection(conn)


# --- Query helpers ---

def get_latest_recorded_at():
    """Return the most recent recorded_at from weather_raw, or None if empty."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(recorded_at) FROM weather_raw")
            result = cur.fetchone()[0]
        return result
    finally:
        release_connection(conn)


def fetch_all_weather():
    """Return all rows from weather_raw as a list of dicts."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM weather_raw ORDER BY recorded_at")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        release_connection(conn)


def fetch_prediction_vs_actual():
    """JOIN predictions with actual weather for evaluation."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    p.id,
                    p.prediction_for,
                    p.predicted_temp,
                    p.model_version,
                    r.temperature AS actual_temp
                FROM weather_predictions p
                JOIN weather_raw r ON p.prediction_for = r.recorded_at
                ORDER BY p.prediction_for
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        release_connection(conn)


if __name__ == "__main__":
    init_db()
