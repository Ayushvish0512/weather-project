import os
import psycopg2
from psycopg2 import pool
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL")

# Connection pool (min 1, max 5 connections)
_pool = None


def get_pool():
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise EnvironmentError("DATABASE_URL environment variable is not set")
        _pool = pool.SimpleConnectionPool(1, 5, DATABASE_URL)
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
