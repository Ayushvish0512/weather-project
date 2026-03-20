import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.predict import router as predict_router
from app.webhook import router as webhook_router, dispatch_to_webhooks


async def hourly_webhook_scheduler():
    """Background task: push next-hour prediction every 60 minutes."""
    while True:
        await asyncio.sleep(3600)  # wait 1 hour
        await dispatch_to_webhooks(hours_ahead=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(hourly_webhook_scheduler())
    yield
    task.cancel()


app = FastAPI(title="Weather Prediction API", lifespan=lifespan)

app.include_router(predict_router)
app.include_router(webhook_router)


@app.get("/")
def root():
    return {"status": "Weather ML API is running"}
