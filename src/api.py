# src/api.py
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field
import pickle
import numpy as np
import logging
import json
from datetime import datetime
import sqlite3
import os
import threading
import subprocess
import time
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


# Ensure required directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/predictions.log"), logging.StreamHandler()],
)
logger = logging.getLogger("iris-api")

app = FastAPI(title="Iris Classification API", version="1.0.0")

# Define metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'api_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

# Global model/scaler and lock for safe reloads
model = None
scaler = None
_model_lock = threading.Lock()


def load_model_and_scaler():
    """Load model and scaler from disk into global variables safely."""
    global model, scaler
    with _model_lock:
        try:
            with open("models/best_model.pkl", "rb") as f:
                loaded_model = pickle.load(f)
            with open("models/scaler.pkl", "rb") as f:
                loaded_scaler = pickle.load(f)
            model = loaded_model
            scaler = loaded_scaler
            logger.info("Model and scaler loaded successfully.")
            return True
        except Exception as exc:
            logger.exception(f"Failed to load model or scaler: {exc}")
            model = None
            scaler = None
            return False


# Attempt initial load
load_model_and_scaler()

# Initialize SQLite for prediction logging
def init_db():
    with sqlite3.connect("logs/predictions.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_data TEXT,
                prediction INTEGER,
                prediction_name TEXT,
                confidence REAL
            )
            """
        )
        conn.commit()


init_db()

# Prometheus metrics
PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

# Request and response models with validation (reasonable ranges)
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0.0, le=10.0)
    sepal_width: float = Field(..., ge=0.0, le=10.0)
    petal_length: float = Field(..., ge=0.0, le=10.0)
    petal_width: float = Field(..., ge=0.0, le=10.0)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_name: str
    confidence: float
    timestamp: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Iris Classification API"}


@app.get("/health")
async def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")
    return {"status": "healthy", "model_loaded": True}


# Middleware to track metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)

    return response


@app.get("/metrics")
def prometheus_metrics():
    """
    Prometheus-compatible metrics endpoint.
    Configure Prometheus to scrape this endpoint.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    Predict endpoint. Validates input using Pydantic, logs results to SQLite,
    and exposes Prometheus metrics for request count and latency.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    PREDICTION_REQUESTS.inc()

    try:
        input_array = np.array(
            [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
        )
        scaled_input = scaler.transform(input_array)

        prediction = model.predict(scaled_input)[0]
        # if model does not support predict_proba, handle gracefully
        proba = None
        confidence = 0.0
        try:
            proba = model.predict_proba(scaled_input)[0]
            confidence = float(np.max(proba))
        except Exception:
            # fallback: if no probabilities, set confidence to 1.0
            confidence = 1.0

        class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
        prediction_name = class_names.get(int(prediction), str(prediction))
        timestamp = datetime.utcnow().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "input": features.dict(),
            "prediction": int(prediction),
            "prediction_name": prediction_name,
            "confidence": confidence,
        }

        logger.info(f"Prediction: {log_entry}")

        # Store to SQLite
        try:
            with sqlite3.connect("logs/predictions.db") as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO predictions (
                        timestamp, input_data, prediction,
                        prediction_name, confidence
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        json.dumps(features.dict()),
                        int(prediction),
                        prediction_name,
                        confidence,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.exception(f"Failed to write prediction to DB: {e}")

        return PredictionResponse(
            prediction=int(prediction),
            prediction_name=prediction_name,
            confidence=confidence,
            timestamp=timestamp,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    finally:
        PREDICTION_LATENCY.observe(time.time() - start_time)


@app.get("/metrics/sqlite")
async def get_metrics():
    """
    Legacy sqlite-based metrics endpoint (keeps your original behavior).
    Returns totals and recent predictions.
    """
    with sqlite3.connect("logs/predictions.db") as conn:
        cursor = conn.cursor()

        # Get prediction counts by class
        cursor.execute("SELECT prediction_name, COUNT(*) FROM predictions GROUP BY prediction_name")
        class_counts = dict(cursor.fetchall())

        # Get total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        # Get recent predictions
        cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10")
        recent = cursor.fetchall()

    return {
        "total_predictions": total_predictions,
        "predictions_by_class": class_counts,
        "recent_predictions_count": len(recent),
        "model_status": "loaded" if model else "not_loaded",
    }


@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    with sqlite3.connect("logs/predictions.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    return {
        "predictions": [
            {
                "id": row[0],
                "timestamp": row[1],
                "input_data": json.loads(row[2]),
                "prediction": row[3],
                "prediction_name": row[4],
                "confidence": row[5],
            }
            for row in rows
        ]
    }


def _run_retrain_command():
    """
    Runs the training command synchronously (but invoked from background).
    Adjust the command below to match your project's training entrypoint.
    """
    try:
        logger.info("Retraining started (background)...")
        # use -m to run package module if that's how your project is structured
        # ensure Python in PATH inside the container/runner is the right one
        #subprocess.run(["python", "-m", "src.model_training"], check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "src.model_training"], check=True, capture_output=True)
        logger.info("Retraining completed. Attempting to reload model from disk...")
        # reload model after successful retraining
        if load_model_and_scaler():
            logger.info("Model reloaded successfully after retraining.")
        else:
            logger.error("Model reload failed after retraining.")
    except subprocess.CalledProcessError as exc:
        logger.exception(f"Retraining process failed: returncode={exc.returncode}; stdout={exc.stdout}; stderr={exc.stderr}")
    except Exception as e:
        logger.exception(f"Unexpected error during retraining: {e}")


@app.post("/retrain", status_code=202)
async def retrain(background_tasks: BackgroundTasks):
    """
    Trigger model retraining in background. Returns 202 Accepted immediately.
    The background task will run the training script (should save model & scaler to models/).
    """
    # Security note: in production, protect this endpoint (auth, CSRF, etc.)
    background_tasks.add_task(_run_retrain_command)
    return {"status": "accepted", "message": "Retraining started in background"}


# Only runs when executed with `python src/api.py`, not `uvicorn`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
