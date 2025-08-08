from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import logging
import json
from datetime import datetime
import sqlite3
import os

# Ensure required directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("iris-api")

app = FastAPI(title="Iris Classification API", version="1.0.0")

# Load model and scaler
model, scaler = None, None
try:
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {e}")

# Initialize SQLite for prediction logging


def init_db():
    with sqlite3.connect('logs/predictions.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_data TEXT,
                prediction INTEGER,
                prediction_name TEXT,
                confidence REAL
            )
        ''')
        conn.commit()


init_db()

# Request and response models


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


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
        raise HTTPException(
            status_code=503, detail="Model or scaler not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_array = np.array([[features.sepal_length, features.sepal_width,
                                 features.petal_length, features.petal_width]])
        scaled_input = scaler.transform(input_array)

        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]
        confidence = float(np.max(proba))

        class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        prediction_name = class_names[prediction]
        timestamp = datetime.utcnow().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "input": features.dict(),
            "prediction": int(prediction),
            "prediction_name": prediction_name,
            "confidence": confidence
        }

        logger.info(f"Prediction: {log_entry}")

        # Store to SQLite
        with sqlite3.connect('logs/predictions.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (timestamp, input_data, prediction, prediction_name, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, json.dumps(features.dict()), prediction, prediction_name, confidence))
            conn.commit()

        return PredictionResponse(
            prediction=prediction,
            prediction_name=prediction_name,
            confidence=confidence,
            timestamp=timestamp
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/metrics")
async def get_metrics():
    """Monitoring metrics endpoint"""
    with sqlite3.connect('logs/predictions.db') as conn:
        cursor = conn.cursor()
        # Get prediction counts by class
        cursor.execute(
            'SELECT prediction_name, COUNT(*) FROM predictions GROUP BY prediction_name')
        class_counts = dict(cursor.fetchall())

        # Get total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]

        # Get recent predictions
        cursor.execute(
            'SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10')
        recent = cursor.fetchall()

    return {
        "total_predictions": total_predictions,
        "predictions_by_class": class_counts,
        "recent_predictions_count": len(recent),
        "model_status": "loaded" if model else "not_loaded"
    }


@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    with sqlite3.connect('logs/predictions.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?''', (limit,))
        rows = cursor.fetchall()

    return {
        "predictions": [
            {
                "id": row[0],
                "timestamp": row[1],
                "input_data": json.loads(row[2]),
                "prediction": row[3],
                "prediction_name": row[4],
                "confidence": row[5]
            }
            for row in rows
        ]
    }


# Only runs when executed with `python api.py`, not `uvicorn`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
