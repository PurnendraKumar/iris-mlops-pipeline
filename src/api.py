from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import logging
import json
from datetime import datetime
import sqlite3
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Iris Classification API", version="1.0.0")

# Load model and scaler
try:
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model, scaler = None, None

# Initialize SQLite for logging
def init_db():
    os.makedirs('logs', exist_ok=True)
    conn = sqlite3.connect('logs/predictions.db')
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
    conn.close()

init_db()

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

# Metrics storage
prediction_count = 0
prediction_history = []

@app.get("/")
async def root():
    return {"message": "Iris Classification API", "status": "healthy"}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    global prediction_count
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Scale features
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        prediction_proba = model.predict_proba(scaled_data)[0]
        confidence = float(max(prediction_proba))
        
        # Map prediction to name
        class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        prediction_name = class_names[prediction]
        
        # Create response
        timestamp = datetime.now().isoformat()
        response = PredictionResponse(
            prediction=int(prediction),
            prediction_name=prediction_name,
            confidence=confidence,
            timestamp=timestamp
        )
        
        # Log prediction
        log_data = {
            'timestamp': timestamp,
            'input': features.dict(),
            'prediction': int(prediction),
            'prediction_name': prediction_name,
            'confidence': confidence
        }
        logger.info(f"Prediction made: {log_data}")
        
        # Store in database
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (timestamp, input_data, prediction, prediction_name, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, json.dumps(features.dict()), int(prediction), prediction_name, confidence))
        conn.commit()
        conn.close()
        
        # Update metrics
        prediction_count += 1
        prediction_history.append(log_data)
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Monitoring metrics endpoint"""
    conn = sqlite3.connect('logs/predictions.db')
    cursor = conn.cursor()
    
    # Get prediction counts by class
    cursor.execute('''
        SELECT prediction_name, COUNT(*) 
        FROM predictions 
        GROUP BY prediction_name
    ''')
    class_counts = dict(cursor.fetchall())
    
    # Get recent predictions
    cursor.execute('''
        SELECT * FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT 10
    ''')
    recent_predictions = cursor.fetchall()
    
    conn.close()
    
    return {
        "total_predictions": prediction_count,
        "predictions_by_class": class_counts,
        "recent_predictions_count": len(recent_predictions),
        "model_status": "loaded" if model else "not_loaded"
    }

@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    """Get prediction history"""
    conn = sqlite3.connect('logs/predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    predictions = []
    for row in cursor.fetchall():
        predictions.append({
            'id': row[0],
            'timestamp': row[1],
            'input_data': json.loads(row[2]),
            'prediction': row[3],
            'prediction_name': row[4],
            'confidence': row[5]
        })
    
    conn.close()
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)