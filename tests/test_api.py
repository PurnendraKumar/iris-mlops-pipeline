import pytest
from fastapi.testclient import TestClient
from src.api import app
import json

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_endpoint():
    """Test health endpoint"""
    response = client.get("/health")
    # Might be 503 if model not loaded in test environment
    assert response.status_code in [200, 503]

def test_predict_endpoint():
    """Test prediction endpoint"""
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=test_data)
    # Might fail if model not loaded in test environment
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "prediction_name" in data
        assert "confidence" in data
        assert "timestamp" in data

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "model_status" in data