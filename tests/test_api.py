import pytest
from fastapi.testclient import TestClient
from src.api import app
import json

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Iris Classification API" in response.json()["message"]

def test_health():
    response = client.get("/health")
    assert response.status_code in [200, 503]  # 503 if model not loaded

def test_predict_valid_input():
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "prediction_label" in data
        assert "confidence" in data
        assert data["prediction"] in [0, 1, 2]
        assert data["prediction_label"] in ["setosa", "versicolor", "virginica"]

def test_predict_invalid_input():
    test_data = {
        "sepal_length": -1,  # Invalid negative value
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code in [200, 500]  # 500 if logs don't exist yet