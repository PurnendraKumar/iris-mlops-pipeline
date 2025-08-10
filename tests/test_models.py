import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from src.data_preprocessing import load_and_preprocess_data
import os

def test_data_preprocessing():
    """Test data preprocessing function"""
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Check shapes
    assert X_train.shape[1] == 4  # 4 features
    assert X_test.shape[1] == 4
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    
    # Check files are created
    assert os.path.exists('data/raw/iris_raw.csv')
    assert os.path.exists('data/processed/train.csv')
    assert os.path.exists('data/processed/test.csv')
    assert os.path.exists('models/scaler.joblib')

def test_data_quality():
    """Test data quality"""
    iris = load_iris()
    assert iris.data.shape == (150, 4)
    assert len(np.unique(iris.target)) == 3