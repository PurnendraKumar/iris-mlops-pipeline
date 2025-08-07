import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, preprocess_data

def test_load_data():
    """Test data loading"""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150  # Iris dataset has 150 samples
    assert 'target' in df.columns
    assert 'target_names' in df.columns

def test_preprocess_data():
    """Test data preprocessing"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Check shapes
    assert X_train.shape[0] == 120  # 80% of 150
    assert X_test.shape[0] == 30    # 20% of 150
    assert X_train.shape[1] == 4    # 4 features
    
    # Check scaling
    assert abs(X_train.mean()) < 1e-10  # Mean should be close to 0
    assert abs(X_train.std() - 1) < 1e-10  # Std should be close to 1