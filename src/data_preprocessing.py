import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess the data"""
    logger.info("Starting data preprocessing")
    
    # Features and target
    X = df.drop(['target', 'target_names'], axis=1)
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Data preprocessing completed. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data():
    """Save processed data"""
    df = load_data()
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save raw data
    df.to_csv('data/raw/iris.csv', index=False)
    
    # Process and save
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Save processed data
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    logger.info("Processed data saved successfully")

if __name__ == "__main__":
    save_processed_data()