import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle
import logging
from src.data_preprocessing import load_data, preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        self.best_model = None
        self.best_score = 0
        self.scaler = None
    
    def train_models(self):
        """Train multiple models and track with MLflow"""
        # Load and preprocess data
        df = load_data()
        X_train, X_test, y_train, y_test, self.scaler = preprocess_data(df)
        
        mlflow.set_experiment("iris-classification")
        
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                logger.info(f"Training {model_name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Log parameters
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=f"iris-{model_name}"
                )
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
                
                # Track best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.best_model_name = model_name
        
        return self.best_model, self.best_model_name
    
    def save_best_model(self):
        """Save the best model and scaler"""
        if self.best_model is not None:
            # Create models directory
            import os
            os.makedirs('models', exist_ok=True)
            
            # Save model and scaler
            with open('models/best_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
            
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"Best model ({self.best_model_name}) saved with accuracy: {self.best_score:.4f}")

def main():
    trainer = ModelTrainer()
    best_model, best_model_name = trainer.train_models()
    trainer.save_best_model()
    
    # Register best model in MLflow
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "iris-best-model")

if __name__ == "__main__":
    main()