import os
import logging
import pickle
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data_preprocessing import load_data, preprocess_data

# Setup logging
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
        self.best_model_name = ""
        self.best_run_id = None
        self.scaler = None

    def train_models(self):
        """Train multiple models and track experiments with MLflow"""
        # Load and preprocess data
        df = load_data()
        X_train, X_test, y_train, y_test, self.scaler = preprocess_data(df)

        # Set experiment
        mlflow.set_experiment("iris-classification")

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name) as run:
                logger.info(f"Training model: {model_name}")

                # Train
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Log params and metrics
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })

                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=f"iris-{model_name}"
                )

                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")

                # Save best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.best_model_name = model_name
                    self.best_run_id = run.info.run_id

        return self.best_model, self.best_model_name

    def save_best_model(self):
        """Save the best performing model and scaler to disk"""
        if self.best_model is not None:
            os.makedirs("models", exist_ok=True)

            with open("models/best_model.pkl", "wb") as f:
                pickle.dump(self.best_model, f)

            with open("models/scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

            logger.info(f"Best model ({self.best_model_name}) saved with accuracy: {self.best_score:.4f}")


def main():
    trainer = ModelTrainer()
    best_model, best_model_name = trainer.train_models()
    trainer.save_best_model()

    # Register best model explicitly
    if trainer.best_run_id:
        model_uri = f"runs:/{trainer.best_run_id}/model"
        mlflow.register_model(model_uri, "iris-best-model")
        logger.info(f"Registered best model from run {trainer.best_run_id}")
    else:
        logger.warning("No best run ID found — model not registered")


if __name__ == "__main__":
    main()
