# iris-mlops-pipeline
# This repo is for MLOps pipeline for an ML model using a well-known Iris (classification) open dataset
# 1. Clone repository
git clone https://github.com/yourusername/iris-mlops-pipeline.git
cd iris-mlops-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data and train models
python src/data_preprocessing.py
python src/model_training.py

# 4. Start MLflow UI (in background)
mlflow ui &

# 5. Run API locally
uvicorn src.api:app --reload

# 6. Run with Docker
docker-compose up -d

# 7. Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'