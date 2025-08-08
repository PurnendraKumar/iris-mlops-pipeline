# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required OS packages before Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# increase timeout


# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy application code and necessary directories
COPY src/ ./src/
COPY models/ ./models/

# Ensure logs directory exists and copy any default log configs
RUN mkdir -p logs
#COPY logs/ ./logs/

# Expose application port
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the application using uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
