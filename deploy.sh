#!/bin/bash

set -e

ENVIRONMENT=${1:-local}
DOCKER_IMAGE=${DOCKER_IMAGE:-purnendrakumar/iris-mlops-pipeline:latest}

echo "Deploying to $ENVIRONMENT environment..."

case $ENVIRONMENT in
  "local")
    echo "Deploying locally with Docker..."
    
    # Stop existing containers
    docker-compose down || true
    
    # Pull latest image
    docker pull $DOCKER_IMAGE || echo "Using local image"
    
    # Start services
    docker-compose up -d
    
    # Wait for health check
    echo "Waiting for service to be healthy..."
    for i in {1..30}; do
      if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "Service is healthy!"
        break
      fi
      echo "Attempt $i/30: Service not ready yet..."
      sleep 10
    done
    ;;
    
 
    #   "production")
    #    echo "Deploying to production..."
    # Add production deployment logic
    # This could include AWS ECS, Kubernetes, etc.
    ;;
    
  *)
    echo "Unknown environment: $ENVIRONMENT"
    exit 1
    ;;
esac

echo "Deployment to $ENVIRONMENT completed successfully!"