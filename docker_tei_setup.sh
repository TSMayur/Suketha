#!/bin/bash
# docker_tei_setup.sh - Official script for Hugging Face TEI on M1/M2 Mac
set -e

echo "Setting up Hugging Face TEI for M1/M2 Mac using the official image..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Stop and remove any existing container
if [ "$(docker ps -a -q -f name=tei-server)" ]; then
    echo "Stopping and removing existing tei-server container..."
    docker rm -f tei-server
fi

# Login to GitHub Container Registry (required for official image)
echo "Please log in to the GitHub Container Registry (ghcr.io)..."
echo "Use your GitHub username and a Personal Access Token (PAT) with read:packages scope."
docker login ghcr.io

# Define the model and the OFFICIAL image from the documentation
MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
DOCKER_IMAGE="ghcr.io/huggingface/text-embeddings-inference:1.2"

echo "Pulling official TEI container: $DOCKER_IMAGE"

# Run the official container, adapted for CPU (removed --gpus flag)
docker run -d \
    --name tei-server \
    -p 8080:80 \
    -e MODEL_ID="$MODEL_ID" \
    --pull always \
    "$DOCKER_IMAGE"

echo "TEI server is starting up..."

# Wait for the server to be ready
echo "Waiting for TEI server to be ready (this may take a minute)..."
for i in {1..60}; do
    if curl -s --fail http://localhost:8080/health > /dev/null 2>&1; then
        echo "TEI server is ready! ✅"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 2
done

if ! curl -s --fail http://localhost:8080/health > /dev/null 2>&1; then
    echo "Error: TEI server failed to start. Check Docker logs with 'docker logs tei-server'"
    exit 1
fi

echo "TEI setup complete!"