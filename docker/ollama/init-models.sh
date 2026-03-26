#!/bin/bash
# Pull the configured model on first startup if not already cached.
# Usage: Run from host after compose is up:
#   docker compose exec mist-ollama bash /app/init-models.sh
# Or: bash docker/ollama/init-models.sh (from host, uses docker exec)

set -e

MODEL="${MODEL:-qwen2.5:7b-instruct}"

echo "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if docker compose exec mist-ollama curl -sf http://localhost:11434/ > /dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Ollama did not become ready in 60s."
        exit 1
    fi
    sleep 2
done

echo "Pulling model: $MODEL (this may take several minutes on first run)..."
docker compose exec mist-ollama ollama pull "$MODEL"
echo "Model $MODEL ready."
