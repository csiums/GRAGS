#!/bin/bash

set -e  # Exit immediately on errors
echo "🛠Starting GoetheGPT setup..."

# 1. Check Python
if ! command -v python3 &>/dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# 2. Create virtual environment (optional)
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 3. Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install sentence-transformers

# 4. Download and save bge-reranker-base model
echo "⬇️  Downloading 'bge-reranker-base' model..."
mkdir -p models
python3 -c "
from sentence_transformers import CrossEncoder
CrossEncoder('BAAI/bge-reranker-base').save('models/bge_reranker_base')
"
echo "Model saved to: models/bge_reranker_base"

# 5. Build Docker image if Dockerfile is present
if [ -f "Dockerfile" ]; then
    echo "🐳 Building Docker container..."
    docker-compose build
    echo "Docker image built successfully."
else
    echo "No Dockerfile found. Skipping Docker build."
fi

# 6. Ask whether to run now
read -p "🚀 Run GoetheGPT now? [y/N] " answer
case "$answer" in
    [yY][eE][sS]|[yY])
        echo "Launching GoetheGPT..."
        docker-compose up
        ;;
    *)
        echo "Setup complete. You can start GoetheGPT anytime with: docker-compose up"
        ;;
esac
