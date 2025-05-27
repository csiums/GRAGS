#!/bin/bash

set -e
echo "Starting GoetheGPT setup..."

detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    else
        echo "unknown"
    fi
}

DISTRO=$(detect_distro)

if ! command -v python3 &>/dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install sentence-transformers

echo "Downloading 'bge-reranker-base' model..."
mkdir -p models
python3 -c "
from sentence_transformers import CrossEncoder
CrossEncoder('BAAI/bge-reranker-base').save('models/bge_reranker_base')
"
echo "Model saved to: models/bge_reranker_base"

if ! command -v docker &>/dev/null; then
    echo "Docker is not installed."
    echo "Please install Docker manually before proceeding:"
    echo "→ https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v nvidia-ctk &>/dev/null; then
    echo "NVIDIA Container Toolkit is not installed."
    echo "If you want GPU support, install it manually:"
    echo "→ https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
fi

DOCKER_CMD="docker"
if ! groups "$USER" | grep -q '\bdocker\b'; then
    echo "Your user is not in the 'docker' group."
    echo "Using 'sudo' for Docker commands."
    echo "To avoid this in the future, run:"
    echo "    sudo usermod -aG docker $USER"
    echo "    and then log out and back in."
    DOCKER_CMD="sudo docker"
fi

if [ -f "Dockerfile" ]; then
    echo "Building Docker container..."
    $DOCKER_CMD compose build
    echo "Docker image built successfully."
else
    echo "No Dockerfile found. Skipping Docker build."
fi

read -p "Run GoetheGPT now? [y/N] " answer
case "$answer" in
    [yY][eE][sS]|[yY])
        echo "Launching GoetheGPT..."
        $DOCKER_CMD compose up
        ;;
    *)
        echo "Setup complete. You can start GoetheGPT anytime with:"
        echo "  $DOCKER_CMD compose up"
        ;;
esac
