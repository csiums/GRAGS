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

install_docker() {
    echo "Installing Docker..."

    case "$DISTRO" in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y ca-certificates curl gnupg lsb-release
            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/$DISTRO/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            echo \
              "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$DISTRO \
              $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            ;;
        fedora)
            sudo dnf install -y dnf-plugins-core
            sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
            sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            ;;
        arch)
            sudo pacman -Sy --noconfirm docker docker-compose
            ;;
        opensuse*)
            sudo zypper install -y docker docker-compose
            ;;
        *)
            echo "Unsupported distro: $DISTRO. Please install Docker manually."
            return
            ;;
    esac

    sudo systemctl enable docker
    sudo systemctl start docker
}

if ! command -v docker &>/dev/null; then
    install_docker
else
    echo "Docker is already installed."
fi

install_nvidia_toolkit() {
    echo "Installing NVIDIA Container Toolkit..."

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    case "$DISTRO" in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y nvidia-container-toolkit
            ;;
        fedora)
            sudo dnf install -y nvidia-container-toolkit
            ;;
        arch)
            sudo pacman -Sy --noconfirm nvidia-container-toolkit
            ;;
        opensuse*)
            sudo zypper install -y nvidia-container-toolkit
            ;;
        *)
            echo "Unsupported distro for NVIDIA Container Toolkit."
            return
            ;;
    esac

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
}

if ! command -v nvidia-ctk &>/dev/null; then
    install_nvidia_toolkit
else
    echo "NVIDIA Container Toolkit is already installed."
fi

if [ -f "Dockerfile" ]; then
    echo "Building Docker container..."
    docker compose build
    echo "Docker image built successfully."
else
    echo "No Dockerfile found. Skipping Docker build."
fi

read -p "Run GoetheGPT now? [y/N] " answer
case "$answer" in
    [yY][eE][sS]|[yY])
        echo "Launching GoetheGPT..."
        docker compose up
        ;;
    *)
        echo "Setup complete. You can start GoetheGPT anytime with: docker compose up"
        ;;
esac
