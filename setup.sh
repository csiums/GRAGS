#!/bin/bash
set -e
set -x

# Sudo password caching: prompt once and keep session alive
if sudo -v; then
  # Keep-alive: update existing sudo time stamp until this script exits
  while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
else
  echo "Sudo authentication failed."
  exit 1
fi

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found."
  exit 1
fi

# Git LFS install (cross-distro)
if ! command -v git-lfs &> /dev/null; then
  echo "git-lfs not found. Installing..."
  if command -v dnf &> /dev/null; then
    sudo dnf install -y git-lfs
  elif command -v zypper &> /dev/null; then
    sudo zypper install -y git-lfs
  elif command -v pacman &> /dev/null; then
    sudo pacman -Sy --noconfirm git-lfs
  elif command -v apk &> /dev/null; then
    sudo apk add git-lfs
  elif command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y git-lfs
  else
    echo "Cannot install git-lfs: No supported package manager found. Please install git-lfs manually."
    exit 1
  fi
fi
git lfs install

# Download model if not present
mkdir -p llm_models
if [ ! -d "llm_models/bge_reranker_base" ]; then
  git clone https://huggingface.co/BAAI/bge-reranker-base llm_models/bge_reranker_base
fi

# --- Python 3.10 and venv setup (for faiss-gpu compatibility) ---
PYTHON_BIN=python3.10
if ! command -v $PYTHON_BIN &> /dev/null; then
  echo "Python 3.10 not found, attempting to install..."
  if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
  elif command -v dnf &> /dev/null; then
    sudo dnf install -y python3.10
  elif command -v zypper &> /dev/null; then
    sudo zypper install -y python310
  elif command -v pacman &> /dev/null; then
    sudo pacman -Sy --noconfirm python310
  else
    echo "Cannot install Python 3.10 automatically. Please install it manually."
    exit 1
  fi
fi

if [ ! -d ".venv" ]; then
  $PYTHON_BIN -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip in venv
pip install --upgrade pip

# --- Conditional FAISS install ---
# Remove faiss-cpu/faiss-gpu from requirements.txt for this step!
grep -vE '^faiss-(cpu|gpu)' requirements.txt > requirements_nofaiss.txt

# Detect NVIDIA GPU and install faiss-gpu or faiss-cpu as appropriate
if command -v nvidia-smi &> /dev/null; then
  echo "NVIDIA GPU detected. Installing faiss-gpu==1.7.2..."
  if ! pip install faiss-gpu==1.7.2; then
    echo "faiss-gpu install failed, falling back to faiss-cpu==1.7.2"
    pip install faiss-cpu==1.7.2
  fi
else
  echo "No NVIDIA GPU detected. Installing faiss-cpu==1.7.2..."
  pip install faiss-cpu==1.7.2
fi

# Install other requirements (excluding faiss-cpu/gpu)
pip install -r requirements_nofaiss.txt
rm requirements_nofaiss.txt

# Ollama install
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found. Installing via official script..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo ""
  echo "Ollama installed."
  echo "Please run: exec \$SHELL"
  echo "Or restart your terminal so the 'ollama' command becomes available in PATH."
  exit 1
fi

if [ -z "$OLLAMA_MODEL" ]; then
  echo "OLLAMA_MODEL environment variable not set in .env"
  exit 1
fi
ollama pull "$OLLAMA_MODEL"

export PYTHONPATH=./scripts:$PYTHONPATH

python -c "from ollama_utils import warn_if_no_gpu; warn_if_no_gpu()"

python -c "
from ollama_utils import ensure_models_available
import os
models = [os.getenv('OLLAMA_MODEL', 'llama3.2')]
ensure_models_available(models)
"

python -c "
from rag_pipeline import load_or_create_vectorstore
load_or_create_vectorstore()
"

echo ""
echo "Setup complete. All models are now available locally for offline use."
