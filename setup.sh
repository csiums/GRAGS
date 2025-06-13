#!/bin/bash
set -e
set -x

# Sudo Password verification and extending the timeout
if sudo -v; then
  while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
else
  echo "Sudo authentication failed."
  exit 1
fi

# .env file check and export environment variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found."
  exit 1
fi

# Install git-lfs if not already installed (cross-distro support)
if ! command -v git-lfs &> /dev/null; then
  echo "git-lfs not found. Installing..."
  if command -v dnf &> /dev/null; then
    sudo dnf install -y git-lfs
  elif command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y git-lfs
  else
    echo "Unable to install git-lfs automatically. Please install manually."
    exit 1
  fi
  git lfs install
fi

# Model download (if not already present)
mkdir -p llm_models
if [ ! -d "llm_models/bge_reranker_base" ]; then
  git clone https://huggingface.co/BAAI/bge-reranker-base llm_models/bge_reranker_base
fi

# Python 3.10 setup
PYTHON_BIN=python3.10
if ! command -v $PYTHON_BIN &> /dev/null; then
  echo "Python 3.10 not found, attempting installation..."
  if command -v dnf &> /dev/null; then
    sudo dnf install -y python3.10
  else
    echo "Python 3.10 installation failed. Please install manually."
    exit 1
  fi
fi

# Set up virtual environment if not already created
if [ ! -d ".venv" ]; then
  $PYTHON_BIN -m venv .venv
fi
source .venv/bin/activate

# Update pip
pip install --upgrade pip || { echo 'Pip upgrade failed'; exit 1; }

# Install dependencies excluding Faiss GPU
grep -vE '^faiss-(cpu|gpu)' requirements.txt > requirements_nofaiss.txt
pip install -r requirements_nofaiss.txt || { echo 'Requirements installation failed'; exit 1; }
rm requirements_nofaiss.txt

# Install Faiss for CPU (no GPU)
echo "Installing Faiss CPU..."
pip install faiss-cpu || { echo 'Faiss installation failed'; exit 1; }

# Install Ollama if not found
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo "Ollama installed. Please restart your terminal or run 'exec \$SHELL'."
  exit 1
fi

# Check if OLLAMA_MODEL is set in .env
if [ -z "$OLLAMA_MODEL" ]; then
  echo "OLLAMA_MODEL not set in .env!"
  exit 1
fi

# Pull the Ollama model
ollama pull "$OLLAMA_MODEL"

# Set PYTHONPATH for scripts
export PYTHONPATH=./scripts:$PYTHONPATH

# Check for GPU warning
python -c "from scripts.ollama_utils import warn_if_no_gpu; warn_if_no_gpu()"

# Ensure models are available
python -c "
from scripts.ollama_utils import ensure_models_available
import os
models = [os.getenv('OLLAMA_MODEL', 'llama3.2')]
ensure_models_available(models)
"

# Create or load the vectorstore
python -c "
from scripts.rag_pipeline import load_or_create_vectorstore
load_or_create_vectorstore()
"

echo ""
echo "Setup completed. All models are now available locally and offline."
