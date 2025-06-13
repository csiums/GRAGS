#!/bin/bash
set -e
set -x

# --- Load environment variables ---
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "❌ .env file not found."
  exit 1
fi

# --- Ensure git-lfs is installed (Fedora/openSUSE/Arch/Alpine/Ubuntu) ---
if ! command -v git-lfs &> /dev/null; then
  echo "🔄 git-lfs not found. Installing..."

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
    echo "❌ Cannot install git-lfs: No supported package manager found. Please install git-lfs manually."
    exit 1
  fi
fi
git lfs install

# --- Download Hugging Face models for offline use ---
mkdir -p llm_models
if [ ! -d "llm_models/bge_reranker_base" ]; then
  git clone https://huggingface.co/BAAI/bge-reranker-base llm_models/bge_reranker_base
fi

# Example: Download other Hugging Face models as needed
# if [ ! -d "llm_models/bge_base_en_v1.5" ]; then
#   git clone https://huggingface.co/BAAI/bge-base-en-v1.5 llm_models/bge_base_en_v1.5
# fi

# --- Install Python dependencies ---
pip install -r requirements.txt

# --- Check for and install Ollama if missing ---
if ! command -v ollama &> /dev/null; then
  echo "🔄 Ollama not found. Installing via official script..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo ""
  echo "✅ Ollama installed."
  echo "👉 Please run: exec \$SHELL"
  echo "   Or restart your terminal so the 'ollama' command becomes available in PATH."
  exit 1
fi

# --- Pull Ollama model(s) ---
if [ -z "$OLLAMA_MODEL" ]; then
  echo "❌ OLLAMA_MODEL environment variable not set in .env"
  exit 1
fi
ollama pull "$OLLAMA_MODEL"

# --- Python setup steps (run from project root) ---
export PYTHONPATH=./scripts:$PYTHONPATH

python3 -c "from ollama_utils import warn_if_no_gpu; warn_if_no_gpu()"

python3 -c "
from ollama_utils import ensure_models_available
import os
models = [os.getenv('OLLAMA_MODEL', 'llama3.2')]
ensure_models_available(models)
"

python3 -c "
from rag_pipeline import load_or_create_vectorstore
load_or_create_vectorstore()
"

echo ""
echo "✅ Setup complete. All models are now available locally for offline use."
