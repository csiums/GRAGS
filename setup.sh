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

# --- Check for and install Ollama if missing ---
if ! command -v ollama &> /dev/null; then
  echo "🔄 Ollama not found. Installing via official script..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo ""
  echo "✅ Ollama installed."
  echo "👉 Please run: exec \$SHELL"
  echo "   Or restart your terminal so the 'ollama' command becomes available in PATH."
fi

# --- Install Python dependencies ---
pip install -r requirements.txt

# --- Warn about device ---
python3 -c "from ollama_utils import warn_if_no_gpu; warn_if_no_gpu()"

# --- Ensure Ollama model is installed ---
python3 -c "
from ollama_utils import ensure_models_available
import os
models = [os.getenv('OLLAMA_MODEL_NAME', 'llama3.2')]
ensure_models_available(models)
"

# --- Build vectorstore cache if needed ---
python3 -c "
from rag_pipeline import load_or_create_vectorstore
load_or_create_vectorstore()
"

echo ""
echo "✅ Setup complete."
