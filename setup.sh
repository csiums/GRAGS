#!/bin/bash
set -e
set -x

# 1. Sudo Password verification and extending the timeout
if sudo -v; then
  while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
else
  echo "Sudo authentication failed."
  exit 1
fi

# 2. .env file check and export environment variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found."
  exit 1
fi

# 3. Install git-lfs if not already installed (cross-distro support)
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

# 4. Python 3.10 setup
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

# 5. Set up virtual environment if not already created
if [ ! -d ".venv" ]; then
  $PYTHON_BIN -m venv .venv
fi
source .venv/bin/activate

# 6. Update pip
pip install --upgrade pip || { echo 'Pip upgrade failed'; exit 1; }

# 7. Install dependencies excluding Faiss GPU
grep -vE '^faiss-(cpu|gpu)' requirements.txt > requirements_nofaiss.txt
pip install -r requirements_nofaiss.txt || { echo 'Requirements installation failed'; exit 1; }
rm requirements_nofaiss.txt

# 8. Install Faiss for CPU (no GPU)
echo "Installing Faiss CPU..."
pip install faiss-cpu || { echo 'Faiss installation failed'; exit 1; }

# 9. Install Ollama if not found
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo "Ollama installed. Please restart your terminal or run 'exec \$SHELL'."
  exit 1
fi

# 10. Check if OLLAMA_MODEL is set in .env
if [ -z "$OLLAMA_MODEL" ]; then
  echo "OLLAMA_MODEL not set in .env!"
  exit 1
fi

# 11. Pull the Ollama model (downloaded and available offline)
ollama pull "$OLLAMA_MODEL"

# 12. Set PYTHONPATH for scripts
export PYTHONPATH=./scripts:$PYTHONPATH

# 13. Check for GPU warning
python -c "from scripts.ollama_utils import warn_if_no_gpu; warn_if_no_gpu()"

# 14. Ensure Ollama models are available (add weitere Modelle bei Bedarf)
python -c "
from scripts.ollama_utils import ensure_models_available
import os
models = [os.getenv('OLLAMA_MODEL', 'llama3.2')]
ensure_models_available(models)
"

# 15. Download and cache the HuggingFace CrossEncoder model using Python (Fallback auf git lfs clone bei Fehler)
CROSSENCODER_DIR="llm_models/bge_reranker_base"
if [ ! -f "${CROSSENCODER_DIR}/pytorch_model.bin" ] && [ ! -f "${CROSSENCODER_DIR}/model.safetensors" ]; then
  echo "Downloading CrossEncoder model BAAI/bge-reranker-base..."
  python -c "
from sentence_transformers import CrossEncoder
import os
local_dir = '${CROSSENCODER_DIR}'
try:
    model = CrossEncoder('BAAI/bge-reranker-base')
    model.save(local_dir)
except Exception as e:
    print('!! Fehler beim Speichern:', e)
    print('Versuche stattdessen Download per git lfs clone...')
    import shutil, subprocess
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    subprocess.run(['git', 'lfs', 'install'])
    subprocess.run(['git', 'clone', 'https://huggingface.co/BAAI/bge-reranker-base', local_dir])
"
else
  echo "CrossEncoder model already cached locally."
fi

# 16. Download and cache the SentenceTransformer Embeddings-Modell (Fallback auf git lfs clone bei Fehler)
EMBEDDING_DIR="llm_models/all-MiniLM-L6-v2"
if [ ! -f "${EMBEDDING_DIR}/pytorch_model.bin" ] && [ ! -f "${EMBEDDING_DIR}/model.safetensors" ]; then
  echo "Downloading Embedding model sentence-transformers/all-MiniLM-L6-v2..."
  python -c "
from sentence_transformers import SentenceTransformer
import os
local_dir = '${EMBEDDING_DIR}'
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.save(local_dir)
except Exception as e:
    print('!! Fehler beim Speichern:', e)
    print('Versuche stattdessen Download per git lfs clone...')
    import shutil, subprocess
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    subprocess.run(['git', 'lfs', 'install'])
    subprocess.run(['git', 'clone', 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2', local_dir])
"
else
  echo "Embeddings model already cached locally."
fi

# 17. Create or load the vectorstore (das legt den lokalen FAISS-Cache an)
python -c "
from scripts.rag_pipeline import load_or_create_vectorstore
load_or_create_vectorstore()
"

echo ""
echo "Setup completed. All models and embeddings are now available locally and offline."