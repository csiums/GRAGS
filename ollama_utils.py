import os
import subprocess
import logging

try:
    import torch
except ImportError:
    torch = None

# --- Geräteerkennung ---
def get_device():
    device_env = os.getenv("DEVICE")
    if device_env:
        return device_env

    use_cuda_env = os.getenv("USE_CUDA", "true").lower() == "true"
    if torch and use_cuda_env and torch.cuda.is_available():
        return "cuda"
    elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def warn_if_no_gpu():
    if get_device() == "cpu":
        print("⚠️  Hinweis: Kein GPU erkannt – Anwendung läuft auf CPU. Für beste Leistung CUDA verwenden.")

# --- Logging-Konfiguration ---
def configure_logging():
    minimal = os.getenv("MINIMAL_LOGGING", "false").lower() == "true"
    if minimal:
        logging.getLogger().setLevel(logging.ERROR)
        for mod in ["torch", "transformers", "langchain", "sentence_transformers", "faiss", "streamlit"]:
            logging.getLogger(mod).setLevel(logging.CRITICAL)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Modellverwaltung für Ollama ---
def ensure_model_available(model_name):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        if model_name not in result.stdout:
            print(f"🔄 Lade Ollama-Modell '{model_name}'...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"✅ Modell '{model_name}' erfolgreich geladen.")
        else:
            print(f"✅ Modell '{model_name}' ist bereits verfügbar.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"❌ Konnte Modell '{model_name}' nicht über 'ollama' laden oder prüfen.\n"
            f"Möglicherweise keine Internetverbindung?\n\nTechnischer Fehler:\n{e}"
        )

def ensure_models_available(model_names):
    for model in model_names:
        ensure_model_available(model)

def list_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                name, size, *rest = parts
                models.append({
                    "name": name,
                    "size": size,
                    "modified": " ".join(rest)
                })
        return models
    except Exception as e:
        print(f"Fehler bei 'ollama list': {e}")
        return []

# --- Sonstige Hilfsfunktionen ---
def get_env_var(name, default=None, required=False):
    value = os.getenv(name, default)
    if required and value is None:
        raise RuntimeError(f"Environment variable '{name}' is required but not set.")
    return value

def get_absolute_path(relative_path):
    return os.path.abspath(relative_path)

def log_rag_action(action, details=""):
    logging.info(f"[RAG] {action}: {details}")

def update_progress_bar(progress, processed, total, description=""):
    progress.progress(processed / total, text=f"{description} ({processed}/{total})")

def get_default_embeddings_model(model_name="all-MiniLM-L6-v2"):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)
