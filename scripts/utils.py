import subprocess
import os
import logging

try:
    import torch
except ImportError:
    torch = None


# --- Device Management ---
def get_device():
    """
    Detect the best device for computation based on environment variables and hardware availability.
    Priority: DEVICE env > CUDA > MPS (Apple Silicon) > CPU.
    """
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
    """
    Print a warning if no GPU is detected and computations will run on the CPU.
    """
    if get_device() == "cpu":
        print("⚠️ Warning: No GPU detected. Running on CPU. For best performance, use a machine with CUDA.")


# --- Logging Configuration ---
def configure_logging():
    """
    Configure logging levels based on environment variables.
    Minimal logging reduces verbosity for production environments.
    """
    minimal = os.getenv("MINIMAL_LOGGING", "false").lower() == "true"
    if minimal:
        logging.getLogger().setLevel(logging.ERROR)
        for mod in ["torch", "transformers", "langchain", "sentence_transformers", "faiss", "streamlit"]:
            logging.getLogger(mod).setLevel(logging.CRITICAL)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# --- Model Management ---
def ensure_model_available(model_name):
    """
    Ensure the specified Ollama model is installed locally. If not, attempt to download it.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        installed_models = [line.split()[0] for line in result.stdout.strip().splitlines()[1:]]

        if model_name in installed_models:
            print(f"Modell '{model_name}' ist lokal verfügbar.")
        else:
            print(f"Modell '{model_name}' nicht lokal installiert.")
            print("Versuche, Modell zu laden (nur wenn online)...")
            try:
                subprocess.run(["ollama", "pull", model_name], check=True)
                print(f"Modell '{model_name}' erfolgreich geladen.")
            except subprocess.CalledProcessError as pull_err:
                raise RuntimeError(
                    f"Modell konnte nicht geladen werden.\n"
                    f"Möglicherweise bist du offline.\n"
                    f"Bitte führe diesen Befehl manuell aus, wenn du wieder online bist:\n"
                    f"    ollama pull {model_name}\n\n"
                    f"Technischer Fehler:\n{pull_err}"
                )
    except subprocess.CalledProcessError as list_err:
        raise RuntimeError(f"❌ Fehler bei 'ollama list': {list_err}")


def list_ollama_models():
    """
    List all locally available Ollama models with their metadata.
    """
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


def ensure_models_available(model_names):
    """
    Ensure a list of Ollama models are available locally.
    """
    for model in model_names:
        ensure_model_available(model)


# --- Utility Functions ---
def get_env_var(name, default=None, required=False):
    """
    Retrieve an environment variable with optional default and required flag.
    """
    value = os.getenv(name, default)
    if required and value is None:
        raise RuntimeError(f"Environment variable '{name}' is required but not set.")
    return value


def get_absolute_path(relative_path):
    """
    Get the absolute path for a given relative path.
    """
    return os.path.abspath(relative_path)


def log_rag_action(action, details=""):
    """
    Log RAG pipeline actions in a standardized format.
    """
    logging.info(f"[RAG] {action}: {details}")


def update_progress_bar(progress, processed, total, description=""):
    """
    Update a Streamlit progress bar with percentage and description.
    """
    progress.progress(processed / total, text=f"{description} ({processed}/{total})")


def get_default_embeddings_model(model_name="all-MiniLM-L6-v2"):
    """
    Load the default embeddings model for the vectorstore.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)