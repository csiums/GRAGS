import os
import subprocess
import logging
import time
import requests
from sentence_transformers import CrossEncoder

try:
    import torch
except ImportError:
    torch = None

from dotenv import load_dotenv
load_dotenv()
HUMAN_READABLE_LOGS = os.getenv("HUMAN_READABLE_LOGS", "false").lower() == "true"

def setup_logging():
    if HUMAN_READABLE_LOGS:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s"
        )
        for noisy_logger in ["httpcore", "httpx", "urllib3"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(levelname)s] %(message)s"
        )
setup_logging()

# --- Geräteerkennung ---
def get_device():
    """Ermittelt das verwendete Gerät (CPU, CUDA, MPS)."""
    device_env = os.getenv("DEVICE")
    if device_env:
        return device_env
    if torch and torch.cuda.is_available():
        return "cuda"
    elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def warn_if_no_gpu():
    """Gibt eine Warnung aus, wenn keine GPU verfügbar ist."""
    if get_device() == "cpu":
        logging.warning("Kein GPU erkannt – Anwendung läuft auf CPU. Für beste Leistung CUDA verwenden.")

# --- Modellverwaltung für Ollama ---
def ensure_model_available(model_name):
    """Stellt sicher, dass das angegebene Ollama-Modell verfügbar ist."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        if model_name not in result.stdout:
            logging.info(f"Lade Ollama-Modell '{model_name}'...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            logging.info(f"Modell '{model_name}' erfolgreich geladen.")
        else:
            logging.info(f"Modell '{model_name}' bereits verfügbar.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Fehler beim Laden des Modells '{model_name}': {e}")

def ensure_models_available(model_names):
    """Stellt sicher, dass alle angegebenen Ollama-Modelle verfügbar sind."""
    for model in model_names:
        ensure_model_available(model)

def list_ollama_models():
    """Listet alle verfügbaren Ollama-Modelle auf."""
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
        logging.error(f"Fehler bei 'ollama list': {e}")
        return []

# --- Erweiterte HTTP-Logging für Ollama API ---
def call_ollama_api_with_logging(payload):
    """Führt eine API-Anfrage an Ollama durch und loggt Details."""
    url = "http://127.0.0.1:11434/api/generate"
    logging.debug(f"API Anfrage: {payload}")

    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        response.raise_for_status()
        end_time = time.time()
        if HUMAN_READABLE_LOGS:
            logging.info(f"Antwort von Ollama erhalten ({end_time - start_time:.2f} Sekunden).")
        else:
            logging.debug(f"Antwort erhalten: {response.status_code}, Zeit: {end_time - start_time:.2f}s")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Fehler bei der API-Anfrage: {e}")
        raise RuntimeError(f"Fehler bei der API-Anfrage: {e}")

# --- Retrieval Hilfsfunktionen ---
def retrieve_docs_with_sources(vectorstore, query, k=10, category=None):
    """Ruft Dokumente aus dem Vektorstore ab und annotiert sie mit Quelle/Kategorie."""
    results = vectorstore.similarity_search(query, k=k)
    if HUMAN_READABLE_LOGS:
        logging.info(f"Es wurden {len(results)} Dokumente für die Anfrage '{query}' gefunden.")
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unbekannt"),
            "category": doc.metadata.get("category", "Unbekannt")
        } for doc in results
        if not category or doc.metadata.get("category") == category
    ]

def retrieve_bm25_docs(query, category=None):
    """Optionaler Platzhalter für klassische BM25-Retrieval-Strategie."""
    return []

def deduplicate_docs(scored_docs):
    """Entfernt doppelte Dokumente basierend auf Inhalt und Quelle."""
    seen = set()
    unique = []
    for doc, score in scored_docs:
        key = (doc.get("content"), doc.get("source"))
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique

def get_env_var(name, default=None, required=False):
    """Holt den Wert einer Umgebungsvariable oder wirft einen Fehler, wenn sie erforderlich ist und nicht gesetzt wurde."""
    value = os.getenv(name, default)
    if required and value is None:
        raise RuntimeError(f"Umgebungsvariable '{name}' ist erforderlich, aber nicht gesetzt.")
    return value

def get_absolute_path(relative_path):
    """Gibt den absoluten Pfad zu einem relativen Pfad zurück."""
    return os.path.abspath(relative_path)

def log_rag_action(action, details=""):
    """Loggt Aktionen im Zusammenhang mit RAG (Retrieval-Augmented Generation)."""
    logging.info(f"{action}: {details}")

def update_progress_bar(progress, processed, total, description=""):
    """Aktualisiert die Fortschrittsanzeige im Streamlit-Frontend."""
    progress.progress(processed / total, text=f"{description} ({processed}/{total})")

def get_default_embeddings_model(model_name="all-MiniLM-L6-v2"):
    """Lädt das Standardmodell für Text-Embeddings von HuggingFace."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)

def load_cross_encoder_with_cache(hub_model_name, local_model_dir, device="cpu"):
    """
    Attempts to load a CrossEncoder model from HuggingFace Hub; if offline or download fails,
    attempts to load from a local cache directory.
    """
    try:
        logging.info(f"Versuche CrossEncoder von HuggingFace Hub zu laden: {hub_model_name}")
        model = CrossEncoder(hub_model_name, device=device)
        if not os.path.exists(local_model_dir):
            model.save(local_model_dir)
        return model
    except Exception as e_hub:
        logging.warning(f"Konnte Modell nicht von HuggingFace laden: {e_hub}")
        if os.path.exists(local_model_dir):
            try:
                logging.info(f"Versuche CrossEncoder aus lokalem Cache zu laden: {local_model_dir}")
                return CrossEncoder(local_model_dir, device=device)
            except Exception as e_local:
                raise RuntimeError(f"Lokaler Modell-Cache konnte nicht geladen werden: {e_local}")
        else:
            raise RuntimeError(
                f"Modell konnte weder von HuggingFace noch lokal geladen werden.\n"
                f"Fehler von HuggingFace: {e_hub}\n"
                f"Lokaler Cache '{local_model_dir}' existiert nicht."
            )
