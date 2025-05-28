import subprocess
import json
import subprocess
import os
import logging
import torch


def ensure_model_available(model_name):
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
    

def configure_logging():
    minimal = os.getenv("MINIMAL_LOGGING", "false").lower() == "true"
    if minimal:
        logging.getLogger().setLevel(logging.ERROR)
        for mod in ["torch", "transformers", "langchain", "sentence_transformers", "faiss", "streamlit"]:
            logging.getLogger(mod).setLevel(logging.CRITICAL)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


try:
    import torch
except ImportError:
    torch = None

def get_device():
    use_cuda_env = os.getenv("USE_CUDA", "true").lower() == "true"
    if torch and use_cuda_env and torch.cuda.is_available():
        return "cuda"
    return "cpu"

