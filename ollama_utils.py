import subprocess
import json
import subprocess
import os
import logging

def ensure_model_available(model_name):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        installed_models = [line.split()[0] for line in result.stdout.strip().splitlines()[1:]]

        if model_name not in installed_models:
            print(f"🔍 Modell '{model_name}' nicht gefunden – wird jetzt geladen...")
            pull_result = subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"✅ Modell '{model_name}' erfolgreich geladen.")
        else:
            print(f"✅ Modell '{model_name}' ist bereits vorhanden.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler beim Prüfen oder Laden des Modells '{model_name}': {e}")


def list_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:  # erste Zeile ist Header
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

