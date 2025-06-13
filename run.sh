#!/bin/bash
set -e

echo "Starte GoetheGPT Anwendung..."

# .venv aktivieren, falls vorhanden
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "Warnung: Virtuelle Umgebung (.venv) nicht gefunden. Es wird empfohlen, Setup zuerst auszuführen."
fi

# .env prüfen und exportieren
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env Datei nicht gefunden. Bitte erstellen."
  exit 1
fi

# Warnung falls keine GPU verfügbar (optional)
python3 -c "from scripts.ollama_utils import warn_if_no_gpu; warn_if_no_gpu()"

# Streamlit App starten
streamlit run scripts/app.py
