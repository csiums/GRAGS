FROM python:3.10-slim

WORKDIR /app

# Systempakete für Ollama + Python-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    git \
    curl \
    unzip \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Ollama installieren
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python requirements installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Dateien kopieren
COPY . .

# Standardmodell
ENV MODEL=mistral:latest

# Port für Streamlit
EXPOSE 8501

# Start: Ollama-Server starten, Modell laden, dann App
CMD ["bash", "-c", "ollama serve & sleep 3 && ollama pull ${MODEL} && streamlit run app.py -- --model ${MODEL} --server.port=8501 --server.enableCORS=false"]

