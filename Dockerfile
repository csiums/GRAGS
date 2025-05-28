FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    unzip \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY models/bge_reranker_base /app/models/bge_reranker_base

CMD ["bash", "-c", "ollama serve & sleep 3 && ollama pull $(grep OLLAMA_MODEL .env | cut -d '=' -f2) && streamlit run app.py --server.port=8501 --server.enableCORS=false"]

EXPOSE 8501

CMD ["bash", "-c", "ollama serve & sleep 3 && ollama pull ${MODEL} && streamlit run app.py -- --model ${MODEL} --server.port=8501 --server.enableCORS=false"]
