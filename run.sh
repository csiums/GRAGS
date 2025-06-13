#!/bin/bash

echo "Starting GoetheGPT application..."

if [ ! -f .env ]; then
    echo ".env file not found. Please create one with the required environment variables."
    exit 1
fi

export $(grep -v '^#' .env | xargs)

python3 -c "from scripts.ollama_utils import warn_if_no_gpu; warn_if_no_gpu()"

streamlit run scripts/app.py
