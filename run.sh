#!/bin/bash

# --- Application Run Script ---
echo "Starting GoetheGPT application..."

# Ensure environment is set up
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create one with the required environment variables."
    exit 1
fi

# Load environment variables from .env
export $(grep -v '^#' .env | xargs)

# Warn if no GPU is available
python3 -c "from ollama_utils import warn_if_no_gpu; warn_if_no_gpu()"

# Start the Streamlit app
streamlit run app.py