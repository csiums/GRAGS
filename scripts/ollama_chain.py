import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from ollama_utils import get_device, ensure_model_available
from prompts import OLLAMA_SYSTEM_PROMPT

# --- Modellparameter aus Umgebungsvariablen ---
def get_model_params():
    def _float_env(var, default):
        try:
            return float(os.getenv(var, default))
        except Exception:
            return default

    def _int_env(var, default):
        try:
            return int(os.getenv(var, default))
        except Exception:
            return default

    return {
        "model_kwargs": {
            "temperature": _float_env("OLLAMA_TEMPERATURE", 0.7),
            "num_predict": _int_env("OLLAMA_MAX_TOKENS", 250),
            "top_p": _float_env("OLLAMA_TOP_P", 0.9),
            "stop": os.getenv("OLLAMA_STOP", None) or None
        }
    }

# --- LLM-Kette mit Goethe-Systemprompt ---
def get_ollama_chain(model_name=None, device=None):
    if model_name is None:
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    mode = device or get_device()
    print(f"LLM läuft im Modus: {mode.upper()}")

    llm = OllamaLLM(
        model=model_name,
        stream=False,
        llm_library=mode,
        **get_model_params()
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=OLLAMA_SYSTEM_PROMPT
    )

    return prompt | llm

# --- Einfaches LLM ohne Prompt-Template ---
def get_simple_llm(model_name=None, device=None):
    if model_name is None:
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    mode = device or get_device()

    return OllamaLLM(
        model=model_name,
        stream=False,
        llm_library=mode,
        **get_model_params()
    )

# --- Wrapper for Blocking LLM Calls ---
def run_llm(llm, prompt, context, question):
    result = llm.invoke({"context": context, "question": question})
    return result
