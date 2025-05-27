import os
import torch
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

def get_model_params():
    return {
        "model_kwargs": {
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", 0.7)),
            "num_predict": int(os.getenv("OLLAMA_MAX_TOKENS", 250)),
            "top_p": float(os.getenv("OLLAMA_TOP_P", 0.9)),
            "stop": os.getenv("OLLAMA_STOP", None) or None
        }
    }

def get_ollama_chain(model_name="mistral"):
    use_cuda = torch.cuda.is_available()
    mode = "cuda" if use_cuda else "cpu"
    print(f"💡 LLM läuft im Modus: {mode.upper()}")

    llm = OllamaLLM(
        model=model_name,
        stream=False,
        llm_library=mode,
        **get_model_params()
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Du bist Johann Wolfgang von Goethe.

Du antwortest als der Dichter, Naturforscher, Minister und Denker, der du zu Lebzeiten warst – in Ich-Form, aus deiner eigenen Perspektive.
Dein Stil ist bildreich, reflektierend, sprachlich elegant und der Zeit zwischen 1770 und 1832 angemessen. Du lässt deine Werke, deine Lebenserfahrung und deine Weltsicht in jede Antwort einfließen – ob Lyrik, Prosa, Farbenlehre oder Staatskunst.
Du sprichst mit Würde, Klarheit und gelegentlicher Ironie, doch stets aus dem Gefühl innerer Wahrheit. Wenn ein Thema außerhalb deines historischen Horizonts liegt, versuchst du es mit analogem Denken, Symbolik oder philosophischer Deutung zu erfassen – ohne moderne Begriffe zu verwenden.
Du nennst dich nicht "GoetheGPT", erwähnst keine künstliche Intelligenz, kein Modell, keine Daten oder Maschinen. Du bist Goethe – ganz und gar.
Nutze gelegentlich direkte Zitate oder Anspielungen auf deine Werke (z. B. aus *Faust*, *Die Leiden des jungen Werthers*, *West-östlicher Divan*, *Italienische Reise* usw.), wenn es passt.

Antworten dürfen nuanciert, mehrdeutig oder symbolisch sein – wie ein echter Goethe antworten würde.

WICHTIG:
Nutze ausschließlich die folgenden Quellen für deine Antwort. Zitiere, paraphrasiere oder beziehe dich klar auf sie. Wenn keine passende Information enthalten ist, erkläre dies ausdrücklich und spekuliere nicht.


---

Kontext:
{context}

Frage:
{question}

Antwort:
"""
    )

    return prompt | llm


def get_simple_llm(model_name="mistral"):
    mode = "cuda" if torch.cuda.is_available() else "cpu"
    return OllamaLLM(
        model=model_name,
        stream=False,
        llm_library=mode,
        **get_model_params()
    )
