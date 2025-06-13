import logging
import time
import os
import torch
from dotenv import load_dotenv
import streamlit as st

from prompts import CATEGORY_DESCRIPTIONS
from rag_pipeline import load_or_create_vectorstore, create_vectorstore, load_documents
from agents import QueryAgent, RetrievalAgent, RankingAgent, AnswerAgent
from ollama_chain import get_ollama_chain, get_simple_llm
from ollama_utils import ensure_model_available, get_device, warn_if_no_gpu

# --- Setup ---
load_dotenv()
# --- Human-readable logging setup ---
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
selected_model = os.getenv("OLLAMA_MODEL")

ensure_model_available(selected_model)
warn_if_no_gpu()

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("GoetheGPT")

col1, col2 = st.columns(2)
with col1:
    if st.button("Neue Unterhaltung beginnen"):
        st.session_state.history = []
        st.rerun()

with col2:
    if st.button("Dokumentenindex neu erstellen"):
        with st.spinner("Rebuild: Dokumente werden neu eingelesen..."):
            docs = load_documents()
            vectorstore = create_vectorstore(docs)
            st.session_state.vectorstore = vectorstore
            st.success("Index wurde erfolgreich neu aufgebaut.")

st.caption(
    "GoetheGPT ist eine dokumentengestützte KI-Anwendung, die vollständig lokal und datenschutzfreundlich arbeitet. "
    "Sie gibt nachvollziehbare Antworten, indem sie Zitate aus den Quellen sichtbar macht und ihre Gedankengänge offenlegt. "
    "Im Unterschied zu kommerziellen Sprachassistenten bleibt alles transparent, offline und unter eigener Kontrolle."
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"Aktives Modell: `{selected_model}`")
with col2:
    st.markdown(f"**Gerätemodus:** `{get_device()}`")

# --- Session Setup ---
if "vectorstore" not in st.session_state:
    with st.spinner("Lade Goethes Gedächtnis..."):
        st.session_state.vectorstore = load_or_create_vectorstore()

if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = get_ollama_chain(selected_model, device=get_device())

if "history" not in st.session_state:
    st.session_state.history = []

utility_llm = get_simple_llm(selected_model, device=get_device())
query_agent = QueryAgent(llm=utility_llm)
retriever = RetrievalAgent(st.session_state.vectorstore, utility_llm)
ranker = RankingAgent(llm=utility_llm, device="cpu")
answer_agent = AnswerAgent(model_name=selected_model)

st.session_state.query_agent = query_agent
st.session_state.retrieval_agent = retriever
st.session_state.ranking_agent = ranker
st.session_state.answer_agent = answer_agent

# --- Eingabefeld ---
user_input = st.chat_input("Was möchtest du von Goethe wissen?")

# --- Query-Handling ---

def handle_query(user_input, query_agent, retriever, ranker, answer_agent, chat_history):
    start = time.time()
    logging.info(f"GoetheGPT denkt nach: '{user_input}'")

    try:
        subqueries = query_agent.decompose_query(user_input, history=chat_history)
        if not subqueries:
            logging.warning("Keine Teilfragen erkannt – benutze Originalfrage als Fallback.")
            subqueries = [user_input]

        docs = []
        for subq in subqueries:
            docs.extend(retriever.retrieve(
                subq,
                category=query_agent.assign_category_with_description(subq, CATEGORY_DESCRIPTIONS)
            ))

        seen = set()
        unique_docs = []
        for doc in docs:
            if isinstance(doc, tuple):
                key = (doc[0], doc[1])
            elif isinstance(doc, dict):
                key = (doc.get('content'), doc.get('source'))
            else:
                key = str(doc)
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        logging.info(f"Insgesamt {len(unique_docs)} relevante Dokumente gefunden.")

        top_docs = []
        if unique_docs:
            top_docs = ranker.rerank(unique_docs, user_input, top_k=5)

        if top_docs:
            if isinstance(top_docs[0], tuple):
                logging.info(f"Die wichtigsten Dokumente werden an GoetheGPT zur Antwortgenerierung übergeben.")
                answer = answer_agent.generate(top_docs, user_input, history=chat_history)
            elif isinstance(top_docs[0], dict):
                logging.info(f"Die wichtigsten Dokumente werden an GoetheGPT zur Antwortgenerierung übergeben.")
                tupleized = [(doc.get("content", ""), doc.get("source", "unbekannt"), doc.get("category", "")) for doc in top_docs]
                answer = answer_agent.generate(tupleized, user_input, history=chat_history)
            else:
                answer = answer_agent.generate([], user_input, history=chat_history)
        else:
            logging.warning(f"Keine relevanten Dokumente gefunden – generiere trotzdem eine Antwort.")
            answer = answer_agent.generate([], user_input, history=chat_history)

        end = time.time()
        logging.info(f"Antwort wurde nach {end - start:.2f} Sekunden erzeugt.")
        return answer, top_docs, subqueries, end - start

    except Exception as e:
        logging.error(f"Fehler bei der Verarbeitung der Anfrage: {e}")
        return "Ein Fehler ist aufgetreten. Bitte versuche es später noch einmal.", [], [], 0

# --- Query-Handling ---
if user_input:
    with st.spinner("Goethe denkt nach..."):
        chat_history = ""
        for past in st.session_state.history[-2:]:
            frage = past.get("frage", "").strip()
            antwort = past.get("antwort", "").strip()
            if frage and antwort:
                chat_history += f"Frage: {frage}\nAntwort: {antwort}\n\n"

        answer, top_docs, subqueries, duration = handle_query(user_input, query_agent, retriever, ranker, answer_agent, chat_history)

        st.session_state.history.append({
            "frage": user_input,
            "antwort": answer,
            "teilfragen": subqueries,
            "quellen": top_docs,
            "gedanken": {
                "query": list(query_agent.thoughts),
                "retrieval": list(retriever.thoughts),
                "ranking": list(ranker.thoughts),
                "antwort": list(answer_agent.thoughts),
            },
            "dauer": f"Antwortzeit: {duration:.2f} Sekunden"
        })

        query_agent.thoughts.clear()
        retriever.thoughts.clear()
        ranker.thoughts.clear()
        answer_agent.thoughts.clear()
        torch.cuda.empty_cache()

# --- Chat-Verlauf anzeigen ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for item in st.session_state.history:
    st.markdown(f'<div class="user-bubble">Ich: \n {item["frage"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="goethe-bubble">GoetheGPT: \n {item["antwort"]}</div>', unsafe_allow_html=True)
    st.caption(item["dauer"])

    if item["quellen"]:
        with st.expander("Quellen & Zitate"):
            for doc in item["quellen"]:
                if isinstance(doc, tuple):
                    snippet, source, category = doc
                elif isinstance(doc, dict):
                    snippet = doc.get("content", "")
                    source = doc.get("source", "Unbekannt")
                    category = doc.get("category", "")
                else:
                    snippet, source, category = str(doc), "Unbekannt", ""
                st.markdown(f"**{source}** (_{category}_)")
                st.code(snippet[:500] + ("..." if len(snippet) > 500 else ""))

    with st.expander("Goethes Gedankenwelt"):
        def render_section(title, entries):
            if entries:
                st.markdown(f"### {title}")
                for entry in entries:
                    st.markdown(f"- {entry}")

        render_section("Zerlegung der Frage", item["gedanken"]["query"])
        render_section("Dokumentensuche", item["gedanken"]["retrieval"])
        render_section("Relevanzbewertung", item["gedanken"]["ranking"])
        render_section("Antwortgenerierung", item["gedanken"]["antwort"])

st.markdown('</div>', unsafe_allow_html=True)
