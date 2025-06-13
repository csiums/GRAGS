import streamlit as st
import time
import os
import torch
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from prompts import CATEGORY_DESCRIPTIONS
from rag_pipeline import load_or_create_vectorstore, create_vectorstore, load_documents
from agents import QueryAgent, RetrievalAgent, RankingAgent, AnswerAgent
from ollama_chain import get_ollama_chain, get_simple_llm
from ollama_utils import ensure_model_available, configure_logging, get_device, warn_if_no_gpu

# --- Setup ---
configure_logging()
load_dotenv()
selected_model = os.getenv("OLLAMA_MODEL")

ensure_model_available(selected_model)
warn_if_no_gpu()

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📜 GoetheGPT")

# Buttons zum Neustarten oder Index neu erstellen
col1, col2 = st.columns(2)
with col1:
    if st.button("Neue Unterhaltung beginnen"):
        st.session_state.history = []
        st.experimental_rerun()

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

# Agenten initialisieren
utility_llm = get_simple_llm(selected_model, device=get_device())
query_agent = QueryAgent(llm=utility_llm)
retriever = RetrievalAgent(st.session_state.vectorstore, utility_llm)
ranker = RankingAgent(llm=utility_llm, device="cpu")
answer_agent = AnswerAgent(model_name=selected_model)

st.session_state.query_agent = query_agent
st.session_state.retrieval_agent = retriever
st.session_state.ranking_agent = ranker
st.session_state.answer_agent = answer_agent

# Debug-Log aktivieren (Checkbox über Chatfenster)
show_debug = st.checkbox("Live Debug anzeigen (über Chat)", value=False)

# --- Eingabefeld ---
user_input = st.chat_input("Was möchtest du von Goethe wissen?")

if user_input:
    with st.spinner("Goethe denkt nach..."):
        start = time.time()

        # Chat History als String (letzte 2 Runden)
        chat_history = ""
        for past in st.session_state.history[-2:]:
            frage = past.get("frage", "").strip()
            antwort = past.get("antwort", "").strip()
            if frage and antwort:
                chat_history += f"Frage: {frage}\nAntwort: {antwort}\n\n"

        # Teilfragen zerlegen mit Kontext
        subqueries = query_agent.decompose_query(user_input, history=chat_history)

        # Parallel Retrieval pro Teilfrage mit Kategorie
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    retriever.retrieve,
                    subq,
                    category=query_agent.assign_category_with_description(subq, CATEGORY_DESCRIPTIONS)
                )
                for subq in subqueries
            ]
            results = [future.result() for future in futures]

        docs = []
        for docs_result in results:
            docs.extend(docs_result)

        # Duplikate filtern
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

        # Ranking nur, wenn docs da sind
        top_docs = []
        if unique_docs:
            top_docs = ranker.rerank(unique_docs, user_input, top_k=5)

        # Antwort generieren, Chat-History mitgeben
        if top_docs and isinstance(top_docs[0], tuple):
            answer = answer_agent.generate(top_docs, user_input, history=chat_history)
        elif top_docs and isinstance(top_docs[0], dict):
            tupleized = [(doc.get("content", ""), doc.get("source", "unbekannt"), doc.get("category", "")) for doc in top_docs]
            answer = answer_agent.generate(tupleized, user_input, history=chat_history)
        else:
            answer = "Ich konnte leider keine relevante Antwort finden."

        end = time.time()

        # Verlauf speichern inkl. Gedanken
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
            "dauer": f"Antwortzeit: {end - start:.2f} Sekunden"
        })

        # Gedanken leeren und CUDA Cache säubern
        query_agent.thoughts.clear()
        retriever.thoughts.clear()
        ranker.thoughts.clear()
        answer_agent.thoughts.clear()
        torch.cuda.empty_cache()

# --- Debugbereich über dem Chat ---
if show_debug:
    with st.expander("🔍 Live Debug (Agenten-Gedanken)", expanded=True):
        for idx, item in enumerate(st.session_state.history[::-1]):  # neueste zuerst
            with st.expander(f"Chat #{len(st.session_state.history)-idx} Debug", expanded=(idx == 0)):
                st.markdown("**Zerlegung der Frage (QueryAgent):**")
                for thought in item["gedanken"]["query"]:
                    st.markdown(f"- {thought}")

                st.markdown("**Dokumentensuche (RetrievalAgent):**")
                for thought in item["gedanken"]["retrieval"]:
                    st.markdown(f"- {thought}")

                st.markdown("**Ranking (RankingAgent):**")
                for thought in item["gedanken"]["ranking"]:
                    st.markdown(f"- {thought}")

                st.markdown("**Antwortgenerierung (AnswerAgent):**")
                for thought in item["gedanken"]["antwort"]:
                    st.markdown(f"- {thought}")

# --- Chat Verlauf ---
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
