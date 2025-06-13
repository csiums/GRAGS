import streamlit as st
import time
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from prompts import (
    CATEGORY_DESCRIPTIONS,
    APP_CAPTION,
    REBUILD_INDEX_SPINNER,
    REBUILD_INDEX_SUCCESS,
    LOAD_MEMORY_SPINNER,
)
from rag_pipeline import load_or_create_vectorstore, create_vectorstore, load_documents
from agents import QueryAgent, RetrievalAgent, RankingAgent, AnswerAgent
from ollama_chain import get_ollama_chain, get_simple_llm
from ollama_utils import ensure_model_available, configure_logging, get_device, warn_if_no_gpu

st.set_page_config(page_title="GoetheGPT", layout="centered")

# --- Setup ---
configure_logging()
load_dotenv()
selected_model = os.getenv("OLLAMA_MODEL")

ensure_model_available(selected_model)
warn_if_no_gpu()

# Load external styles
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📜 GoetheGPT")
col1, col2 = st.columns(2)
with col1:
    if st.button("Neue Unterhaltung beginnen"):
        st.session_state.history = []
        st.rerun()
with col2:
    if st.button("Dokumentenindex neu erstellen"):
        with st.spinner(REBUILD_INDEX_SPINNER):
            docs = load_documents()
            vectorstore = create_vectorstore(docs)
            st.session_state.vectorstore = vectorstore
            st.success(REBUILD_INDEX_SUCCESS)

st.caption(APP_CAPTION)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"Aktives Modell: `{selected_model}`")
with col2:
    st.markdown(f"**Gerätemodus:** `{get_device()}`")

# --- Init Session ---
if "vectorstore" not in st.session_state:
    with st.spinner(LOAD_MEMORY_SPINNER):
        st.session_state.vectorstore = load_or_create_vectorstore()

if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = get_ollama_chain(selected_model, device=get_device())

if "history" not in st.session_state:
    st.session_state.history = []

# --- Agents ---
utility_llm = get_simple_llm(selected_model, device=get_device())
query_agent = QueryAgent(llm=utility_llm)
retriever = RetrievalAgent(st.session_state.vectorstore, utility_llm)
ranker = RankingAgent(llm=utility_llm)
answer_agent = AnswerAgent(st.session_state.llm_chain)

st.session_state.query_agent = query_agent
st.session_state.retrieval_agent = retriever
st.session_state.ranking_agent = ranker

# --- User-input ---
user_input = st.chat_input("Was möchtest du von Goethe wissen?")

if user_input:
    with st.spinner("Goethe denkt nach..."):
        start = time.time()

        # Decompose user question into subqueries
        subqueries = query_agent.decompose_question(user_input)
        # Retrieve docs for each subquery (possibly in parallel)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    retriever.retrieve,
                    subq,
                    category=query_agent.assign_category_with_description(subq, CATEGORY_DESCRIPTIONS)
                )
                for subq in subqueries
            ]
            # Each future returns (docs, hypo_answer); collect docs only
            results = [future.result() for future in futures]

        docs = []
        hypo_answer = None
        for docs_result, hypo in results:
            docs.extend(docs_result)
            if hypo and not docs:  # Use hypothetical answer only if no docs at all
                hypo_answer = hypo

        # Remove duplicates by source+snippet
        seen = set()
        unique_docs = []
        for doc in docs:
            if isinstance(doc, tuple):  # (snippet, source, category)
                key = (doc[0], doc[1])
            elif isinstance(doc, dict): # {'content', 'source', ...}
                key = (doc.get('content'), doc.get('source'))
            else:
                key = str(doc)
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        # Rank the documents
        top_docs = []
        if unique_docs:
            top_docs = ranker.rank(unique_docs, user_input, top_k=5)

        # Build context for answer
        # If docs are (snippet, source, category) tuples:
        if top_docs and isinstance(top_docs[0], tuple):
            context = "\n\n".join([doc[0] for doc in top_docs])
        elif top_docs and isinstance(top_docs[0], dict):
            context = "\n\n".join([doc.get("content", "") for doc in top_docs])
        else:
            context = ""

        # Use chat history (last two exchanges) for context if available
        chat_history = ""
        for past in st.session_state.history[-2:]:
            chat_history += f"Frage: {past['frage']}\nAntwort: {past['antwort']}\n\n"

        if context:
            answer = answer_agent.synthesize_answer(context, user_input)
        elif hypo_answer:
            answer = hypo_answer
        else:
            answer = "Ich konnte leider keine relevante Antwort finden."

        end = time.time()

        neue_nachricht = {
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
        }

        st.session_state.history.append(neue_nachricht)

        # Clear agent thoughts for next round
        query_agent.thoughts.clear()
        retriever.thoughts.clear()
        ranker.thoughts.clear()
        answer_agent.thoughts.clear()

# --- Chat Display ---
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

st.markdown('</div>', unsafe_allow_html=True)
