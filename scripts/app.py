import streamlit as st
import time
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from rag_pipeline import load_or_create_vectorstore, create_vectorstore, load_documents
from agents import QueryAgent, RetrievalAgent, RankingAgent, AnswerAgent
from ollama_chain import get_ollama_chain, get_simple_llm
from ollama_utils import ensure_model_available, configure_logging, get_device, warn_if_no_gpu

# --- Setup ---
configure_logging()
load_dotenv()
selected_model = os.getenv("OLLAMA_MODEL")

# Ensure that the model is available locally
ensure_model_available(selected_model)

# Warn the user if no GPU is available
warn_if_no_gpu()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="GoetheGPT", layout="centered")

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

# --- Categories & Descriptions ---
CATEGORY_DESCRIPTIONS = {
    "Biographie": "Informationen zu Goethes Leben, persönliche Hintergründe, Reisen und biografische Ereignisse.",
    "Briefe": "Briefwechsel und persönliche Korrespondenz Goethes mit Freunden, Bekannten und bedeutenden Persönlichkeiten seiner Zeit.",
    "Weltwissen": "Goethes wissenschaftliche Erkenntnisse, philosophische Betrachtungen und seine Beschäftigung mit Natur, Farbenlehre und allgemeinem Wissen.",
    "Werke": "Literarische Werke Goethes, darunter Gedichte, Dramen, Romane und Essays, wie Faust, Werther, West-östlicher Divan und mehr.",
    "Werkdeutung": "Literaturwissenschaftliche Sekundärliteratur verschiedener Experten auf Goethes Werk."
}

# --- Init Session ---
if "vectorstore" not in st.session_state:
    with st.spinner("Lade Goethes Gedächtnis..."):
        st.session_state.vectorstore = load_or_create_vectorstore()

if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = get_ollama_chain(selected_model, device=get_device())

if "history" not in st.session_state:
    st.session_state.history = []

# --- Agents ---
utility_llm = get_simple_llm(selected_model, device=get_device())
query_agent = QueryAgent(llm=utility_llm)
retriever = RetrievalAgent(st.session_state.vectorstore, utility_llm)
ranker = RankingAgent(llm=st.session_state.llm_chain)
answer_agent = AnswerAgent(selected_model, device=get_device())

st.session_state.query_agent = query_agent
st.session_state.retrieval_agent = retriever
st.session_state.ranking_agent = ranker

# --- User-input ---
user_input = st.chat_input("Was möchtest du von Goethe wissen?")

if user_input:
    with st.spinner("Goethe denkt nach..."):
        start = time.time()

        subqueries = query_agent.decompose_query(user_input)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    retriever.retrieve,
                    subq,
                    category=query_agent.assign_category_with_description(subq, CATEGORY_DESCRIPTIONS)
                )
                for subq in subqueries
            ]
            docs = [doc for future in futures for doc in future.result()]

        unique_docs = list({doc[0]: doc for doc in docs}.values())
        top_docs = ranker.rerank(unique_docs, user_input) if unique_docs else []

        chat_history = ""
        for past in st.session_state.history[-2:]:
            chat_history += f"Frage: {past['frage']}\nAntwort: {past['antwort']}\n\n"

        answer = answer_agent.generate(top_docs, user_input, history=chat_history)
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
                "antwort": list(answer_agent.thoughts)
            },
            "dauer": f"Antwortzeit: {end - start:.2f} Sekunden"
        }

        st.session_state.history.append(neue_nachricht)

        query_agent.thoughts.clear()
        retriever.thoughts.clear()
        ranker.thoughts.clear()
        answer_agent.thoughts.clear()

# --- Chat ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for item in st.session_state.history:
    st.markdown(f'<div class="user-bubble">Ich: \n {item["frage"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="goethe-bubble">GoetheGPT: \n {item["antwort"]}</div>', unsafe_allow_html=True)
    st.caption(item["dauer"])

    if item["quellen"]:
        with st.expander("Quellen & Zitate"):
            for snippet, source, category in item["quellen"]:
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