import streamlit as st
import time
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from rag_pipeline import load_or_create_vectorstore
from agents import QueryAgent, RetrievalAgent, RankingAgent, AnswerAgent
from ollama_chain import get_ollama_chain, get_simple_llm
from ollama_utils import ensure_model_available, configure_logging

# --- Setup ---
configure_logging()
load_dotenv()
selected_model = os.getenv("OLLAMA_MODEL")
ensure_model_available(selected_model)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="GoetheGPT", layout="centered")
st.markdown("""
    <style>
        .user-bubble {
            background-color: #1a73e8;
            color: white;
            padding: 0.5em 1em;
            border-radius: 1em;
            margin-bottom: 0.3em;
            max-width: 80%;
            align-self: flex-end;
        }
        .goethe-bubble {
            background-color: #444;
            color: white;
            padding: 0.5em 1em;
            border-radius: 1em;
            margin-bottom: 0.3em;
            max-width: 80%;
            align-self: flex-start;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 1em;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📜 GoetheGPT")
st.caption("GoetheGPT ist eine dokumentengestützte KI-Anwendung, die vollständig lokal und datenschutzfreundlich arbeitet. Sie gibt nachvollziehbare Antworten, indem sie Zitate aus den Quellen sichtbar macht und ihre Gedankengänge offenlegt. Im Unterschied zu kommerziellen Sprachassistenten bleibt alles transparent, offline und unter eigener Kontrolle.")
st.markdown(f"🧠 Aktives Modell: `{selected_model}`")

# --- Kategorie-Definitionen ---
CATEGORY_DESCRIPTIONS = {
    "Biographie": "Informationen zu Goethes Leben, persönliche Hintergründe, Reisen und biografische Ereignisse.",
    "Briefe": "Briefwechsel und persönliche Korrespondenz Goethes mit Freunden, Bekannten und bedeutenden Persönlichkeiten seiner Zeit.",
    "Weltwissen": "Goethes wissenschaftliche Erkenntnisse, philosophische Betrachtungen und seine Beschäftigung mit Natur, Farbenlehre und allgemeinem Wissen.",
    "Werke": "Literarische Werke Goethes, darunter Gedichte, Dramen, Romane und Essays, wie Faust, Werther, West-östlicher Divan und mehr.",
    "Werkdeutung": "Literaturwissenschaftliche Sekundärliteratur verschiedener Experten auf Goethes Werk."
}

# --- Init Session ---
if "vectorstore" not in st.session_state:
    with st.spinner("📚 Lade Goethes Gedächtnis..."):
        st.session_state.vectorstore = load_or_create_vectorstore()

if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = get_ollama_chain(selected_model)

if "history" not in st.session_state:
    st.session_state.history = []

# --- Agenten ---
utility_llm = get_simple_llm(selected_model)
query_agent = QueryAgent(llm=utility_llm)
retriever = RetrievalAgent(st.session_state.vectorstore, utility_llm)
ranker = RankingAgent(llm=utility_llm)
answer_agent = AnswerAgent(selected_model)

# --- Nutzerfrage unten eingeben ---

user_input = st.chat_input("Was möchtest du von Goethe wissen?")

if user_input:
    with st.spinner("🔍 Goethe denkt nach..."):
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
        answer = answer_agent.generate(top_docs, user_input)
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
            "dauer": f"⏱️ Antwortzeit: {end - start:.2f} Sekunden"
        }

        st.session_state.history.append(neue_nachricht)

        query_agent.thoughts.clear()
        retriever.thoughts.clear()
        ranker.thoughts.clear()
        answer_agent.thoughts.clear()

# --- Chatverlauf anzeigen ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for item in st.session_state.history:
    st.markdown(f'<div class="user-bubble">🙋 {item["frage"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="goethe-bubble">🧠 {item["antwort"]}</div>', unsafe_allow_html=True)
    st.caption(item["dauer"])

    if item["quellen"]:
        with st.expander("📄 Quellen & Zitate"):
            for snippet, source, category in item["quellen"]:
                st.markdown(f"**{source}** (_{category}_)")
                st.code(snippet[:500] + ("..." if len(snippet) > 500 else ""))

    with st.expander("🎨 Goethes Gedankenwelt"):
        st.markdown("##### 🧩 Der Weg der Frage")
        for thought in item["gedanken"]["query"]:
            st.markdown(f"> 💭 *\"{thought}\"*")

        st.markdown("##### 📚 Die Spurensuche in den Archiven")
        for thought in item["gedanken"]["retrieval"]:
            st.markdown(f"> 📄 *\"{thought}\"*")

        st.markdown("##### 🔎 Die Auswahl der Essenz")
        for thought in item["gedanken"]["ranking"]:
            st.markdown(f"> 📊 *\"{thought}\"*")

        st.markdown("##### ✍️ Die Form der Antwort")
        for thought in item["gedanken"]["antwort"]:
            st.markdown(f"> 🖋️ *\"{thought}\"*")

st.markdown('</div>', unsafe_allow_html=True)

