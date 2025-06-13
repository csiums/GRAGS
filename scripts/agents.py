import logging
import re
import traceback
import os
from sentence_transformers import CrossEncoder
from ollama_utils import (
    retrieve_docs_with_sources,
    retrieve_bm25_docs,
    deduplicate_docs,
    load_cross_encoder_with_cache,
)
from prompts import (
    SUBQUESTION_PROMPT,
    CATEGORY_ASSIGNMENT_PROMPT,
    EXPAND_QUERY_PROMPT,
    HYDE_PROMPT,
    STYLE_SCORE_PROMPT,
    get_random_style_prompt,
    GOETHE_DEFAULT_HISTORY,
    CLUELESS_PROMPT,
    OLLAMA_SYSTEM_PROMPT,
)
from ollama_chain import get_ollama_chain

from dotenv import load_dotenv
load_dotenv()
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

def log_retrieval_process(step, data=None):
    if data:
        logging.info(f"{step} - {len(data)} Dokumente gefunden.")
    else:
        logging.info(f"{step} - Keine Ergebnisse für diese Anfrage.")

def build_chat_context(history, user_question, max_turns=4):
    """
    Baut einen Chat-Verlauf als String für den Prompt auf.
    history: Liste von {"frage": ..., "antwort": ...}
    """
    chat_context = ""
    if isinstance(history, list):
        for turn in history[-max_turns:]:
            frage = turn.get("frage", "").strip()
            antwort = turn.get("antwort", "").strip()
            if frage:
                chat_context += f"User: {frage}\n"
            if antwort:
                chat_context += f"GoetheGPT: {antwort}\n"
        chat_context += f"User: {user_question}\n"
    else:
        chat_context = f"{history}\nUser: {user_question}\n"
    return chat_context

class QueryAgent:
    def __init__(self, llm):
        self.llm = llm
        self.thoughts = []

    def decompose_query(self, question, history=""):
        chat_context = build_chat_context(history, question)
        logging.info(f"Zerlege die Frage: '{question}' in Teilfragen.")
        base_prompt = SUBQUESTION_PROMPT.format(question=question)
        prompt = f"{chat_context}\n\n{base_prompt}"
        response = self.llm.invoke(prompt)
        subqueries = re.findall(r"\d+\.\s*(.+?)\s*(?=\n|$)", response)
        filtered_subqueries = [
            s.strip() for s in subqueries if "?" in s and (len(s.strip()) >= 8 or chat_context)
        ][:3]
        self.thoughts.append(
            f"Erkannte Teilfragen: {filtered_subqueries}" if filtered_subqueries else "Keine gültigen Teilfragen erkannt."
        )
        logging.info(f"Erkannte Teilfragen: {filtered_subqueries}")
        return filtered_subqueries

    def assign_category_with_description(self, subquery, category_descriptions):
        logging.info(f"Ordne Kategorie für Teilfrage '{subquery}' zu.")
        categories_prompt = "\n".join([f"- {cat}: {desc}" for cat, desc in category_descriptions.items()])
        prompt = CATEGORY_ASSIGNMENT_PROMPT.format(subquery=subquery, categories_prompt=categories_prompt)
        category = self.llm.invoke(prompt).strip()
        valid_cats = {k.lower(): k for k in category_descriptions}
        cat_key = category.lower()
        if cat_key in valid_cats:
            category = valid_cats[cat_key]
        else:
            self.thoughts.append(f"Konnte keine Kategorie zuordnen – verwende alle Kategorien für: '{subquery}'")
            logging.warning(f"Keine Kategorie zugeordnet für '{subquery}', alle Kategorien werden verwendet.")
            category = None
        return category

class RetrievalAgent:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.thoughts = []

    def expand_query(self, query):
        logging.info(f"Erweitere die Anfrage '{query}' um verwandte Fragen.")
        prompt = EXPAND_QUERY_PROMPT.format(query=query)
        response = self.llm.invoke(prompt)
        expanded_queries = re.findall(r"-\s*(.+)", response)
        logging.info(f"Erweiterte Anfragen: {expanded_queries}")
        return expanded_queries

    def retrieve(self, query, k=10, category=None, sparse_weight=0.3, chat_history=None):
        logging.info(f"Suche relevante Dokumente für: '{query}'")
        queries = []
        if chat_history and isinstance(chat_history, list):
            last_user_questions = " ".join([turn["frage"] for turn in chat_history[-2:] if "frage" in turn])
            if last_user_questions.strip() and last_user_questions.strip() != query.strip():
                queries.append(f"{last_user_questions} {query}")
        queries += [query] + self.expand_query(query)
        dense_docs = [
            (doc, 1.0)
            for q in queries
            for doc in retrieve_docs_with_sources(self.vectorstore, q, k=k, category=category)
        ]
        sparse_docs = retrieve_bm25_docs(query, category)
        combined = dense_docs + sparse_docs
        combined.sort(key=lambda x: x[1], reverse=True)
        unique_docs = deduplicate_docs(combined)
        if not unique_docs and category is not None:
            logging.info("Keine Treffer – versuche alle Kategorien.")
            unique_docs = []
            for cat in [None, "Unbekannt"]:
                for q in queries:
                    unique_docs += [
                        (doc, 1.0)
                        for doc in retrieve_docs_with_sources(self.vectorstore, q, k=k, category=cat)
                    ]
            unique_docs = deduplicate_docs(unique_docs)
        if not unique_docs:
            self.thoughts.append("Keine Treffer – generiere hypothetische Antwort zur Verbesserung.")
            prompt = HYDE_PROMPT.format(query=query)
            hypo_answer = self.llm.invoke(prompt).strip()
            unique_docs = retrieve_docs_with_sources(self.vectorstore, hypo_answer, k=k, category=None)
            unique_docs = deduplicate_docs(unique_docs)
        if not unique_docs:
            unique_docs = retrieve_bm25_docs(query, None)
            unique_docs = deduplicate_docs(unique_docs)
        self.thoughts.append(f"{len(unique_docs)} Dokumente gefunden für: '{query}' (Kategorie: {category})")
        logging.info(f"Für '{query}' wurden {len(unique_docs)} Dokumente gefunden.")
        return unique_docs

class RankingAgent:
    def __init__(self, llm, reranker_model_hub="BAAI/bge-reranker-base", reranker_model_path="llm_models/bge_reranker_base", device=None):
        self.reranker_model = load_cross_encoder_with_cache(reranker_model_hub, reranker_model_path, device=device)
        self.llm = llm
        self.thoughts = []

    def llm_style_score(self, doc_content, query):
        prompt = STYLE_SCORE_PROMPT.format(text=doc_content, text_prompt=query)
        try:
            score_str = self.llm.invoke(prompt).strip()
            score = int(float(score_str.replace(",", ".").replace(" ", "").rstrip(".")))
            score = max(1, min(score, 10))
            logging.info(f"Stilbewertung: {score}")
        except Exception as e:
            self.thoughts.append(f"Fehler bei Stilbewertung: {e}")
            logging.error(f"Fehler bei der Stilbewertung: {e}")
            score = 5
        return score

    def rerank(self, docs, query, top_k=5, style_weight=0.2):
        logging.info("Bewerte die Dokumente nach Relevanz und Stil.")
        try:
            def get_content(doc):
                if isinstance(doc, dict):
                    return doc.get("content", "")
                elif isinstance(doc, (list, tuple)):
                    return doc[0]
                return str(doc)
            inputs = [[query, get_content(doc)] for doc in docs]
            logging.debug(f"CrossEncoder inputs: {inputs}")
            relevance_scores = self.reranker_model.predict(inputs)
            combined_scores = []
            for doc, rel_score in zip(docs, relevance_scores):
                style_score = self.llm_style_score(get_content(doc), query)
                combined = rel_score + style_weight * style_score
                combined_scores.append((doc, combined))
            top_docs = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
            top_results = [doc for doc, _ in top_docs]
            logging.info(f"Top {top_k} Dokumente ausgewählt.")
            return top_results
        except Exception as e:
            logging.error(f"Reranking fehlgeschlagen: {e}")
            traceback.print_exc()
            return docs[:top_k]

class AnswerAgent:
    def __init__(self, model_name=None, device=None):
        self.chain = get_ollama_chain(model_name=model_name, device=device)
        self.thoughts = []

    def generate(self, docs, question, category=None, history=None):
        if not history:
            chat_context = build_chat_context([], question)
        else:
            chat_context = build_chat_context(history, question)

        def get_content(doc):
            if isinstance(doc, dict):
                return doc.get("content", ""), doc.get("source", "Unbekannt")
            elif isinstance(doc, (list, tuple)):
                return doc[0], doc[1] if len(doc) > 1 else "Unbekannt"
            return str(doc), "Unbekannt"

        context_snippets = "\n\n".join(
            [f"[Quelle: {source}]\n{content}" for content, source in [get_content(doc) for doc in docs]]
        )
        style_prompt = get_random_style_prompt(category)
        self.thoughts.append(f"Stilhinweis: {style_prompt}")
        logging.info(f"Stilhinweis genutzt: {style_prompt}")

        prompt_data = {
            "chat_history": chat_context,
            "context": context_snippets if context_snippets.strip() else CLUELESS_PROMPT.format(question=question),
            "question": question,
        }

        try:
            response = self.chain.invoke(prompt_data)
            self.thoughts.append("Antwort erfolgreich generiert.")
            logging.info(f"Antwort erfolgreich generiert für: '{question}'")
        except Exception as e:
            self.thoughts.append(f"Fehler bei der Antwortgenerierung: {e}")
            logging.error(f"Fehler bei der Antwortgenerierung: {e}")
            response = "Fehler bei der Antwortgenerierung."
        return response.strip()

def process_user_question(user_question, agent, retrieval_agent, ranking_agent, answer_agent, history=None):
    subquestions = agent.decompose_query(user_question, history=history)
    if not subquestions:
        logging.warning("Keine Teilfragen erkannt – benutze Originalfrage als Fallback.")
        subquestions = [user_question]

    all_docs = []
    thoughts = []
    for sq in subquestions:
        docs = retrieval_agent.retrieve(sq, chat_history=history)
        all_docs.extend(docs)
        if hasattr(retrieval_agent, "thoughts"):
            thoughts.extend(retrieval_agent.thoughts)
            retrieval_agent.thoughts.clear()

    seen = set()
    deduped_docs = []
    for d in all_docs:
        content = d["content"] if isinstance(d, dict) else str(d)
        if content not in seen:
            deduped_docs.append(d)
            seen.add(content)

    if deduped_docs:
        top_docs = ranking_agent.rerank(deduped_docs, user_question)
    else:
        top_docs = []

    answer = answer_agent.generate(
        top_docs,
        user_question,
        history=history
    )

    return {
        "answer": answer,
        "thoughts": thoughts,
        "subquestions": subquestions,
        "docs_used": top_docs
    }