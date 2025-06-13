import logging
import re
import traceback
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
)
from ollama_chain import get_ollama_chain

GOETHE_DEFAULT_HISTORY = (
    "Dies ist der Beginn eines Gesprächs über Johann Wolfgang von Goethe, "
    "sein Werk, seine Zeit und sein Leben. "
    "Bitte beantworte alle Fragen sachlich und im Geiste Goethes."
)

def log_retrieval_process(step, data=None):
    if data:
        logging.info(f"{step} - {len(data)} Dokumente gefunden.")
    else:
        logging.info(f"{step} - Keine Ergebnisse für diese Anfrage.")

class QueryAgent:
    def __init__(self, llm):
        self.llm = llm
        self.thoughts = []

    def decompose_query(self, question, history=""):
        # Goethe-Default für leeren Verlauf
        if not history:
            history = GOETHE_DEFAULT_HISTORY
        logging.info(f"Beginne mit der Zerlegung der Frage: '{question}'")
        base_prompt = SUBQUESTION_PROMPT.format(question=question)
        prompt = f"Vorheriger Gesprächsverlauf:\n{history}\n\n{base_prompt}"
        response = self.llm.invoke(prompt)
        subqueries = re.findall(r"\d+\.\s*(.+?)\s*(?=\n|$)", response)
        filtered_subqueries = [
            s.strip() for s in subqueries if "?" in s and (len(s.strip()) >= 8 or history)
        ][:3]
        self.thoughts.append(
            f"Teilfragen erkannt: {filtered_subqueries}" if filtered_subqueries else "Keine gültigen Teilfragen erkannt."
        )
        logging.info(f"Teilfragen zerlegt: {filtered_subqueries}")
        return filtered_subqueries

    def assign_category_with_description(self, subquery, category_descriptions):
        logging.info(f"Ordne Kategorie für Teilfrage '{subquery}' zu.")
        categories_prompt = "\n".join([f"- {cat}: {desc}" for cat, desc in category_descriptions.items()])
        prompt = CATEGORY_ASSIGNMENT_PROMPT.format(subquery=subquery, categories_prompt=categories_prompt)
        category = self.llm.invoke(prompt).strip()
        category = category if category in category_descriptions else None
        if category is None:
            self.thoughts.append(f"Konnte keine Kategorie zuordnen – verwende alle Kategorien für: '{subquery}'")
            logging.warning(f"Keine Kategorie zugeordnet für '{subquery}', alle Kategorien werden verwendet.")
        else:
            self.thoughts.append(f"Die Teilfrage '{subquery}' wurde der Kategorie '{category}' zugeordnet.")
            logging.info(f"Teilfrage '{subquery}' der Kategorie '{category}' zugeordnet.")
        return category

class RetrievalAgent:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.thoughts = []

    def expand_query(self, query):
        logging.info(f"Erweitere die Anfrage '{query}' mit verwandten Fragen.")
        prompt = EXPAND_QUERY_PROMPT.format(query=query)
        response = self.llm.invoke(prompt)
        expanded_queries = re.findall(r"-\s*(.+)", response)
        logging.info(f"Erweiterte Anfragen: {expanded_queries}")
        return expanded_queries

    def retrieve(self, query, k=10, category=None, sparse_weight=0.3):
        logging.info(f"Starte die Suche nach Dokumenten für die Anfrage: '{query}'")
        queries = [query] + self.expand_query(query)
        dense_docs = [
            (doc, 1.0)
            for q in queries
            for doc in retrieve_docs_with_sources(self.vectorstore, q, k=k, category=category)
        ]
        sparse_docs = retrieve_bm25_docs(query, category)
        combined = dense_docs + sparse_docs
        combined.sort(key=lambda x: x[1], reverse=True)
        unique_docs = deduplicate_docs(combined)
        if not unique_docs:
            self.thoughts.append("Initiale Suche ergab keine Treffer – generiere hypothetische Antwort zur Verbesserung.")
            prompt = HYDE_PROMPT.format(query=query)
            hypo_answer = self.llm.invoke(prompt).strip()
            unique_docs = retrieve_docs_with_sources(self.vectorstore, hypo_answer, k=k, category=category)
        self.thoughts.append(f"{len(unique_docs)} Dokumente gefunden für: '{query}' (Kategorie: {category})")
        logging.info(f"Dokumente für '{query}' gefunden: {len(unique_docs)}")
        return unique_docs

class RankingAgent:
    def __init__(self, llm, reranker_model_hub="BAAI/bge-reranker-base", reranker_model_path="llm_models/bge_reranker_base", device=None):
        self.reranker_model = load_cross_encoder_with_cache(reranker_model_hub, reranker_model_path, device=device)
        self.llm = llm
        self.thoughts = []

    def llm_style_score(self, doc_content, query):
        prompt = STYLE_SCORE_PROMPT.format(text=doc_content, text_prompt=query)
        try:
            score = int(self.llm.invoke(prompt).strip())
            score = max(1, min(score, 10))
            logging.info(f"Stilbewertung für Dokument: {score}")
        except Exception as e:
            self.thoughts.append(f"Fehler bei Stilbewertung: {e}")
            logging.error(f"Fehler bei der Stilbewertung des Dokuments: {e}")
            score = 5
        return score

    def rerank(self, docs, query, top_k=5, style_weight=0.2):
        logging.info("Starte die Rangfolge der Dokumente basierend auf Relevanz und Stil.")
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
            logging.info(f"Top {top_k} Dokumente ausgewählt nach Relevanz und Stilbewertung.")
            return top_results
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            traceback.print_exc()
            return docs[:top_k]

class AnswerAgent:
    def __init__(self, model_name=None, device=None):
        self.chain = get_ollama_chain(model_name=model_name, device=device)
        self.thoughts = []

    def generate(self, docs, question, category=None, history=""):
        if not history:
            history = GOETHE_DEFAULT_HISTORY
        def get_content(doc):
            if isinstance(doc, dict):
                return doc.get("content", ""), doc.get("source", "Unbekannt")
            elif isinstance(doc, (list, tuple)):
                return doc[0], doc[1] if len(doc) > 1 else "Unbekannt"
            return str(doc), "Unbekannt"

        context_snippets = "\n\n".join(
            [f"[Quelle: {source}]\n{content}" for content, source in [get_content(doc) for doc in docs]]
        )

        if not context_snippets.strip():
            self.thoughts.append("Kein Kontext verfügbar.")
            logging.warning("Kein Kontext für die Antwort gefunden.")

        style_prompt = get_random_style_prompt(category)
        self.thoughts.append(f"Verwendetes Stil-Prompt: {style_prompt}")
        logging.info(f"Stil-Prompt verwendet: {style_prompt}")

        prompt_data = {
            "context": f"""Vorheriger Gesprächsverlauf:
{history}

Neue Wissensquellen:
{context_snippets}

Stil-Information:
{style_prompt}
""",
            "question": question,
        }

        try:
            response = self.chain.invoke(prompt_data)
            self.thoughts.append("Antwort erfolgreich generiert.")
            logging.info(f"Antwort erfolgreich generiert für die Frage: '{question}'")
        except Exception as e:
            self.thoughts.append(f"Fehler bei der Antwortgenerierung: {e}")
            logging.error(f"Fehler bei der Antwortgenerierung: {e}")
            response = "Fehler bei der Antwortgenerierung."
        return response.strip()