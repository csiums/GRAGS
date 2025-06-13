import os
import re
import random
from rag_pipeline import retrieve_docs_with_sources, load_documents
from ollama_chain import get_ollama_chain
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ollama_utils import get_device

from prompts import (
    SUBQUESTION_PROMPT,
    CATEGORY_ASSIGNMENT_PROMPT,
    EXPAND_QUERY_PROMPT,
    HYDE_PROMPT,
    STYLE_SCORE_PROMPT,
    CATEGORY_STYLE_PROMPTS
)

# --- Device Initialization ---
DEVICE = get_device()

# --- TF-IDF Setup ---
documents_corpus = []
vectorizer = TfidfVectorizer()

def retrieve_bm25_docs(query, category=None, top_n=5):
    if not documents_corpus:
        return []

    filtered = [doc for doc in documents_corpus if category is None or doc[2] == category]
    if not filtered:
        return []

    texts = [doc[0] for doc in filtered]
    tfidf_matrix = vectorizer.fit_transform(texts + [query])
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_indices = scores.argsort()[::-1][:top_n]
    return [(filtered[i], scores[i]) for i in top_indices if scores[i] > 0.1]

def deduplicate_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        key = (doc[0][0][:100], doc[0][1]) if isinstance(doc, tuple) and isinstance(doc[0], tuple) else (doc[0][:100], doc[1])
        if key not in seen:
            seen.add(key)
            unique.append(doc if not isinstance(doc, tuple) else doc[0])
    return unique

# --- Load Corpus at module level ---
all_docs = load_documents()
documents_corpus = [
    (doc.page_content, doc.metadata.get("source", "?"), doc.metadata.get("category", None))
    for doc in all_docs
]

# --- Style Prompt Variation ---
def get_random_style_prompt(category):
    base_prompt = CATEGORY_STYLE_PROMPTS.get(
        category,
        "Ich antworte als Johann Wolfgang von Goethe – mit bildreicher Sprache, innerer Wahrheit und Anklängen an meine Werke und Gedanken."
    )
    variations = [
        base_prompt,
        base_prompt + "\nAntworte diesmal besonders kurz und prägnant.",
        base_prompt + "\nNutze viele Metaphern und Bilder.",
        base_prompt + "\nFormuliere poetisch und mit Melancholie.",
    ]
    chosen = random.choice(variations)
    return chosen

# --- Agents ---

class QueryAgent:
    def __init__(self, llm):
        self.llm = llm
        self.thoughts = []

    def decompose_query(self, question, history=""):
        base_prompt = SUBQUESTION_PROMPT.format(question=question)
        if history:
            prompt = f"Vorheriger Gesprächsverlauf:\n{history}\n\n{base_prompt}"
        else:
            prompt = base_prompt

        response = self.llm.invoke(prompt)
        subqueries = re.findall(r"\d+\.\s*(.+?)\s*(?=\n|$)", response)

        filtered_subqueries = []
        for s in subqueries:
            s_clean = s.strip()
            if "?" in s_clean:
                if len(s_clean) >= 8:
                    filtered_subqueries.append(s_clean)
                elif len(s_clean) < 8 and history:
                    filtered_subqueries.append(s_clean)

        filtered_subqueries = filtered_subqueries[:3]

        self.thoughts.append(
            f"Teilfragen erkannt: {filtered_subqueries}"
            if filtered_subqueries else "⚠️ Keine gültigen Teilfragen erkannt."
        )
        return filtered_subqueries

    def assign_category_with_description(self, subquery, category_descriptions):
        categories_prompt = "\n".join([
            f"- {cat}: {desc}" for cat, desc in category_descriptions.items()
        ])
        prompt = CATEGORY_ASSIGNMENT_PROMPT.format(
            subquery=subquery,
            categories_prompt=categories_prompt
        )
        category = self.llm.invoke(prompt).strip()
        category = category if category in category_descriptions else None

        if category is None:
            self.thoughts.append(f"Konnte keine Kategorie zuordnen – verwende alle Kategorien für: '{subquery}'")
            return None
        else:
            self.thoughts.append(f"Ich ordne die Teilfrage '{subquery}' der Kategorie '{category}' zu.")
            return category

class RetrievalAgent:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.thoughts = []

    def expand_query(self, query):
        prompt = EXPAND_QUERY_PROMPT.format(query=query)
        response = self.llm.invoke(prompt)
        return re.findall(r"-\s*(.+)", response)

    def retrieve(self, query, k=10, category=None, sparse_weight=0.3):
        queries = [query] + self.expand_query(query)

        dense_docs = []
        for q in queries:
            dense_docs.extend([(doc, 1.0) for doc in retrieve_docs_with_sources(self.vectorstore, q, k=k, category=category)])

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
        return unique_docs

class RankingAgent:
    def __init__(self, llm, reranker_model_path="llm_models/bge_reranker_base", device=None):
        if device is None:
            device = "cpu"
        self.reranker_model = CrossEncoder(reranker_model_path, device=device)
        self.llm = llm
        self.thoughts = []
        self.device = device

    def llm_style_score(self, doc, query):
        prompt = STYLE_SCORE_PROMPT.format(text=doc, text_prompt=query)
        try:
            score = int(self.llm.invoke(prompt).strip().split()[0])
            score = max(1, min(score, 10))
        except Exception as e:
            self.thoughts.append(f"llm_style_score Fehler: {e}")
            score = 5
        return score

    def rerank(self, docs, query, top_k=5, style_weight=0.2):
        inputs = [[query, doc[0]] for doc in docs]
        relevance_scores = self.reranker_model.predict(inputs)

        combined_scores = []
        for doc, rel_score in zip(docs, relevance_scores):
            style_score = self.llm_style_score(doc[0], query)
            combined = rel_score + style_weight * style_score
            combined_scores.append((doc, combined))

        top_docs = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
        top_results = [doc for doc, _ in top_docs]

        self.thoughts.append(
            f"{top_k} Dokumente ausgewählt nach kombinierter Relevanz+Stilwertung (Device: {self.device})."
        )
        return top_results

class AnswerAgent:
    def __init__(self, model_name=None, device=None):
        if model_name is None:
            model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        if device is None:
            device = DEVICE
        self.chain = get_ollama_chain(model_name, device=device)
        self.thoughts = []
        self.device = device

    def generate(self, docs, question, category=None, history=""):
        context_snippets = "\n\n".join(
            [f"[Quelle: {source}]\n{content}" for content, source, *_ in docs]
        )

        if not context_snippets.strip():
            self.thoughts.append("⚠️ Kein Kontext vorhanden – LLM erhält leeren Wissensinput.")

        style_prompt = get_random_style_prompt(category)
        self.thoughts.append(f"Stil-Prompt ausgewählt: {style_prompt}")

        prompt_data = {
            "context": f"""Vorheriger Gesprächsverlauf:
{history}

Neue Wissensquellen:
{context_snippets}

Stil-Information:
{style_prompt}
""",
            "question": question
        }

        try:
            response = self.chain.invoke(prompt_data)
            self.thoughts.append("Antwort erfolgreich generiert (invoke).")
        except Exception as e:
            self.thoughts.append(f"❌ Fehler bei Antwortgenerierung: {e}")
            response = "Ich bin verstummt – mein Geist vermag keine Antwort zu formen."

        return response.strip()
