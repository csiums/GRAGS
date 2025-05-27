import re
from rag_pipeline import retrieve_docs_with_sources, load_documents
from ollama_chain import get_ollama_chain
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simple BM25-like TF-IDF-based sparse retriever (local fallback)
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


all_docs = load_documents()
documents_corpus = [
    (doc.page_content, doc.metadata.get("source", "?"), doc.metadata.get("category", None))
    for doc in all_docs
]


class QueryAgent:
    def __init__(self, llm):
        self.llm = llm
        self.thoughts = []

    def decompose_query(self, question):
        prompt = f"""
Zerlege die folgende komplexe Frage in maximal drei sinnvolle Teilfragen.
Die Teilfragen sollen eigenständig verständlich sein, ein Fragezeichen enthalten und sich inhaltlich voneinander unterscheiden.
Gib nur die Fragen an – nummeriert mit 1., 2., 3.

Frage: {question}

Teilfragen:
1."""
        response = self.llm.invoke(prompt)
        subqueries = re.findall(r"\d+\.\s*(.+?)\s*(?=\n|$)", response)
        subqueries = [s.strip() for s in subqueries if "?" in s][:3]
        self.thoughts.append(f"Teilfragen erkannt: {subqueries}" if subqueries else "⚠️ Keine gültigen Teilfragen erkannt.")
        return subqueries

    def assign_category_with_description(self, subquery, category_descriptions):
        categories_prompt = "\n".join([
            f"- {cat}: {desc}" for cat, desc in category_descriptions.items()
        ])

        prompt = f"""
Ordne folgende Teilfrage der passendsten Kategorie zu – auch wenn sie nur indirekt passt.
Wenn du unsicher bist, wähle diejenige Kategorie, die am ehesten zutrifft.

Teilfrage: "{subquery}"

Kategorien:
{categories_prompt}

Zugeordnete Kategorie (nur eine):"""

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
        prompt = f"""
Nenne alternative Formulierungen oder Synonyme für folgende Frage, um mehr relevante Dokumente zu finden.
Gib maximal 3 Varianten als Liste zurück.

Frage: "{query}"

Varianten:
-"""
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
            hyde_prompt = f"""Formuliere eine mögliche, kurze Antwort auf folgende Frage:

Frage: {query}

Antwort (nicht mehr als drei Sätze):"""
            hypo_answer = self.llm.invoke(hyde_prompt).strip()
            unique_docs = retrieve_docs_with_sources(self.vectorstore, hypo_answer, k=k, category=category)

        self.thoughts.append(f"{len(unique_docs)} Dokumente gefunden für: '{query}' (Kategorie: {category})")
        return unique_docs


class RankingAgent:
    def __init__(self, llm):
        self.reranker_model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        self.llm = llm
        self.thoughts = []

    def llm_style_score(self, doc, query):
        prompt = f"""Bewerte den Stil des folgenden Textes auf einer Skala von 1 (nicht goethehaft) bis 10 (sehr goethehaft).

Text:
\"\"\"{doc}\"\"\"

Frage:
{query}

Antwort (nur eine Zahl von 1 bis 10):"""
        try:
            score = int(self.llm.invoke(prompt).strip().split()[0])
            score = max(1, min(score, 10))
        except Exception:
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

        self.thoughts.append(f"{top_k} Dokumente ausgewählt nach kombinierter Relevanz+Stilwertung.")
        return top_results

class AnswerAgent:
    def __init__(self, model_name="mistral"):
        self.chain = get_ollama_chain(model_name)
        self.thoughts = []

    def get_prompt_for_category(self, category):
        prompts = {
            "Werke": "Antworte im literarischen Stil Goethes, nutze Metaphern und Anspielungen auf seine Werke wie Faust, Werther oder West-östlicher Divan.",
            "Werkdeutung": "Antworte aus der Perspektive Goethes, als erklärtest du deine eigenen Werke auf Basis zeitgenössischer Gedankenwelt und der Quellen.",
            "Weltwissen": "Antworte als Goethe, Naturdenker und Wissenschaftler, mit Anklängen an Farbenlehre, Naturphilosophie und klassischem Denken.",
            "Biographie": "Antworte in Ich-Form, wie aus deiner Autobiographie *Dichtung und Wahrheit*, mit reflektierendem Ton.",
            "Briefe": "Antworte im Stil eines persönlichen Briefes, wie an Charlotte von Stein oder Schiller, mit Emotion und Anmut."
        }
        return prompts.get(category, "Antworte im Stil Goethes, mit Tiefe, Anspielungen und literarischer Eleganz.")

    def generate(self, docs, question, category=None, history=""):
        context_snippets = "\n\n".join(
            [f"[Quelle: {source}]\n{content}" for content, source, *_ in docs]
        )

        prompt_instruction = self.get_prompt_for_category(category)

        prompt_data = {
            "context": f"""{prompt_instruction}

Vorheriger Gesprächsverlauf:
{history}

Neue Wissensquellen:
{context_snippets}
""",
            "question": question
        }

        response = ""
        for chunk in self.chain.stream(prompt_data):
            response += chunk

        return response.strip()
