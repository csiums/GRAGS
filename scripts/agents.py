import re
from prompts import (
    SUBQUESTION_PROMPT,
    CATEGORY_ASSIGNMENT_PROMPT,
    EXPAND_QUERY_PROMPT,
    HYDE_PROMPT,
)

class QueryAgent:
    """
    Handles the decomposition of user questions and categorization of sub-questions.
    """

    def __init__(self, llm):
        self.llm = llm
        self.thoughts = []

    def decompose_question(self, question):
        """
        Decomposes a complex question into up to three sub-questions.
        Returns a list of sub-questions.
        """
        prompt = SUBQUESTION_PROMPT.format(question=question)
        response = self.llm.invoke(prompt)
        subqueries = re.findall(r"\d+\.\s*(.+?)\s*(?=\n|$)", response)
        subqueries = [s.strip() for s in subqueries if "?" in s][:3]
        if subqueries:
            self.thoughts.append(f"Teilfragen erkannt: {subqueries}")
        else:
            self.thoughts.append("Keine gültigen Teilfragen erkannt.")
        return subqueries

    def assign_category_with_description(self, subquery, category_descriptions):
        """
        Assigns a category to a sub-question using the provided category descriptions.
        Returns the category name or None if no category matched.
        """
        categories_prompt = "\n".join(
            [f"- {cat}: {desc}" for cat, desc in category_descriptions.items()]
        )
        prompt = CATEGORY_ASSIGNMENT_PROMPT.format(
            subquery=subquery, categories_prompt=categories_prompt
        )
        category = self.llm.invoke(prompt).strip()
        category = category if category in category_descriptions else None

        if category is None:
            self.thoughts.append(
                f"Konnte keine Kategorie zuordnen – verwende alle Kategorien für: '{subquery}'"
            )
            return None
        else:
            self.thoughts.append(
                f"Ich ordne die Teilfrage '{subquery}' der Kategorie '{category}' zu."
            )
            return category

class RetrievalAgent:
    """
    Handles query expansion, retrieval, and hypothetical answer generation.
    """

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.thoughts = []

    def expand_query(self, query):
        """
        Suggests alternative phrasings or synonyms for a query.
        Returns a list of variants.
        """
        prompt = EXPAND_QUERY_PROMPT.format(query=query)
        response = self.llm.invoke(prompt)
        return re.findall(r"-\s*(.+)", response)

    def retrieve(self, query, k=10, category=None, sparse_weight=0.3):
        """
        Retrieves documents relevant to the (expanded) query from the vectorstore.
        If no results, generates a hypothetical answer using the Hyde prompt.
        """
        queries = [query] + self.expand_query(query)
        dense_docs = []
        for q in queries:
            dense_docs.extend(
                [(doc, 1.0) for doc in retrieve_docs_with_sources(self.vectorstore, q, k=k, category=category)]
            )
        sparse_docs = retrieve_bm25_docs(query, category)
        combined = dense_docs + sparse_docs
        combined.sort(key=lambda x: x[1], reverse=True)
        unique_docs = deduplicate_docs(combined)

        if not unique_docs:
            self.thoughts.append("Initiale Suche ergab keine Treffer – generiere hypothetische Antwort zur Verbesserung.")
            hyde_prompt = HYDE_PROMPT.format(query=query)
            hypo_answer = self.llm.invoke(hyde_prompt).strip()
            return [], hypo_answer

        return unique_docs, None

class RankingAgent:
    """
    Handles ranking of retrieved documents using LLM relevance scoring.
    """

    def __init__(self, llm):
        self.llm = llm
        self.thoughts = []

    def rank(self, docs, question, top_k=5):
        """
        Ranks documents by relevance to the question using the LLM.

        Args:
            docs (list): List of document dicts with 'content' key.
            question (str): The user's query/question.
            top_k (int): Number of top documents to return.

        Returns:
            List of docs sorted by LLM-assessed relevance (highest first).
        """
        if not docs:
            return []

        scored_docs = []
        for i, doc in enumerate(docs):
            prompt = (
                f"Bewerte die Relevanz des folgenden Dokuments für die Frage auf einer Skala von 1 (irrelevant) bis 5 (sehr relevant):\n"
                f"Frage: {question}\n"
                f"Dokument:\n{doc['content']}\n"
                f"Relevanz (nur Zahl):"
            )
            try:
                score_str = self.llm.invoke(prompt).strip()
                score = int(score_str.split()[0])  # In case LLM returns "5 Sehr relevant" etc.
            except Exception:
                score = 1  # fallback: minimal relevance
            scored_docs.append((score, doc))
            self.thoughts.append(f"Doc {i}: Score {score}")

        # Sort by score descending, return top_k docs
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        ranked_docs = [doc for (score, doc) in scored_docs][:top_k]
        return ranked_docs

class AnswerAgent:
    """
    Handles answer synthesis using the LLM.
    """

    def __init__(self, llm):
        self.llm = llm
        self.thoughts = []

    def synthesize_answer(self, context, question):
        """
        Generates an answer using the provided context and question.
        """
        # The actual prompt is handled by the ollama_chain's prompt template.
        return self.llm.invoke({"context": context, "question": question})
