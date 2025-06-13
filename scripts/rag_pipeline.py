import os
import xml.etree.ElementTree as ET
import pickle
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import logging

import dotenv
dotenv.load_dotenv()
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

# --- Konstanten für Speicherpfade ---
DOCS_PATH = "rag/docs"
CACHE_PATH = "rag/index"
METADATA_FILE = os.path.join(CACHE_PATH, "file_metadata.pkl")

# --- Metadaten: Prüfen, ob sich Dateien verändert haben ---
def get_file_metadata():
    metadata = {}
    for root, _, files in os.walk(DOCS_PATH):
        for f in files:
            path = os.path.join(root, f)
            metadata[path] = os.path.getmtime(path)
    return metadata

def is_cache_valid():
    if not os.path.exists(METADATA_FILE):
        return False
    with open(METADATA_FILE, "rb") as f:
        old_meta = pickle.load(f)
    current_meta = get_file_metadata()
    return old_meta == current_meta

def save_file_metadata():
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(get_file_metadata(), f)

# --- Load Documents ---
def load_documents():
    docs = []
    categories = [d for d in os.listdir(DOCS_PATH) if os.path.isdir(os.path.join(DOCS_PATH, d))]
    total_files = sum(len(os.listdir(os.path.join(DOCS_PATH, c))) for c in categories)
    progress = st.progress(0, text="📚 Lade Dokumente...")
    processed_files = 0

    for category in categories:
        category_path = os.path.join(DOCS_PATH, category)
        files = os.listdir(category_path)

        for filename in files:
            filepath = os.path.join(category_path, filename)
            loaded = []
            try:
                if filename.endswith((".txt", ".md")):
                    loader = TextLoader(filepath)
                    loaded = loader.load()
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                    loaded = loader.load()
                elif filename.endswith(".xml"):
                    tree = ET.parse(filepath)
                    root = tree.getroot()
                    text = ET.tostring(root, encoding="unicode", method="text")
                    loaded = [Document(page_content=text)]

                for doc in loaded:
                    doc.metadata["source"] = filename
                    doc.metadata["category"] = category
                    docs.append(doc)

            except Exception as e:
                st.warning(f"⚠️ Fehler beim Laden von {filename}: {e}")

            processed_files += 1
            progress.progress(processed_files / total_files, text=f"📄 {filename} geladen ({processed_files}/{total_files})")

    progress.progress(1.0, text="✅ Alle Dokumente geladen.")
    logging.info(f"{len(docs)} Dokumente wurden erfolgreich geladen.")
    return docs

# --- Vektorstore erstellen ---
def create_vectorstore(docs):
    logging.info("Erstelle Vektorstore aus den geladenen Dokumenten...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="llm_models/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(CACHE_PATH)
    save_file_metadata()
    logging.info("Vektorstore wurde gespeichert.")
    return vectorstore

# --- Vektorstore laden oder neu erstellen ---
def load_or_create_vectorstore():
    if is_cache_valid():
        logging.info("Lade bestehenden Vektorstore (Cache ist gültig).")
        return FAISS.load_local(
            CACHE_PATH,
            HuggingFaceEmbeddings(model_name="llm_models/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
    else:
        logging.info("Kein gültiger Cache gefunden. Dokumente werden neu geladen und Vektorstore wird erstellt.")
        docs = load_documents()
        return create_vectorstore(docs)

# --- Dokumentensuche im Vektorstore ---
def retrieve_docs_with_sources(vstore, query, k=4, category=None):
    results = vstore.similarity_search(query, k=50)
    if HUMAN_READABLE_LOGS:
        logging.info(f"Suche nach '{query}' ergab {len(results)} ungefilterte Dokumente.")
    else:
        print(f"[RAG] Gefundene Dokumente ohne Filter: {len(results)}")

    if category:
        results = [doc for doc in results if doc.metadata.get("category") == category]
        if HUMAN_READABLE_LOGS:
            logging.info(f"{len(results[:k])} Dokumente passen zur Kategorie '{category}'.")
        else:
            print(f"[RAG] Gefundene Dokumente nach Filter: {len(results[:k])}")
    else:
        if not HUMAN_READABLE_LOGS:
            print(f"[RAG] Gefundene Dokumente ohne Filter: {len(results)}")

    return [
        (doc.page_content, doc.metadata.get("source", "unbekannt"), doc.metadata.get("category", "unbekannt"))
        for doc in results[:k]
    ]