import os
import xml.etree.ElementTree as ET
import pickle
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DOCS_PATH = "rag/docs"
CACHE_PATH = "rag/index"
METADATA_FILE = os.path.join(CACHE_PATH, "file_metadata.pkl")


def get_file_metadata():
    metadata = {}
    for root, dirs, files in os.walk(DOCS_PATH):
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


def load_documents():
    docs = []
    categories = [d for d in os.listdir(DOCS_PATH) if os.path.isdir(os.path.join(DOCS_PATH, d))]
    total_files = sum(len(os.listdir(os.path.join(DOCS_PATH, c))) for c in categories)
    progress = st.progress(0, text="Dokumente laden...")
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
                st.warning(f"Fehler beim Laden von {filename}: {e}")

            processed_files += 1
            progress.progress(processed_files / total_files, text=f"📚 Lade: {filename} ({processed_files}/{total_files})")

    return docs


def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(CACHE_PATH)
    save_file_metadata()
    return vectorstore


def load_or_create_vectorstore():
    if is_cache_valid():
        return FAISS.load_local(
            CACHE_PATH,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
    else:
        docs = load_documents()
        return create_vectorstore(docs)


def retrieve_docs_with_sources(vstore, query, k=4, category=None):
    results = vstore.similarity_search(query, k=20)
    if category:
        results = [doc for doc in results if doc.metadata.get("category") == category]
        print(f"[RAG] Gefundene Dokumente nach Filter: {len(results[:k])}")

    return [(doc.page_content, doc.metadata.get("source", "unbekannt"), doc.metadata.get("category", "unbekannt")) for doc in results[:k]]


def get_last_modified():
    return max(
        os.path.getmtime(os.path.join(root, f))
        for root, dirs, files in os.walk(DOCS_PATH)
        for f in files
    )