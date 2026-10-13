"""RAG engine: load documents, index them in Chroma, answer questions with sources."""
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TOP_K = 4

PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: I don't know based on the provided documents.
Keep the answer short and clear.

Context:
{context}

Question: {question}

Answer:"""

_embeddings = None
_vectorstore = None
_llm = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def load_documents():
    docs = []
    for path in sorted(DATA_DIR.glob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in (".txt", ".md"):
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            continue
        for doc in loader.load():
            doc.metadata["source"] = path.name
            docs.append(doc)
    return docs


def build_index():
    """Read everything in data/, split it, and store it in Chroma. Returns (docs, chunks)."""
    docs = load_documents()
    if not docs:
        raise ValueError("No .txt, .md or .pdf files found in the data/ folder.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    if DB_DIR.exists():
        try:
            Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings()).delete_collection()
        except Exception:
            pass
    global _vectorstore
    _vectorstore = None
    Chroma.from_documents(chunks, get_embeddings(), persist_directory=str(DB_DIR))
    return len(docs), len(chunks)


def get_vectorstore():
    global _vectorstore
    if not DB_DIR.exists():
        build_index()
    if _vectorstore is None:
        _vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())
    return _vectorstore


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return _llm


def answer_question(question, k=TOP_K):
    """Return the answer, the sources used, the raw retrieved chunks and timings."""
    t0 = time.time()
    docs = get_vectorstore().similarity_search(question, k=k)
    t1 = time.time()
    context = "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
    reply = get_llm().invoke(PROMPT.format(context=context, question=question))
    t2 = time.time()

    sources = []
    for d in docs:
        page = d.metadata.get("page")
        sources.append(
            {
                "file": d.metadata.get("source", "unknown"),
                "page": None if page is None else int(page) + 1,
                "snippet": d.page_content[:300],
            }
        )
    return {
        "answer": reply.content.strip(),
        "sources": sources,
        "chunks": [d.page_content for d in docs],
        "timing": {"search": round(t1 - t0, 2), "llm": round(t2 - t1, 2)},
    }