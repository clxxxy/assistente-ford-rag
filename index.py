import os, shutil, hashlib
from datetime import datetime, timezone
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =============================
# pastas e parâmetros fixos
# =============================

VECTOR_ROOT = "vector_stores"
UPLOAD_ROOT = "uploads"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
os.makedirs(VECTOR_ROOT, exist_ok=True)
os.makedirs(UPLOAD_ROOT, exist_ok=True)


# =============================
# funções auxiliares
# =============================

# --- simplificação inicial do nome do arquivo ---
def _sanitize_filename(name: str) -> str:
    keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in name)

# --- id do documento ---
def _doc_id_for_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:12]

# --- salva arquivo enviado pelo usuário ---
def _save_upload(file_bytes: bytes, filename: str) -> str:
    safe = _sanitize_filename(filename)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    path = os.path.join(UPLOAD_ROOT, f"{ts}-{safe}")
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


# =============================
# indexação de documentos
# =============================

# --- indexação geral do PDF ---
def _index_pdf(pdf_path: str, persist_dir: str, collection: str, emb_model_name: str) -> int:
    docs = PyPDFLoader(pdf_path).load()

    # --- divisão em chunks ---
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    ).split_documents(docs)

    # --- embeddings ---
    embed = HuggingFaceEmbeddings(model_name=emb_model_name, model_kwargs={"device": "cpu"})
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    # --- criação do vector store ---
    Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        persist_directory=persist_dir,
        collection_name=collection,
    )
    return len(chunks)

# --- indexação a partir do upload ---
def index_from_upload(file_bytes: bytes, filename: str, emb_model_name: str) -> Dict[str, str]:
    doc_id = _doc_id_for_bytes(file_bytes)
    pdf_path = _save_upload(file_bytes, filename)
    persist_dir = os.path.join(VECTOR_ROOT, doc_id)
    collection = f"manual_{doc_id}"
    _index_pdf(pdf_path, persist_dir, collection, emb_model_name)
    return {
        "pdf_path": pdf_path,
        "persist_directory": persist_dir,
        "collection_name": collection,
        "embedding_model": emb_model_name,
    }

# --- carregamento do vector store ---
def load_vector_store(persist_dir: str, collection_name: str, emb_model_name: str):
    embed = HuggingFaceEmbeddings(model_name=emb_model_name, model_kwargs={"device": "cpu"})
    
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embed,
        collection_name=collection_name,
    )
