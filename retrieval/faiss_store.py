import os
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CACHE_PATH = "retrieval/cache/embeddings.npy"

# ✅ LOCAL embedding model (NO API, NO QUOTA)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

_vectorstore = None


def build_faiss_index(documents):
    global _vectorstore

    texts = [doc.page_content for doc in documents]

    # Load cached embeddings if available
    if os.path.exists(CACHE_PATH):
        embeddings = np.load(CACHE_PATH)
        print(f"💾 Loaded cached embeddings from {CACHE_PATH}")
    else:
        embeddings = embedding_model.embed_documents(texts)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.save(CACHE_PATH, embeddings)
        print(f"💾 Cached {len(embeddings)} embeddings to {CACHE_PATH}")

    # Build FAISS index
    _vectorstore = FAISS.from_embeddings(
        list(zip(texts, embeddings)),
        embedding_model
    )

    print(f"✅ FAISS index built with {len(texts)} vectors")
    return _vectorstore


def search(query, k=3):
    if _vectorstore is None:
        raise RuntimeError("FAISS index not built")

    return _vectorstore.similarity_search(query, k=k)
