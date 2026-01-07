# rag_pipeline.py
from retrieval.loader import load_documents
from retrieval.faiss_store import build_faiss_index, search
from llm.llm_service import GeminiLLMService
from langchain_core.documents import Document

# =====================================================
# LOAD DOCUMENTS
# =====================================================
print("🔄 Loading documents...")
docs = load_documents("data")
print(f"✅ Loaded {len(docs)} document chunks")

# =====================================================
# BUILD FAISS INDEX
# =====================================================
print("🔄 Building FAISS index...")
# Use test_limit=5 to avoid API quota issues during testing
build_faiss_index(docs, test_limit=5)
print("✅ FAISS index ready")

# =====================================================
# INITIALIZE LLM SERVICE
# =====================================================
llm_service = GeminiLLMService()

# =====================================================
# RAG ANSWER FUNCTION
# =====================================================
def rag_answer(query: str) -> str:
    """
    Get a RAG-based answer for the query
    """
    # Step 1: Search relevant documents
    results = search(query, top_k=3)
    if not results:
        return "Sorry, no relevant information found."

    # Step 2: Prepare context
    context_docs = []
    context_text = ""
    for doc in results:
        context_text += "\n\n" + doc["text"]
        context_docs.append(Document(page_content=doc["text"], metadata=doc.get("metadata")))

    # Step 3: Generate answer using LLM
    response = llm_service.generate_response(query=query, context=context_docs)
    return response["response"]
