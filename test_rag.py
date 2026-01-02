from retrieval.loader import load_documents
from retrieval.faiss_store import build_faiss_index, search
from llm.llm_service import GeminiLLMService

print("🔄 Loading documents...")
documents = load_documents()
print(f"✅ Loaded {len(documents)} document chunks")

print("🔄 Building FAISS index...")
build_faiss_index(documents)
print("✅ FAISS index ready\n")

llm_service = GeminiLLMService()

print("💬 Ask a career or tech question (type 'exit' to quit):\n")

while True:
    query = input("> ")
    if query.lower() == "exit":
        break

    docs = search(query, k=3)
    answer = llm_service.generate_response(query=query, context=docs)

    print("\n🤖 Answer:\n", answer, "\n")
