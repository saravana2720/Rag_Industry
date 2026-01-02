import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiLLMService:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3,
        )
        logger.info(f"GeminiLLMService initialized with model: {model_name}")

    def generate_response(self, query: str, context: list[Document]) -> str:
        context_text = "\n\n".join([doc.page_content for doc in context])

        prompt = f"""
You are a helpful career and technology assistant.

Use the context below if relevant.
If not relevant, answer from general knowledge.

Context:
{context_text}

Question:
{query}

Answer clearly and concisely:
"""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return "⚠️ Gemini quota exceeded or unavailable. Please try again later."
