import os
import logging
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiLLMService:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in your .env file."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )

        logger.info(f"GeminiLLMService initialized with model: {model_name}")

    def generate_response(self, query: str, context: list[Document]) -> str:
        context_text = "\n\n".join(doc.page_content for doc in context)

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
            return "⚠️ Gemini service unavailable or quota exceeded. Please try again later."
