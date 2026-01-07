# retrieval/loader.py

import os
from typing import List
from langchain_core.documents import Document


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def load_documents(data_dir: str = "data") -> List[Document]:
    documents: List[Document] = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 🔥 RECURSIVE WALK
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = chunk_text(text)

                for chunk in chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file_path.replace(data_dir, "")
                            }
                        )
                    )

    print(f"✅ Loaded {len(documents)} document chunks")
    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(docs[:2])
