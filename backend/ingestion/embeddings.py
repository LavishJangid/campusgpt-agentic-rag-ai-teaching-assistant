"""Embedding generator using sentence-transformers."""

from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger


class EmbeddingGenerator:
    """
    Generate embeddings using sentence-transformers/all-MiniLM-L6-v2.
    Provides both single and batch embedding generation.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings = None
        logger.info(f"Embedding generator created | model={model_name}")

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load the embedding model."""
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
            logger.info("Embedding model loaded successfully")
        return self._embeddings

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        logger.info(f"Embedding {len(texts)} texts...")
        return self.embeddings.embed_documents(texts)

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Return LangChain-compatible embedding object."""
        return self.embeddings
