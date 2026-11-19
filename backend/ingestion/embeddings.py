"""Embedding backends with a shared LangChain-compatible interface.

Local sentence-transformers support is imported only when selected. Gemini is
useful on memory-constrained hosts, but uses a separate Chroma collection from
the legacy MiniLM collection to prevent incompatible vectors from mixing.
"""

import re
from typing import Any

from loguru import logger

from backend.config import get_settings

# Kept as a lazy sentinel so existing tests/integrations can patch this symbol
# without importing sentence-transformers into the default process.
HuggingFaceEmbeddings: Any = None


def collection_name_for_embedding(base_name: str, provider: str, model_name: str) -> str:
    """Use the legacy name only for the original MiniLM vector space."""
    provider = provider.lower()
    legacy = (
        provider == "local"
        and model_name == "sentence-transformers/all-MiniLM-L6-v2"
    )
    if legacy:
        return base_name
    suffix = re.sub(r"[^a-zA-Z0-9]+", "_", f"{provider}:{model_name}").strip("_")
    return f"{base_name}__{suffix}"[:512]


class EmbeddingGenerator:
    """Generate normalized embeddings locally or through Gemini."""

    def __init__(self, model_name: str | None = None, provider: str | None = None):
        self.settings = get_settings()
        self.provider = (provider or self.settings.embedding_provider).lower()
        if self.provider not in {"local", "gemini"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'gemini'")
        self.model_name = model_name or (
            self.settings.gemini_embedding_model
            if self.provider == "gemini"
            else self.settings.embedding_model
        )
        self._embeddings: Any = None
        self._google_genai: Any = None
        logger.info("Embedding generator configured | provider={}", self.provider)

    @property
    def embeddings(self):
        """Return the local LangChain embedding adapter, loading it on demand."""
        if self.provider != "local":
            raise RuntimeError("Gemini embeddings do not use a local embeddings adapter")
        if self._embeddings is None:
            logger.info("Loading local embedding model: {}", self.model_name)
            embedding_factory = HuggingFaceEmbeddings
            if embedding_factory is None:
                from langchain_community.embeddings import HuggingFaceEmbeddings as embedding_factory

            self._embeddings = embedding_factory(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
            )
            logger.info("Local embedding model loaded")
        return self._embeddings

    @property
    def collection_model(self) -> str:
        """Stable provider/model identity used to isolate vector dimensions."""
        return f"{self.provider}:{self.model_name}"

    def collection_name(self, base_name: str) -> str:
        """Keep the existing collection for legacy MiniLM; isolate other models."""
        return collection_name_for_embedding(
            base_name, self.provider, self.model_name
        )

    def _gemini_embed(self, text: str, task_type: str) -> list[float]:
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings")
        if self._google_genai is None:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            self._google_genai = genai
        response = self._google_genai.embed_content(
            model=self.model_name,
            content=text,
            task_type=task_type,
        )
        return response["embedding"]

    def embed_text(self, text: str) -> list[float]:
        if self.provider == "gemini":
            return self._gemini_embed(text, "retrieval_query")
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info("Embedding {} texts", len(texts))
        if self.provider == "gemini":
            # Batch sequentially to stay within the API's per-request limits and
            # keep peak memory low. Gemini's output dimensions remain model-defined.
            return [self._gemini_embed(text, "retrieval_document") for text in texts]
        return self.embeddings.embed_documents(texts)

    def get_langchain_embeddings(self):
        """Return the native local adapter (Gemini is used through this interface)."""
        return self.embeddings
