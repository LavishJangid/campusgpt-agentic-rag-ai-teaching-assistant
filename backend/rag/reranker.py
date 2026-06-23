"""CrossEncoder reranker for improved RAG retrieval."""

from typing import Optional

from loguru import logger

from backend.config import get_settings


class CrossEncoderReranker:
    """Rerank retrieved chunks using BAAI/bge-reranker-base."""

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.reranker_model
        self._model = None
        self._disabled = self.settings.reranker_disabled

    @property
    def model(self):
        if self._disabled:
            return None
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading reranker: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
                logger.info("Reranker loaded successfully")
            except Exception as e:
                logger.warning(f"Reranker unavailable: {e}")
                self._disabled = True
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """Score query-document pairs and return top_k reranked docs."""
        if not documents:
            return []

        if self._disabled or self.model is None:
            return documents[:top_k]

        pairs = [(query, doc.get("text", "")) for doc in documents]
        try:
            scores = self.model.predict(pairs)
            ranked = sorted(
                zip(documents, scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            results = []
            for doc, score in ranked[:top_k]:
                doc = dict(doc)
                doc["rerank_score"] = round(float(score), 4)
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
