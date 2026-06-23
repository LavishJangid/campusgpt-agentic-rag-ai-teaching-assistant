"""Additional unit tests for coverage."""

from unittest.mock import MagicMock, patch


def test_reranker_with_mock_model(monkeypatch):
    monkeypatch.setenv("RERANKER_DISABLED", "false")
    from backend.config import get_settings

    get_settings.cache_clear()

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.3]

    with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
        from backend.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        docs = [
            {"text": "ML basics", "source": "a.pdf"},
            {"text": "DBMS notes", "source": "b.pdf"},
        ]
        result = reranker.rerank("machine learning", docs, top_k=1)
        assert len(result) == 1
        assert "rerank_score" in result[0]

    get_settings.cache_clear()


def test_ragas_heuristic_evaluation():
    from backend.evaluation.ragas_eval import RAGASEvaluator

    ev = RAGASEvaluator()
    ev._available = False
    scores = ev.evaluate(
        question="What is data science?",
        answer="Data science combines statistics and programming.",
        contexts=["Data science is an interdisciplinary field."],
    )
    assert scores["method"] == "heuristic"
    assert 0 <= scores["faithfulness"] <= 1


def test_engine_retrieve_with_rerank():
    from backend.rag.engine import RAGEngine

    with patch("langchain_google_genai.ChatGoogleGenerativeAI"):
        with patch("backend.ingestion.embeddings.HuggingFaceEmbeddings"):
            with patch("chromadb.PersistentClient"):
                engine = RAGEngine()
                engine.reranker.rerank = MagicMock(
                    return_value=[{"source": "t.pdf", "page_number": 1, "text": "x", "similarity_score": 0.8, "metadata": {}}]
                )
                engine.vector_store.hybrid_search = MagicMock(
                    return_value=[{"source": "t.pdf", "page_number": 1, "text": "x", "similarity_score": 0.8, "metadata": {}}]
                )
                docs = engine._retrieve_with_rerank("test query", subject="ML")
                assert len(docs) == 1


def test_redis_memory_cache():
    from backend.memory import ConversationMemory

    mem = ConversationMemory()
    mem.set_cache("s1", {"key": "value"})
    assert mem.get_cache("s1") is None  # in-memory mode returns None
