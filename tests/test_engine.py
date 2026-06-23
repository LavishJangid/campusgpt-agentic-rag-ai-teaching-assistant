"""Direct RAG engine mode tests."""

from unittest.mock import MagicMock, patch


def _make_engine():
    with patch("langchain_google_genai.ChatGoogleGenerativeAI"):
        with patch("backend.ingestion.embeddings.HuggingFaceEmbeddings"):
            with patch("chromadb.PersistentClient"):
                from backend.rag.engine import RAGEngine

                engine = RAGEngine()
                engine.llm.invoke = MagicMock(
                    return_value=MagicMock(content="Generated educational content.")
                )
                engine.vector_store.hybrid_search = MagicMock(
                    return_value=[
                        {
                            "source": "notes.pdf",
                            "page_number": 2,
                            "text": "Sample context text.",
                            "similarity_score": 0.88,
                            "metadata": {"subject": "ML"},
                        }
                    ]
                )
                return engine


def test_engine_exam_preparation():
    engine = _make_engine()
    result = engine.exam_preparation(subject="ML", topic="Supervised Learning")
    assert "content" in result
    assert result["subject"] == "ML"


def test_engine_generate_quiz():
    engine = _make_engine()
    result = engine.generate_quiz(subject="DS", topic="Regression", num_questions=5)
    assert "quiz" in result


def test_engine_generate_viva():
    engine = _make_engine()
    result = engine.generate_viva_questions(subject="AI", num_questions=5)
    assert "questions" in result


def test_engine_assignment_help():
    engine = _make_engine()
    result = engine.assignment_help("Explain gradient descent", subject="ML")
    assert "answer" in result


def test_engine_find_important_questions():
    engine = _make_engine()
    result = engine.find_important_questions(subject="DBMS", unit="3")
    assert "questions" in result


def test_engine_search_helpers():
    engine = _make_engine()
    engine.vector_store.similarity_search = MagicMock(return_value=[])
    engine.vector_store.hybrid_search = MagicMock(return_value=[])
    assert engine.search_by_topic("SQL") == []
    assert engine.search_by_subject("DBMS") == []
    assert engine.search_by_semester("5") == []
