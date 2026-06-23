"""Tests for citation formatting and reranker."""

from backend.rag.citations import format_source_citation, format_sources_list
from backend.rag.reranker import CrossEncoderReranker


def test_format_source_citation():
    doc = {
        "source": "DBMS Unit 3.pdf",
        "page_number": 17,
        "similarity_score": 0.85,
        "text": "Normalization reduces redundancy in databases.",
        "metadata": {
            "subject": "DBMS",
            "semester": "5",
            "topic": "Normalization",
        },
    }
    citation = format_source_citation(doc)
    assert citation["source"] == "DBMS Unit 3.pdf"
    assert citation["page_number"] == 17
    assert citation["subject"] == "DBMS"
    assert "DBMS Unit 3.pdf" in citation["citation_label"]
    assert "Page 17" in citation["citation_label"]


def test_format_sources_list():
    docs = [
        {
            "source": "notes.pdf",
            "page_number": 2,
            "similarity_score": 0.9,
            "text": "Sample",
            "metadata": {"subject": "AI"},
        }
    ]
    result = format_sources_list(docs)
    assert len(result) == 1
    assert result[0]["citation_label"]


def test_reranker_fallback_without_model(monkeypatch):
    monkeypatch.setenv("RERANKER_DISABLED", "true")
    from backend.config import get_settings

    get_settings.cache_clear()
    reranker = CrossEncoderReranker()
    docs = [
        {"text": "Machine learning basics", "source": "ml.pdf", "page_number": 1},
        {"text": "Database systems", "source": "db.pdf", "page_number": 3},
    ]
    result = reranker.rerank("machine learning", docs, top_k=1)
    assert len(result) == 1
    get_settings.cache_clear()
